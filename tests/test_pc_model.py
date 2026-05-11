"""M3 unit tests for PredictiveCodingLM and the checkpoint-trigger logic.

Per `docs/V2_IMPLEMENTATION_PLAN.md` M3 (Days 6-8).

The full M3 gate ("training converges by epoch 59") is a GPU run blocked
on the v1 ablation pipeline. These unit tests exercise the CPU-side
correctness of the model and the trigger logic itself.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from luthi.v2 import PredictiveCodingLM
from luthi.v2.train_pc import (
    evaluate_checkpoint_trigger,
    emit_grid_search_marker,
    measure_prediction_frob_norm,
)


def _make_model(
    vocab: int = 32, d_model: int = 16, n_blocks: int = 2,
    backward_pass_enabled: bool = False,
) -> PredictiveCodingLM:
    torch.manual_seed(0)
    return PredictiveCodingLM(
        vocab_size=vocab,
        d_model=d_model,
        n_blocks=n_blocks,
        max_seq_len=32,
        backward_pass_enabled=backward_pass_enabled,
    )


# ---------------------------------------------------------------------------
# Model smoke tests
# ---------------------------------------------------------------------------

def test_forward_returns_correct_shape():
    model = _make_model()
    x = torch.randint(0, 32, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 32)
    assert not torch.isnan(logits).any()


def test_parameter_counts_are_additive():
    model = _make_model()
    counts = model.total_parameters()
    assert counts["trainable"] > 0
    assert counts["living_buffers"] > 0
    assert counts["total"] == counts["trainable"] + counts["living_buffers"]


def test_non_feedforward_signal_is_positive():
    """Two consecutive identical-input passes through the full model
    differ — confirms the model as a whole is non-feedforward via the
    PC layers' intrinsic self-modification."""
    model = _make_model()
    x = torch.randint(0, 32, (2, 16))
    nff = model.non_feedforward_signal(x)
    assert nff > 0.0


# ---------------------------------------------------------------------------
# Training behavior
# ---------------------------------------------------------------------------

def test_training_step_decreases_loss():
    """Overfit on a tiny synthetic corpus — loss should drop noticeably
    over a handful of optimizer steps. Verifies the trainable parts of
    the model (embeddings, attention, output_proj) are wired correctly
    and that PC self-modification doesn't break gradient flow.
    """
    torch.manual_seed(0)
    vocab = 16
    model = _make_model(vocab=vocab)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
    )

    x = torch.randint(0, vocab, (4, 8))
    y = torch.randint(0, vocab, (4, 8))

    initial_loss = None
    final_loss = None
    for step in range(50):
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1)
        )
        if step == 0:
            initial_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.9, (
        f"Training loss did not decrease meaningfully: "
        f"{initial_loss:.4f} -> {final_loss:.4f}"
    )


def test_top_down_sweep_runs_through_stacked_blocks():
    """With backward_pass_enabled=True and training mode, the top-down
    sweep should run without errors and propagate modulation to the
    lower block (its plasticity should drift from the uniform init).
    """
    torch.manual_seed(0)
    model = _make_model(n_blocks=3, backward_pass_enabled=True)
    model.train()

    initial_lower_plasticity = model.blocks[0].living_ffn.plasticity.clone()

    x = torch.randint(0, 32, (2, 16))
    for _ in range(30):
        _ = model(x)

    final_lower_plasticity = model.blocks[0].living_ffn.plasticity
    drift = (final_lower_plasticity - initial_lower_plasticity).abs().mean()
    assert drift > 0.0, (
        f"Top-down sweep did not modulate the lowest block's plasticity; "
        f"signal did not reach it (drift={drift.item():.6f})"
    )


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------

def test_state_dict_roundtrip_preserves_pc_buffers(tmp_path: Path):
    """save → load round-trip preserves all PC living-state buffers
    (weight, prediction, set_point, momentum, update_ema, precision,
    error_acc, plasticity) — not just the trainable parameters.
    """
    torch.manual_seed(0)
    src = _make_model()
    # Run a few steps so the buffers diverge from init.
    x = torch.randint(0, 32, (2, 16))
    for _ in range(5):
        src(x)

    ckpt_path = tmp_path / "model.pt"
    torch.save(src.state_dict(), ckpt_path)

    torch.manual_seed(99)  # Different seed for fresh model
    dst = _make_model()
    dst.load_state_dict(torch.load(ckpt_path, weights_only=True))

    src_block_ffn = src.blocks[0].living_ffn
    dst_block_ffn = dst.blocks[0].living_ffn

    for name in [
        "weight", "prediction", "set_point", "momentum",
        "update_ema", "precision", "error_acc", "plasticity",
    ]:
        src_buf = getattr(src_block_ffn, name)
        dst_buf = getattr(dst_block_ffn, name)
        assert torch.allclose(src_buf, dst_buf), (
            f"PC buffer {name!r} did not round-trip via state_dict"
        )


# ---------------------------------------------------------------------------
# Checkpoint trigger logic (refinement 1)
# ---------------------------------------------------------------------------

def test_checkpoint_trigger_healthy_run_passes():
    """Healthy 10-epoch metrics — train loss decreasing, val not diverging,
    NFF above threshold, prediction norm bounded, no NaNs — return healthy."""
    diagnostic = evaluate_checkpoint_trigger(
        train_losses=[5.0, 4.7, 4.5, 4.3, 4.2, 4.1, 4.0, 3.95, 3.9, 3.85],
        val_losses=[5.1, 4.8, 4.6, 4.5, 4.4, 4.3, 4.25, 4.2, 4.15, 4.1],
        nff_signal=0.05,
        prediction_frob_norms=[0.1, 0.5, 1.2, 2.0, 2.5, 2.7, 2.8, 2.9, 3.0, 3.1],
        nan_events=0,
    )
    assert diagnostic["healthy"]
    assert all(diagnostic["checks"].values())


def test_checkpoint_trigger_train_plateau_fails():
    """Train loss plateaued or diverged — trigger flags as unhealthy."""
    diagnostic = evaluate_checkpoint_trigger(
        train_losses=[5.0, 5.0, 5.0, 5.1, 5.0, 5.0, 5.0, 5.0, 5.1, 5.2],
        val_losses=[5.1, 5.1, 5.1, 5.1, 5.1, 5.1, 5.1, 5.1, 5.1, 5.1],
        nff_signal=0.05,
        prediction_frob_norms=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        nan_events=0,
    )
    assert not diagnostic["healthy"]
    assert not diagnostic["checks"]["train_loss_decreased"]


def test_checkpoint_trigger_nff_collapse_fails():
    """NFF below threshold — trigger flags as unhealthy."""
    diagnostic = evaluate_checkpoint_trigger(
        train_losses=[5.0, 4.7, 4.5, 4.3, 4.2, 4.1, 4.0, 3.95, 3.9, 3.85],
        val_losses=[5.1, 4.8, 4.6, 4.5, 4.4, 4.3, 4.25, 4.2, 4.15, 4.1],
        nff_signal=0.001,  # Below default threshold 0.01
        prediction_frob_norms=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        nan_events=0,
    )
    assert not diagnostic["healthy"]
    assert not diagnostic["checks"]["nff_above_threshold"]


def test_checkpoint_trigger_nan_event_fails():
    """Any NaN event during training — trigger flags as unhealthy."""
    diagnostic = evaluate_checkpoint_trigger(
        train_losses=[5.0, 4.7, 4.5, 4.3, 4.2, 4.1, 4.0, 3.95, 3.9, 3.85],
        val_losses=[5.1, 4.8, 4.6, 4.5, 4.4, 4.3, 4.25, 4.2, 4.15, 4.1],
        nff_signal=0.05,
        prediction_frob_norms=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        nan_events=3,
    )
    assert not diagnostic["healthy"]
    assert not diagnostic["checks"]["no_nan_events"]


def test_checkpoint_trigger_prediction_runaway_fails():
    """Prediction Frobenius norm ratio > 100 — trigger flags as unhealthy."""
    diagnostic = evaluate_checkpoint_trigger(
        train_losses=[5.0, 4.7, 4.5, 4.3, 4.2, 4.1, 4.0, 3.95, 3.9, 3.85],
        val_losses=[5.1, 4.8, 4.6, 4.5, 4.4, 4.3, 4.25, 4.2, 4.15, 4.1],
        nff_signal=0.05,
        prediction_frob_norms=[0.1, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 5000.0, 10000.0],
        nan_events=0,
    )
    assert not diagnostic["healthy"]
    assert not diagnostic["checks"]["prediction_growth_bounded"]


def test_emit_grid_search_marker_writes_marker(tmp_path: Path):
    """When the trigger fails, the grid-search marker file is written
    with the diagnostic, current HPs, recommended grid, and the command
    template a human or 4.6 can run."""
    diagnostic = {
        "healthy": False,
        "checks": {"train_loss_decreased": False},
        "metrics": {},
    }
    args = SimpleNamespace(
        pc_rate=0.001, pred_learning_rate=0.0001,
        data_dir="corpus_build/gutenberg_100",
        tokenizer="bpe", load_tokenizer="corpus_build/gutenberg_100_bpe32k.json",
        d_model=256, n_blocks=2, seq_len=256, batch_size=16, stride=64,
        epochs=59,
    )
    marker_path = emit_grid_search_marker(tmp_path, diagnostic, args)
    assert marker_path.exists()
    payload = json.loads(marker_path.read_text())
    assert payload["diagnostic"] == diagnostic
    assert payload["current_hp"]["pc_rate"] == 0.001
    assert payload["recommended_grid"]["pc_rate"] == [1e-4, 5e-4, 1e-3, 2e-3]
    assert payload["recommended_grid"]["pred_learning_rate"] == [5e-5, 1e-4, 5e-4]


def test_kv_cache_smoke():
    """KV cache runs cleanly and produces valid logits.

    Note: exact equivalence between cached and non-cached paths does NOT
    hold for v2's PC model because the PC layer is stateful — its episodic
    recall context and self-modification depend on the input batch each
    forward. Single-token incremental forwards see different recall
    contexts than a single full forward. The KV cache is still
    mathematically correct for the *attention* path (the reason the audit
    flagged the O(S²) cost), it just doesn't reproduce the stateful PC
    dynamics bit-for-bit. This test verifies the cache plumbing runs
    end-to-end and produces well-formed output; downstream we accept that
    generation with cache is a different living-inference trajectory, not
    a replay of the equivalent non-cached trajectory.
    """
    torch.manual_seed(0)
    model = _make_model(vocab=16, d_model=16, n_blocks=2)
    model.eval()

    prompt = torch.randint(0, 16, (1, 6))

    with torch.no_grad():
        # Step 0: full prompt, initialize cache.
        logits_step0, caches = model(prompt, return_kv_caches=True)
        assert logits_step0.shape == (1, 6, 16)
        assert len(caches) == 2  # 2 blocks
        # Each cache is (K, V), both [B, H, S=6, D]
        for K, V in caches:
            assert K.shape[-2] == 6 and V.shape[-2] == 6

        # Step 1: single new token, extend the cache.
        next_token = torch.tensor([[7]], dtype=torch.long)
        logits_step1, caches = model(
            next_token, kv_caches=caches, return_kv_caches=True,
        )
        assert logits_step1.shape == (1, 1, 16)
        # Cache should have grown by 1.
        for K, V in caches:
            assert K.shape[-2] == 7 and V.shape[-2] == 7
        assert not torch.isnan(logits_step1).any()


def test_measure_prediction_frob_norm_returns_mean_across_blocks():
    """Helper used by the trigger — mean Frobenius norm across all PC blocks."""
    model = _make_model(n_blocks=3)
    norm = measure_prediction_frob_norm(model)
    # Prediction is initialized to zeros, so norm == 0.0 at start.
    assert norm == 0.0
    # Drive the model briefly so prediction matrices grow.
    x = torch.randint(0, 32, (2, 16))
    for _ in range(10):
        model(x)
    norm_after = measure_prediction_frob_norm(model)
    assert norm_after > 0.0
