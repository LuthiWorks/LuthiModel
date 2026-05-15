@echo off
REM Phase 3G ablation #3: Sparse PC gating threshold sweep at 256d/2 blocks.
REM
REM Source: pattern-matched off v1's spiking gate, with SpikingBrain 1.0
REM (Aug 2025) showing 70-90%% activation sparsity in trained SNN-style
REM language models without quality loss. See
REM docs/RESEARCH_LITERATURE_2026-05-13.md.
REM
REM Goal: measure (a) gate-on rate post-warmup, (b) convergence delta vs
REM no-gating baseline. Target: >=50%% of PC rows gated off after warmup
REM with <5%% val loss penalty.
REM
REM NOTE: when --sparse-threshold > 0, pc_self_modify falls through to
REM the Python path (C++ kernel skips sparse_gate). Expect ~3-5x slower
REM per-step. Documented in docs/KNOWN_INCOMPLETE.md; the Triton kernel
REM landing would unblock the C++ fast path.
REM
REM Estimated wall-clock: ~16h sequential for 4 thresholds at 30 epochs
REM each. Reduce to --epochs 10 if you just want the gate-rate signal
REM rather than full convergence.

cd /d "C:\Users\Hasha Smokes\Desktop\LuthiModel\LuthiModel"

echo ============================================================
echo Phase 3G ablation: sparse-gating threshold sweep at 256d/2 blocks
echo Started: %DATE% %TIME%
echo ============================================================

set COMMON_ARGS=--data_dir corpus_build/gutenberg_100 --load_tokenizer corpus_build/gutenberg_100_bpe32k.json --epochs 30 --batch_size 32 --seq_len 128 --d_model 256 --n_blocks 2 --n_heads 4 --ffn_expansion 1 --stride 64 --lr 3e-4 --lr_schedule cosine --lr_warmup_epochs 2 --seed 42 --output_dir runs/phase3g_sparse

echo.
echo === sparse threshold = 0.0 (no gating baseline) ===
python -u -m luthi.v2.m5_runner --arch v2 --sparse-threshold 0.0 %COMMON_ARGS% --run_name v2_seed42_thr0 && ^
echo. && echo === sparse threshold = 0.01 === && ^
python -u -m luthi.v2.m5_runner --arch v2 --sparse-threshold 0.01 %COMMON_ARGS% --run_name v2_seed42_thr0p01 && ^
echo. && echo === sparse threshold = 0.05 === && ^
python -u -m luthi.v2.m5_runner --arch v2 --sparse-threshold 0.05 %COMMON_ARGS% --run_name v2_seed42_thr0p05 && ^
echo. && echo === sparse threshold = 0.1 === && ^
python -u -m luthi.v2.m5_runner --arch v2 --sparse-threshold 0.1 %COMMON_ARGS% --run_name v2_seed42_thr0p1

echo.
echo ============================================================
echo Phase 3G sparse-gating sweep finished: %DATE% %TIME%
echo Falsifier per To-Do.md: cannot achieve >=50%% sparsity post-warmup
echo with <10%% val loss penalty, OR creates dead-output collapse.
echo ============================================================
