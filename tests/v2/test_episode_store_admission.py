"""Episode-store admission, retention, and recall.

Regression tests for the 2026-07-27 defect: every completed v5 run had a store
frozen since ~step 1000 -- three of four blocks never admitted anything, and
the fourth filled during the initialization transient and never changed a slot
again. See docs/research/2026-07-27_episode-store-frozen-defect.md.

The decisive property is *turnover under a decaying signal*: a store that fills
early and then freezes is the bug, so these tests drive salience series that
start high and decay, exactly as the real runs do. Note that the measured
per-block salience medians in production are 0.001-0.004, well BELOW the legacy
absolute threshold of 0.1 -- any admission rule that keeps an absolute floor is
inert at real scale, which is what several of these tests pin down.
"""

import torch

from luthi.v2.living_layer_pc import PredictiveCodingLayer


def _layer(**kw):
    # The fix is opt-in per arm (default False preserves pre-2026-07-27
    # behaviour for completed families), so these tests enable it explicitly.
    kw.setdefault("adaptive_episodes", True)
    kw.setdefault("adaptive_recall", True)
    kw.setdefault("in_features", 8)
    kw.setdefault("out_features", 8)
    kw.setdefault("num_episodes", 4)
    kw.setdefault("salience_window_size", 32)
    kw.setdefault("episode_warmup_steps", 32)
    kw.setdefault("episode_age_tau", 200.0)
    kw.setdefault("refractory_calls", 5)      # tiny, so short tests can write
    kw.setdefault("surprise_decay", 0.2)      # fast, so the baseline settles
    return PredictiveCodingLayer(**kw)


def _feed(layer, saliences, seed=0):
    """Drive _store_episode directly with a salience series, varied contexts."""
    g = torch.Generator().manual_seed(seed)
    for s in saliences:
        ctx = torch.randn(layer.episode_contexts.shape[1], generator=g)
        ctx = ctx / ctx.norm()
        pattern = torch.randn(layer.in_features, generator=g)
        layer._store_episode(ctx, float(s), pattern)


def test_warmup_uses_legacy_admission_so_the_store_is_never_inert():
    """Warmup is statistical (window fill), not a step count. Before there are
    enough samples to compute a percentile, admission falls back to the legacy
    absolute rule -- a mechanism that quietly stores nothing while reporting
    healthy is the failure this fix exists to end, and a step-count lockout
    would reproduce it in every short run."""
    layer = _layer(salience_threshold=0.0)
    _feed(layer, [1.0] * 3)
    assert int(layer.episode_count.item()) > 0
    assert int(layer.episode_writes.item()) > 0


def test_transient_episodes_are_not_permanent():
    """Warmup admissions must not fossilize: age decay has to let them go."""
    layer = _layer(salience_threshold=0.0, episode_age_tau=10.0)
    _feed(layer, [5.0] * 8)                       # 'initialization transient'
    transient = layer.episode_saliences.clone()
    g = torch.Generator().manual_seed(21)
    settled = [0.02 + 0.01 * torch.rand(1, generator=g).item() for _ in range(600)]
    _feed(layer, settled, seed=22)
    assert not torch.equal(transient, layer.episode_saliences), (
        "transient-era episodes were never recycled -- the fossil is back"
    )


def test_admits_after_warmup_on_relative_bar():
    layer = _layer()
    _feed(layer, [0.01] * 64)
    before = int(layer.episode_writes.item())
    _feed(layer, [5.0], seed=1)
    assert int(layer.episode_writes.item()) == before + 1


def test_admission_survives_a_signal_below_the_legacy_floor():
    """Production saliences sit at 0.001-0.004. Admission must work there."""
    layer = _layer(salience_threshold=0.1)
    g = torch.Generator().manual_seed(3)
    tiny = [0.002 + 0.0005 * torch.rand(1, generator=g).item() for _ in range(600)]
    _feed(layer, tiny)
    assert int(layer.episode_writes.item()) > 0, (
        "no admission below the legacy absolute floor -- the fix is inert"
    )


def test_store_does_not_freeze_under_decaying_salience():
    """The defect in one assertion: high early salience must not lock the
    store for the rest of the run."""
    g = torch.Generator().manual_seed(5)
    early = [1.0 + 0.5 * torch.rand(1, generator=g).item() for _ in range(200)]
    late = [0.01 + 0.005 * torch.rand(1, generator=g).item() for _ in range(800)]
    layer = _layer()
    _feed(layer, early)
    writes_after_early = int(layer.episode_writes.item())
    _feed(layer, late, seed=7)
    assert int(layer.episode_writes.item()) > writes_after_early, (
        "store froze: no episode admitted once salience decayed"
    )


def test_ordinary_wobble_is_not_admitted():
    """THE PROBE FAILURE, pinned. A trailing percentile has no notion of
    scale: real salience wobbles ~1.5% within a window, p99.5 sits a hair
    above the median, and local drift clears it -- 85% of calls admitted,
    store filled with consecutive steps, similarity 1.0000. Admission must
    key on surprise relative to how much this layer ordinarily varies."""
    layer = _layer(refractory_calls=0)
    g = torch.Generator().manual_seed(31)
    # 1.5% wobble on a slowly rising baseline -- exactly the probe's regime
    series = [0.00185 * (1 + 0.015 * torch.rand(1, generator=g).item()
                         + 0.02 * i / 400) for i in range(400)]
    _feed(layer, series)
    rate = int(layer.episode_writes.item()) / len(series)
    assert rate < 0.10, f"admitted {rate:.0%} of ordinary wobble (probe hit 85%)"


def test_admits_on_a_decaying_signal_with_spikes():
    """BOTH previous rules died here, in mirror-image ways.

    v1 (trailing percentile) admitted 85% of calls, because a locally RISING
    series always clears its own recent maximum. v2 (surprise over a plain
    EMA) admitted essentially nothing -- 1 write per block in 3,000 real
    steps -- because on a DECAYING series the EMA baseline always sits above
    the current value, so the residual is negative and nothing is ever
    surprising. Real salience decays 45x over a run, so v2's regime is the
    one production actually lives in.

    Drift correction is what makes both cases behave: a smooth trend in
    either direction produces no residual, and only a departure from the
    trend does."""
    layer = _layer(refractory_calls=10, surprise_decay=0.1)
    g = torch.Generator().manual_seed(41)
    series = []
    for i in range(600):
        base = 0.05 * (0.995 ** i)                     # steady 95% decay
        noise = 1.0 + 0.02 * torch.rand(1, generator=g).item()
        spike = 3.0 if i in (300, 420, 500) else 1.0   # three genuine events
        series.append(base * noise * spike)
    _feed(layer, series)
    writes = int(layer.episode_writes.item())
    assert writes > 0, "froze on a decaying signal (the v2 failure)"
    assert writes < 60, f"admitted {writes}/600 -- thrashing (the v1 failure)"


def test_refractory_prevents_consecutive_writes():
    """Diversity in time is a precondition for diversity in content: the
    probe's store held steps 39985, 39986, 39987... hence similarity 1.0."""
    layer = _layer(refractory_calls=20)
    _feed(layer, [0.01] * 64)
    _feed(layer, [9.0] * 40, seed=9)          # sustained huge salience
    steps = layer.episode_steps[layer.episode_steps >= 0].tolist()
    steps.sort()
    gaps = [b - a for a, b in zip(steps, steps[1:])]
    assert all(g >= 20 for g in gaps), f"writes closer than refractory: {gaps}"


def test_age_decay_makes_stale_slots_evictable():
    layer = _layer(episode_age_tau=10.0)
    _feed(layer, [0.01] * 64)
    _feed(layer, [5.0, 6.0, 7.0, 8.0], seed=2)
    stored = layer.episode_saliences.clone()
    _feed(layer, [0.01] * 400, seed=3)
    _feed(layer, [0.5], seed=4)
    assert not torch.equal(stored, layer.episode_saliences), (
        "aged high-salience slots were never recycled"
    )


def test_legacy_path_is_preserved_for_ab():
    layer = _layer(adaptive_episodes=False, salience_threshold=0.1)
    _feed(layer, [0.5] * 4)
    assert int(layer.episode_count.item()) == 4
    _feed(layer, [0.05] * 50, seed=5)
    assert int(layer.episode_count.item()) == 4


def test_adaptive_recall_does_not_fire_on_every_step():
    """Fixed threshold 0.5 against similarities >0.9 blended a stored delta
    into every forward pass."""
    layer = _layer()
    _feed(layer, [0.01] * 64)
    _feed(layer, [5.0, 6.0, 7.0, 8.0], seed=6)
    g = torch.Generator().manual_seed(11)
    fires = 0
    for _ in range(200):
        ctx = torch.randn(layer.episode_contexts.shape[1], generator=g)
        ctx = ctx / ctx.norm()
        if layer._recall_episode(ctx) is not None:
            fires += 1
    assert fires < 100, f"recall fired {fires}/200 times -- not discriminating"


def test_stats_expose_store_health():
    layer = _layer()
    _feed(layer, [0.01] * 64)
    _feed(layer, [5.0, 6.0, 7.0], seed=8)
    stats = layer._episode_store_stats()
    for key in ("episode_writes", "recall_fires", "episode_salience_floor",
                "episode_admission_bar", "episode_context_similarity"):
        assert key in stats
    assert stats["episode_context_similarity"] is not None
    assert stats["episode_admission_bar"] is not None
