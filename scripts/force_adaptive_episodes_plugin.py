"""Force the episode-store fix ON for a test run, without editing source.

Used as `pytest -p force_on` via PYTHONPATH, or imported by conftest when
LUTHI_FORCE_ADAPTIVE_EPISODES=1. Patches the layer's __init__ defaults so every
construction in the suite opts in -- the point is to prove the suite passes
with the feature enabled, not to change the shipped default.
"""
import os

if os.environ.get("LUTHI_FORCE_ADAPTIVE_EPISODES") == "1":
    import inspect
    from luthi.v2.living_layer_pc import PredictiveCodingLayer

    _orig = PredictiveCodingLayer.__init__
    _sig = inspect.signature(_orig)

    def _patched(self, *args, **kwargs):
        kwargs.setdefault("adaptive_episodes", True)
        kwargs.setdefault("adaptive_recall", True)
        return _orig(self, *args, **kwargs)

    _patched.__signature__ = _sig
    PredictiveCodingLayer.__init__ = _patched
