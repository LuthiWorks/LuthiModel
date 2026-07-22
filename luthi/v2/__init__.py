"""LuthiModel v2 — predictive coding parallel research track.

See `docs/V2_IMPLEMENTATION_PLAN.md` for the architectural specification
and `docs/LUTHI_V2_PREDICTIVE_CODING_BRIEF.md` (archived — see
`docs/ARCHIVED.md`) for the design rationale.

v2 replaces v1's Hebbian self-modification with hierarchical predictive
coding (Whittington-Bogacz variant). Living-weight semantics are preserved
— the forward pass IS still learning — but updates are driven by
prediction error rather than input-output correlation.
"""

from luthi.v2.living_layer_pc import PredictiveCodingLayer
from luthi.v2.hybrid_block_pc import PredictiveCodingBlock
from luthi.v2.model_pc import PredictiveCodingLM
from luthi.v2.backward_pass_pc import (
    TopDownSignal,
    compute_block_top_down,
    create_initial_signal,
    mask_signal,
)
from luthi.v2.consolidation import (
    ConsolidationTracker,
    consolidate_layer,
    consolidate_layer_attractor,
)

__all__ = [
    "PredictiveCodingLayer",
    "PredictiveCodingBlock",
    "PredictiveCodingLM",
    "TopDownSignal",
    "compute_block_top_down",
    "create_initial_signal",
    "mask_signal",
    "ConsolidationTracker",
    "consolidate_layer",
    "consolidate_layer_attractor",
]
