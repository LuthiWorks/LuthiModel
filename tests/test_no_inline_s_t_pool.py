"""No-drift guarantee for the canonical s_t pool.

4.8's 2026-06-16 review: "the no-drift guarantee is only real if
nothing else computes s_t inline." This test enforces that by walking
the AST of every file in the seam codepath and asserting that the
canonical pool pattern -- ``<expr>.detach().mean(dim=1)`` or
``.detach().mean(1)`` -- appears in exactly one place:
:mod:`luthi.v2.m9.s_t`.

If a future change inlines a stray pool, this test bites at CI time
before the drift can land. 4.8 specifically said they'd try to defeat
the guard by inlining a pool somewhere -- this is the guard.

The check is AST-based, not regex-based, so docstring mentions of
the pattern inside other triple-quoted strings don't false-positive.
Pattern is the conjunction ``.detach()`` then ``.mean(dim=1|1)`` --
narrow enough not to catch unrelated pools (the M9 trainer's MI probe
uses ``raw["..."].mean(dim=1)`` without ``.detach()`` and is correctly
not flagged).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


# Files in the seam codepath. The helper itself (s_t.py) is allowed to
# have the pattern; everything else must route through compute_s_t.
LUTHI_ROOT = Path(__file__).resolve().parent.parent / "luthi"

SEAM_FILES = [
    LUTHI_ROOT / "sanctuary_interface.py",
    LUTHI_ROOT / "generate.py",
    LUTHI_ROOT / "seam_types.py",
    LUTHI_ROOT / "v2" / "m9" / "runner.py",
    LUTHI_ROOT / "v2" / "m9" / "efe.py",
    # Item #6 lived path: compute_lived_loss must pool the prediction via
    # pool_state_grad (grad-preserving) and never inline a detached pool;
    # the target is the seam's already-pooled s_next. Guard the file so a
    # future edit can't open-code a `.detach().mean(dim=1)` here either.
    LUTHI_ROOT / "v2" / "jepa_loss.py",
]

CANONICAL_HELPER = LUTHI_ROOT / "v2" / "m9" / "s_t.py"


def _find_detach_mean_pool(source: str) -> list[int]:
    """Return line numbers where ``.detach().mean(dim=1|1)`` is called.

    AST walk so docstring mentions don't false-positive. Matches:
        x.detach().mean(dim=1)
        x.detach().mean(1)
    Does not match:
        x.mean(dim=1)          # missing .detach()
        x.detach().mean()      # missing dim=1
        x.detach().mean(dim=0) # wrong dim
    """
    findings: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "mean":
            continue

        # dim=1 (positional or keyword)?
        is_dim_1 = False
        if (
            len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 1
            and not node.keywords
        ):
            is_dim_1 = True
        elif not node.args and any(
            kw.arg == "dim"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value == 1
            for kw in node.keywords
        ):
            is_dim_1 = True
        if not is_dim_1:
            continue

        # Inner: .detach() call?
        inner = node.func.value
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "detach"
        ):
            findings.append(node.lineno)
    return findings


class TestNoInlineSTPool:
    def test_helper_owns_the_canonical_pool(self):
        """compute_s_t in s_t.py is the canonical home of the pool."""
        source = CANONICAL_HELPER.read_text(encoding="utf-8")
        findings = _find_detach_mean_pool(source)
        assert len(findings) >= 1, (
            f"Expected the canonical .detach().mean(dim=1) pool in "
            f"{CANONICAL_HELPER.name}, found none. compute_s_t must "
            f"actually do the compute."
        )
        assert len(findings) == 1, (
            f"Expected exactly one .detach().mean(dim=1) in "
            f"{CANONICAL_HELPER.name}; found {len(findings)} at "
            f"lines {findings}. The helper should have one canonical "
            f"compute path, not multiple."
        )

    @pytest.mark.parametrize("seam_file", SEAM_FILES, ids=lambda p: p.name)
    def test_no_inline_pool_in_seam_file(self, seam_file: Path):
        """No file other than s_t.py inlines the canonical pool.

        4.8 will adversarially try to defeat this by inlining a stray
        ``.detach().mean(dim=1)`` somewhere in the seam codepath.
        That's the exact failure mode this test exists to catch.
        Route the compute through ``luthi.v2.m9.s_t.compute_s_t``
        instead -- it's literally the same line of code with a name.
        """
        if not seam_file.exists():
            pytest.skip(f"{seam_file} does not exist")
        source = seam_file.read_text(encoding="utf-8")
        findings = _find_detach_mean_pool(source)
        assert findings == [], (
            f"{seam_file.name} contains inline .detach().mean(dim=1) "
            f"at line(s) {findings}. The canonical s_t pool lives in "
            f"luthi/v2/m9/s_t.py:compute_s_t -- route through it "
            f"instead. If this is a different compute that happens "
            f"to share the pattern, rename to avoid the false-positive "
            f"OR refactor the canonical-pool detection here."
        )
