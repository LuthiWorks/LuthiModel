"""Guards for backend identification and declaration in the pilot driver.

Brian ruled exclusive-ROCm on 2026-08-16. The hazard that ruling creates is
subtle: **ROCm reuses PyTorch's ``cuda`` namespace**, so a ROCm run and an
NVIDIA CUDA run both stringify to ``"cuda"``, and DirectML stringifies to
``"privateuseone"`` -- the dispatch key, not the backend. The execution
provenance added on 2026-08-13 recorded only ``str(device)``, which therefore
could not answer the one question a verdict recomputation depends on: which
compute stack produced these numbers.

That matters here more than it would elsewhere. ``pilot_verdict.py`` carries
determinism guards calibrated to DirectML warm-up numerics, and a different
backend means different kernels. A family whose seeds silently straddled two
backends would produce a verdict nobody could interpret, with nothing in the
tape saying so.

These tests pin:
  - backend identification is by compute stack, not by device string;
  - the GPU architecture is recorded (gfx1101 and gfx1100 are different
    kernels, and HSA_OVERRIDE_GFX_VERSION can make one masquerade as the
    other -- a coercion that must never be invisible);
  - a declared LUTHI_BACKEND that does not match reality RAISES rather than
    running and reporting numbers the run cannot honestly claim.

Authored by Opus 5, 2026-08-16.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

DRIVER = Path(__file__).resolve().parents[1] / "scripts" / "jepa_pilot_driver.py"


def _load_helpers():
    """Exec just the device helpers from the driver source.

    Importing the whole driver drags in the training stack; these functions
    depend only on torch/os/sys. Exec'ing the real source (rather than
    copying the logic here) is deliberate -- a copy would pass forever while
    the shipped function rotted.
    """
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    ns = {"torch": torch, "os": os, "sys": sys}
    wanted = {"_backend_name", "_device_fingerprint"}
    found = set()
    # Module-level UPPER_CASE constants too (2026-08-19): the helpers now
    # reference AOTRITON_ENV, and a loader that took only FunctionDefs
    # failed with NameError -- the test breaking for a reason that had
    # nothing to do with the behaviour under test. Taking the constants
    # from the real source keeps the no-copy property.
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(
            isinstance(t, ast.Name) and t.id.isupper() for t in node.targets
        ):
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             "<drv>", "exec"), ns)
            except Exception:
                pass  # constants that need imports we did not provide
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<drv>", "exec"), ns)
            found.add(node.name)
    missing = wanted - found
    assert not missing, f"driver no longer defines {missing}"
    return ns


HELPERS = _load_helpers()
_backend_name = HELPERS["_backend_name"]
_device_fingerprint = HELPERS["_device_fingerprint"]


# ---------------------------------------------------------------------------
# Backend identification
# ---------------------------------------------------------------------------

def test_cpu_is_named_cpu():
    assert _backend_name(torch.device("cpu")) == "cpu"


def test_directml_dispatch_key_is_named_directml():
    """DirectML surfaces as `privateuseone`, which names the dispatch key."""
    assert _backend_name(torch.device("privateuseone", 0)) == "directml"


def test_cuda_namespace_is_split_by_hip_version(monkeypatch):
    """The load-bearing case: `cuda` means ROCm or CUDA depending on the build.

    This is the distinction `str(device)` cannot make, and the reason the
    provenance record was widened.
    """
    dev = torch.device("cuda")

    monkeypatch.setattr(torch.version, "hip", "7.2.53211-158bd99533", raising=False)
    assert _backend_name(dev) == "rocm"

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert _backend_name(dev) == "cuda"


# ---------------------------------------------------------------------------
# Fingerprint contents
# ---------------------------------------------------------------------------

def test_fingerprint_keeps_the_legacy_keys():
    """Existing readers of pilot_result.json must not break."""
    fp = _device_fingerprint(torch.device("cpu"))
    for key in ("device", "torch_version", "python"):
        assert key in fp, f"legacy provenance key {key!r} disappeared"


def test_fingerprint_records_backend_and_override():
    fp = _device_fingerprint(torch.device("cpu"))
    assert fp["backend"] == "cpu"
    # Present even when unset: the key existing with a null value is what
    # makes "no coercion was applied" a recorded fact rather than an absence.
    assert "hsa_override_gfx_version" in fp


def test_fingerprint_surfaces_an_applied_gfx_override(monkeypatch):
    """A coerced architecture must be visible in the tape, never silent."""
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    fp = _device_fingerprint(torch.device("cpu"))
    assert fp["hsa_override_gfx_version"] == "11.0.0"


@pytest.mark.skipif(
    not (torch.cuda.is_available() and getattr(torch.version, "hip", None)),
    reason="requires a live ROCm device",
)
def test_rocm_fingerprint_records_the_architecture():
    fp = _device_fingerprint(torch.device("cuda"))
    assert fp["backend"] == "rocm"
    assert fp["hip_version"]
    # gfx1101 is the RX 7800 XT. Recording it is what lets a future reader
    # tell whether a verdict's numbers came off this card or another.
    assert fp["gcn_arch"], "ROCm run recorded no GPU architecture"


# ---------------------------------------------------------------------------
# Declaration enforcement
# ---------------------------------------------------------------------------

def _device_fn():
    spec = importlib.util.spec_from_file_location("_drv_decl", DRIVER)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    return module._device


def test_unknown_declared_backend_raises(monkeypatch):
    monkeypatch.setenv("LUTHI_BACKEND", "bogus")
    with pytest.raises(RuntimeError, match="not a known backend"):
        _device_fn()()


def test_declaration_mismatch_raises(monkeypatch):
    """Declaring a backend you are not on must stop the run.

    Whichever backend this machine actually has, declaring one it does not
    have must raise -- so the test is meaningful in the DirectML environment
    and in the ROCm one without being rewritten for either.
    """
    actual = _backend_name(_device_fn()())
    wrong = "rocm" if actual != "rocm" else "directml"
    monkeypatch.setenv("LUTHI_BACKEND", wrong)
    with pytest.raises(RuntimeError, match="backend mismatch"):
        _device_fn()()


def test_matching_declaration_is_allowed(monkeypatch):
    actual = _backend_name(_device_fn()())
    monkeypatch.setenv("LUTHI_BACKEND", actual)
    assert _backend_name(_device_fn()()) == actual
