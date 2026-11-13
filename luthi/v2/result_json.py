"""Writing an instrument's result as JSON any reader accepts (2026-11-13).

Python writes a float that is not a number as a bare ``NaN``, which is not
JSON. Python reads it back, and a strict reader refuses the file. The
adaptation instrument writes one on purpose: a dead control's differences
are all exactly 0, so their standard error is 0 and the t-value is 0
divided by 0. All six dead-control results from Brian's card had it.

Here a float that is not finite is written as ``null``. In a result, null
means "not a number": undefined, or infinite. It is not hidden and it is
not replaced by a number.
"""

from __future__ import annotations

import json
import math
from typing import Any


def _finite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def dumps(result: Any) -> str:
    """``result`` as strict JSON, indented as the instruments write it."""
    return json.dumps(_finite(result), indent=2, allow_nan=False)
