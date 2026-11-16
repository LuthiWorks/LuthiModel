"""Why a job on Brian's card finishes its work and then does not exit.

``adaptation_eval.py`` wrote a whole result and then did not exit, 38 times
in 38, on 2026-11-13. ``living_write_probe.py`` and ``per_block_probe.py``
exited every time (30 of 30).

Found by bisection on 2026-11-16 (ROCm, torch 2.9.1+rocm7.2.1): **a process
that returns with work still queued on the card does not exit.** It spins on
one core. Anything that waits for the card before returning lets it exit:
``torch.cuda.synchronize()``, reading a value back (``.item()``, ``float()``,
``.cpu()``), or ``torch.cuda.empty_cache()``. Sleeping does not.

That fits all three scripts. ``adaptation_eval`` ends by restoring the model
(``load_state_dict``: copies on the card, never read back). The other two end
by reading numbers off the card.

Each case below does a little work on the card and returns. Run one at a
time with a timeout from outside; 2 to 3 seconds is a clean exit.

    python redteam/terminal_fable_2026_11_16/probe_exit_hang.py <case>

Measured, each twice or more, with other jobs running on the card:

    case            exits?
    host_to_card    yes   (a copy from the CPU to the card; no kernel queued)
    queued          NO    (copy_ and matmul on the card, then return)
    sleep           NO    (the same, then 5 seconds of sleep)
    synchronize     yes   (the same, then torch.cuda.synchronize())
    read_back       yes   (the same, then .sum().item())
    empty_cache     yes   (the same, then del and torch.cuda.empty_cache())
"""

import sys
import time

import torch

CASES = ("host_to_card", "queued", "sleep", "synchronize", "read_back", "empty_cache")
what = sys.argv[1] if len(sys.argv) > 1 else ""
if what not in CASES:
    raise SystemExit(f"one of: {', '.join(CASES)}")

if what == "host_to_card":
    a = torch.randn(512, 512).cuda()
else:
    a = torch.randn(512, 512, device="cuda")
    b = a.clone()
    with torch.no_grad():
        a.copy_(b)
    y = a @ b
    if what == "sleep":
        time.sleep(5)
    elif what == "synchronize":
        torch.cuda.synchronize()
    elif what == "read_back":
        y.sum().item()
    elif what == "empty_cache":
        del a, b, y
        torch.cuda.empty_cache()
print(f"{what}: returning at {time.strftime('%H:%M:%S')}", flush=True)
