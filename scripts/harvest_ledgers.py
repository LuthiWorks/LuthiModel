"""Read-only ledger harvester for the live seed43 run.

Every 5 minutes: find rolling checkpoints not yet harvested (mtime stable for
>60s so we never read a slot mid-write), copy each aside, extract ONLY the
per-block precision (trust ledger) buffers + step, save a tiny snapshot, delete
the copy. Never touches the run directory's own files. Exits when the run
completes (pilot_result.json appears) or after 14 hours.
"""
import glob as _glob
import json, os, shutil, sys, time
import torch

BASE = r"C:\Dev\LuthiModel\runs\jepa_pilot"
# ALL mode: watch every v5-family run dir (including the stage-11 rerun arm)
# and harvest whichever is active. Legacy short dir names preserved for the
# two runs already harvested under them.
LEGACY_OUT = {
    "living_v5_4x_d4_512d_seed43": "ledger_harvest_seed43",
    "living_v5_4x_d4_512d_seed44": "ledger_harvest_seed44",
}
def out_dir_for(run_name):
    return os.path.join(BASE, LEGACY_OUT.get(run_name, f"ledger_harvest_{run_name}"))
def watched_runs():
    return sorted(
        d for d in _glob.glob(os.path.join(BASE, "living_v5_4x_d4*_512d_seed*"))
        if os.path.isdir(os.path.join(d, "checkpoints"))
    )
seen = set()  # (run_name, ckpt_name, int(mtime))
t_end = time.time() + 72 * 3600   # long horizon; covers seeds 45/46 + rerun

print(f"harvester up (ALL mode); base {BASE}", flush=True)
while time.time() < t_end:
    now = time.time()
    for run in watched_runs():
        run_name = os.path.basename(run)
        ckpt_dir = os.path.join(run, "checkpoints")
        out = out_dir_for(run_name)
        os.makedirs(out, exist_ok=True)
        tmp = os.path.join(out, "_tmp.pt")
        try:
            names = sorted(os.listdir(ckpt_dir))
        except OSError:
            continue
        for name in names:
            if not name.endswith(".pt"):
                continue
            p = os.path.join(ckpt_dir, name)
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            key = (run_name, name, int(mt))
            if key in seen or now - mt < 60:
                continue
            try:
                shutil.copy2(p, tmp)
                ck = torch.load(tmp, map_location="cpu", weights_only=False)
                # trainer checkpoint layout (verified 2026-07-25): the living
                # substrate's buffers live in online_state_dict as
                # blocks.N.living_ffn.precision
                state = ck.get("online_state_dict") or {}
                step = ck.get("global_step") or ck.get("step")
                # Capture every per-dimension living buffer, not just trust.
                # Lesson from 2026-07-27: harvesting `precision` alone left us
                # able to see WHEN dimensions were abandoned but not whether
                # their updates died first -- and the rolling checkpoints that
                # held the answer had already rotated away. Cheap to take,
                # impossible to recover later.
                WANT = (".precision", ".error_acc", ".plasticity",
                        ".episode_saliences", ".episode_contexts",
                        ".episode_steps", ".episode_count")
                ledgers = {k: v.detach().clone().float() for k, v in state.items()
                           if hasattr(v, "detach") and k.endswith(WANT)}
                # [out, in] buffers reduced to per-input-dimension summaries so
                # the snapshot stays small while keeping the signal we needed.
                for k, v in state.items():
                    if (hasattr(v, "detach") and v.dim() == 2
                            and k.endswith((".update_ema", ".momentum"))):
                        ledgers[k + "__col_mean"] = (
                            v.detach().float().abs().mean(dim=0)
                        )
                if not ledgers:
                    raise RuntimeError("no living buffers in online_state_dict")
                if step is None:
                    step = int(mt)
                out_path = os.path.join(out, f"ledger_step_{int(step):08d}.pt")
                if not os.path.exists(out_path):
                    torch.save({"step": step, "mtime": mt, "source": name,
                                "ledgers": ledgers}, out_path)
                    print(f"[{run_name}] harvested {name} -> step {step}", flush=True)
                # JSON twin for LuthiScope's ledger endpoint (its server has
                # no torch; JSON keeps the instrument dependency-free)
                json_path = os.path.join(out, f"ledger_step_{int(step):08d}.json")
                if not os.path.exists(json_path):
                    with open(json_path, "w", encoding="utf-8") as jf:
                        json.dump({"step": int(step),
                                   "ledgers": {k: v.tolist() for k, v in ledgers.items()}}, jf)
                seen.add(key)
                del ck, state, ledgers
            except Exception as e:
                print(f"[{run_name}] skip {name}: {e}", flush=True)
            finally:
                if os.path.exists(tmp):
                    try: os.remove(tmp)
                    except OSError: pass
    time.sleep(300)
print("harvester done", flush=True)
