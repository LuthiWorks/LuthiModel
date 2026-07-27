"""Event-locked overlay: precision_spread around each extreme Greek serving.

For each serving: pre = mean spread in [s-300, s-100], post = mean in [s, s+300],
peak_post = max in [s, s+1000]. Isolation = distance to nearest other serving
(>=600 steps -> clean read). Split by run phase (thirds) to test state-dependence.
Runs on seed42 (complete) and seed43 (partial, live).
"""
import json, os

SCHED = {
    42: [942, 4220, 4592, 8323, 9535, 10997, 13839, 13890, 15851, 20138, 22100,
         23015, 24881, 29090, 29986, 32028, 32619, 32676, 36139, 36865, 39179,
         41653, 42315, 42565, 50095, 51580, 53992, 54350, 55053, 59145, 61356,
         61627, 63439, 65390, 65779, 66512],
    43: [828, 3717, 3942, 6886, 7613, 11158, 12866, 13670, 14920, 17871, 22263,
         22354, 28686, 29695, 33044, 34105, 35929, 37643, 38605, 38970, 41713,
         44353, 46605, 46822, 48305, 49261, 50464, 51371, 51429, 58793, 59618,
         63796, 66516, 67264, 69345, 69787],
}
BASE = r"C:\Dev\LuthiModel\runs\jepa_pilot"

def spread_series(run):
    path = os.path.join(BASE, run, "training_log.jsonl")
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            v = (r.get("substrate") or {}).get("precision_spread")
            if isinstance(v, (int, float)):
                out.append((r["step"], v))
    return out

def mean_in(series, lo, hi):
    vs = [v for s, v in series if lo <= s <= hi]
    return sum(vs) / len(vs) if vs else None

def max_in(series, lo, hi):
    vs = [v for s, v in series if lo <= s <= hi]
    return max(vs) if vs else None

for seed in (42, 43):
    run = f"living_v5_4x_d4_512d_seed{seed}"
    series = spread_series(run)
    if not series:
        print(f"\n=== seed {seed}: no data ===")
        continue
    last_step = series[-1][0]
    print(f"\n=== seed {seed}: spread points {len(series)}, through step {last_step:,} ===")
    sched = SCHED[seed]
    rows = []
    for i, s in enumerate(sched):
        if s > last_step - 300:
            continue
        near = min(abs(s - o) for o in sched if o != s)
        pre = mean_in(series, s - 300, s - 100)
        post = mean_in(series, s, s + 300)
        peak = max_in(series, s, s + 1000)
        if pre is None or post is None:
            continue
        rows.append((s, near, pre, post, peak, post - pre))
    print(f"{'serving':>8} {'isolated':>8} {'pre':>7} {'post':>7} {'peak1k':>7} {'delta':>8}")
    for s, near, pre, post, peak, d in rows:
        iso = "yes" if near >= 600 else f"({near})"
        flag = " <-- clean" if near >= 600 and abs(d) > 0.1 else ""
        print(f"{s:>8} {iso:>8} {pre:>7.3f} {post:>7.3f} {peak:>7.3f} {d:>+8.3f}{flag}")
    # phase summary (clean servings only)
    clean = [(s, d) for s, near, pre, post, peak, d in rows if near >= 600]
    if clean:
        third = (last_step if seed == 43 else 72042) / 3
        for name, lo, hi in (("early", 0, third), ("mid", third, 2 * third), ("late", 2 * third, 10 ** 9)):
            ds = [d for s, d in clean if lo <= s < hi]
            if ds:
                print(f"  {name:>5} phase: {len(ds)} clean servings, mean delta {sum(ds)/len(ds):+.4f}, max {max(ds):+.4f}")
