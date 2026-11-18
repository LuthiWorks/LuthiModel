"""From E1's result files: when a read moves the targets, do they move with the
prediction, toward it, or away from it?

Fable 5.2 (chat), 2026-11-18, after card report 5. Reads only files under
runs_meta; runs no model.

Opus's identity (Amendment 6 draft, (a)), per document:

    nmse - held = drift - 2*a + nmse*(1 - var_B/var_A)

with a = mean((P_B - T_A) * (T_B - T_A)) / var_A. Every other term is in the
file (var = l_pred / nmse), so a follows. Then a / drift says:

    about 1   the prediction and the target moved together (the frame moved);
    well over 1   the target moved toward the prediction by more than that;
    below 0   the target moved away from the prediction.

    python redteam/fable_2026_11_18/read_how_targets_and_predictions_move.py
"""
import json, os, statistics as st

BASE = "runs_meta/living_gradient_2026_11_17/"
RUNS = ("living_v5_4x_d4_512d_seed42", "probe_768_visreg_768d_seed97")
READS = ("gradient_1e-5_held", "living_gradient_1e-5_held", "error_low",
         "error_turned_low", "error_mid", "error_turned_mid")


def parts(path):
    rows = {"B": [], "C": []}
    for r in json.load(open(path))["per_doc"]:
        A = r["A"]
        var_a = A["l_pred"] / A["nmse"]
        for arm in "BC":
            X = r[arm]
            if X.get("held") is None or not X.get("drift"):
                continue
            ratio = (X["l_pred"] / X["nmse"]) / var_a
            a = (X["drift"] + X["nmse"] * (1 - ratio) - (X["nmse"] - X["held"])) / 2
            rows[arm].append((a / X["drift"], X["drift"], X["nmse"] - A["nmse"], X["held"] - A["nmse"]))
    return rows


for run in RUNS:
    print(run)
    for name in READS:
        path = f"{BASE}{run}/e1_read64_{name}.json"
        if not os.path.exists(path):
            print(f"  {name:26s} no result (the read raised)")
            continue
        for arm, rows in parts(path).items():
            if not rows:
                continue
            med = lambda i: st.median(v[i] for v in rows)
            print(f"  {name:26s} {arm}: a/drift {med(0):+.2f} | drift {med(1):.1e} | "
                  f"nmse less A {med(2):+.1e} | held less A {med(3):+.1e}")
