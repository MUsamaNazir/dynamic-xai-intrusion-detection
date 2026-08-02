"""
04b_prioritization_unsw.py
--------------------------
STEP 4b - Alert prioritization on the UNSW-NB15 XGBoost backbone.

Mirrors 04_prioritization.py (v3) exactly. Only the data/results paths and
the benign class name differ:

    data/unsw/*        instead of data/*
    results/unsw/*     instead of results/*
    BENIGN = "Normal"  instead of "Benign"

Run:  python 04b_prioritization_unsw.py

Outputs (results/unsw/):
    weights.txt                 chosen alpha, beta, gamma and the recall floor
    weight_sensitivity.csv      full validation simplex sweep
    theta_sweep.csv             test threshold sweep  -> cross-dataset ARR table
    perclass_recall_theta.csv   per-class recall vs theta -> rare-class table

NOTE ON THE C_ctx TERM
----------------------
The third term of Eq. 7 is G[:, 2] from models.build_graph_embed(). Because
UNSW-NB15 supplies host addresses and timestamps, that function takes its
host-graph branch and column 2 is the source node's FAN-OUT (distinct
destinations contacted within a 5-minute slot). On CICIoT2023, which lacks
those fields, column 2 is instead the rolling standard deviation of local
activity within the protocol group. The two are not the same quantity: on
UNSW-NB15, C_ctx is the structural measure Section 3.7 actually motivates,
whereas on CICIoT2023 it is a dispersion proxy. This difference must be
stated wherever the two prioritization results are compared.
"""

import os
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

import models as M

DATA = "data/unsw"
RESULTS = "results/unsw"
BENIGN = "Normal"
TUNE_THETA = 0.6          # screening threshold for weight selection
# Recall floors expressed as a FRACTION of the unprioritized baseline macro
# recall on the validation split, tried in order. Absolute floors cannot be
# transferred between datasets: CICIoT2023 has a validation macro recall near
# 88%, so its published 85% absolute floor is ~96% of baseline, whereas
# UNSW-NB15 has a baseline near 69% and no absolute floor of 85% is reachable
# (prioritization can only lower recall). Relative floors keep the selection
# rule identical in intent across both benchmarks.
REL_FLOORS = (0.98, 0.96, 0.94, 0.90)
THETAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
os.makedirs(RESULTS, exist_ok=True)

te = np.load(f"{DATA}/test.npz")
va = np.load(f"{DATA}/val.npz")
trn = np.load(f"{DATA}/train_noSMOTE.npz")
meta = pd.read_parquet(f"{DATA}/meta.parquet")
classes = np.load(f"{DATA}/classes.npy", allow_pickle=True)
benign = int(np.where(classes == BENIGN)[0][0])
nF = len([l for l in open(f"{DATA}/feature_names.txt").read().split("\n")
          if l.strip()])

print("=" * 72)
print("  STEP 4b  Alert prioritization on UNSW-NB15")
print("=" * 72)
print(f"classes ({len(classes)}): {list(classes)}")
print(f"benign  : {BENIGN} (index {benign})")
print(f"features: {nF}   val {len(va['y']):,}   test {len(te['y']):,}")

xgb = XGBClassifier()
xgb.load_model(f"{RESULTS}/xgb_seed42.json")

# ---- the host-graph context is built once over the full row order --------
full = np.zeros((len(meta), nF), dtype="float32")
for part in (trn, va, te):
    full[part["idx"]] = part["X"]
G = np.nan_to_num(M.build_graph_embed(full, meta))
host_graph = ("Src IP" in meta.columns) and ("Timestamp" in meta.columns)
print(f"\ntopology source: {'host graph (fan-out)' if host_graph else 'protocol-group proxy (dispersion)'}")
print(f"G[:,2] mean {G[:,2].mean():.3f}  std {G[:,2].std():.3f}  "
      f"min {G[:,2].min():.3f}  max {G[:,2].max():.3f}")

explainer = shap.TreeExplainer(xgb)


def terms(X, idx):
    """The three normalized terms of Eq. 7 for one split."""
    prob = xgb.predict_proba(X)
    p_att = 1 - prob[:, benign]

    raw = np.array(explainer.shap_values(X))
    if raw.ndim == 3 and raw.shape[0] == len(X):     # new shap: (n, feat, class)
        sv = np.abs(raw).mean(2)
    else:                                            # old shap: (class, n, feat)
        sv = np.abs(raw).mean(0)
    s = np.sort(sv, 1)[:, -10:].mean(1)
    s = np.nan_to_num((s - s.min()) / (np.ptp(s) + 1e-9))

    c = G[idx][:, 2]
    c = np.nan_to_num((c - c.min()) / (np.ptp(c) + 1e-9))
    return prob, np.nan_to_num(p_att), s, c


print("\ncomputing terms (val)...", flush=True)
prob_v, p_v, s_v, c_v = terms(va["X"], va["idx"])
print("computing terms (test)...", flush=True)
prob_t, p_t, s_t, c_t = terms(te["X"], te["idx"])


def evaluate(a, b, g, theta, prob, p_att, s_shap, c_ctx, y):
    """Suppressed alerts are scored as benign against the FULL split."""
    P = a * p_att + b * s_shap + g * c_ctx
    keep = P >= theta
    yhat = np.where(keep, prob.argmax(1), benign)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, yhat, average="macro", zero_division=0)
    return (1 - keep.mean()) * 100, rc * 100, pr * 100, f1 * 100, keep, yhat


# ---- baseline (no prioritization) on validation -------------------------
_, rc_base_v, _, _ = precision_recall_fscore_support(
    va["y"], prob_v.argmax(1), average="macro", zero_division=0)
rc_base_v *= 100
FLOORS = [(f, round(f * rc_base_v, 2)) for f in REL_FLOORS]
print(f"\nvalidation baseline macro recall: {rc_base_v:.2f}%")
print("recall floors (relative -> absolute): " +
      ", ".join(f"{int(f*100)}% -> {a:.2f}%" for f, a in FLOORS))

# ---- tune alpha/beta/gamma on validation --------------------------------
grid = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
rows, best = [], None
for a in grid:
    for b in grid:
        g = round(1 - a - b, 2)
        if g < -1e-9:
            continue
        g = max(g, 0.0)
        arr, rc, pr, f1, _, _ = evaluate(a, b, g, TUNE_THETA,
                                         prob_v, p_v, s_v, c_v, va["y"])
        rows.append(dict(alpha=a, beta=b, gamma=g,
                         ARR=round(arr, 2), recall=round(rc, 2)))
        for rel, absf in FLOORS:
            if rc >= absf and (best is None or best[4] < rel or
                               (best[4] == rel and arr > best[0])):
                best = (arr, a, b, g, rel)
                break

sens = pd.DataFrame(rows)
sens.to_csv(f"{RESULTS}/weight_sensitivity.csv", index=False)

if best is None:
    # Never crash: fall back to the highest-recall configuration and say so.
    top = sens.loc[sens["recall"].idxmax()]
    A, B, Gm = float(top.alpha), float(top.beta), float(top.gamma)
    floor_desc = (f"NONE MET - fell back to max-recall configuration "
                  f"(validation recall {top.recall:.2f}% = "
                  f"{100*top.recall/rc_base_v:.1f}% of baseline)")
    print(f"\n!! no configuration met even the {int(min(REL_FLOORS)*100)}% "
          f"relative floor; using the max-recall configuration")
else:
    _, A, B, Gm, rel = best
    absf = dict(FLOORS)[rel]
    floor_desc = (f"{int(rel*100)}% of baseline "
                  f"(absolute {absf:.2f}% macro recall)")
    print(f"\nchosen weights: alpha={A} beta={B} gamma={Gm}  "
          f"(floor {int(rel*100)}% of baseline = {absf:.2f}%)")

open(f"{RESULTS}/weights.txt", "w").write(
    f"alpha={A} beta={B} gamma={Gm}\n"
    f"recall floor used: {floor_desc}\n"
    f"validation baseline macro recall: {rc_base_v:.2f}%\n"
    f"tuned at theta={TUNE_THETA} on the validation split\n")

near = sens[(sens.alpha.between(A - 0.1, A + 0.1)) &
            (sens.beta.between(B - 0.1, B + 0.1))]
print("\nvalidation sensitivity near the operating point:")
print(near.sort_values(["alpha", "beta"]).to_string(index=False))

# ---- theta sweep on test ------------------------------------------------
sweep, perclass = [], []
for th in THETAS:
    arr, rc, pr, f1, keep, yhat = evaluate(A, B, Gm, th,
                                           prob_t, p_t, s_t, c_t, te["y"])
    sweep.append(dict(theta=th, presented=int(keep.sum()),
                      ARR=round(arr, 1), recall=round(rc, 2),
                      precision=round(pr, 2), f1=round(f1, 2)))
    for ci, cls in enumerate(classes):
        m = te["y"] == ci
        perclass.append(dict(theta=th, cls=cls,
                             recall=round(100 * (yhat[m] == ci).mean(), 1)))

sw = pd.DataFrame(sweep)
sw.to_csv(f"{RESULTS}/theta_sweep.csv", index=False)
pc = pd.DataFrame(perclass).pivot(index="cls", columns="theta", values="recall")
pc.to_csv(f"{RESULTS}/perclass_recall_theta.csv")

print("\n" + "=" * 72)
print("threshold sweep (test):")
print(sw.to_string(index=False))

# baseline (no prioritization) for the comparison table
p0 = prob_t.argmax(1)
pr0, rc0, f10, _ = precision_recall_fscore_support(
    te["y"], p0, average="macro", zero_division=0)
print(f"\nno prioritization: precision {pr0*100:.2f}  recall {rc0*100:.2f}  "
      f"F1 {f10*100:.2f}  alerts {len(te['y']):,}")

print("\nper-class recall vs theta:")
print(pc.reindex([c for c in classes]).to_string())
print("\n" + "=" * 72)
print(f"wrote {RESULTS}/ weights.txt weight_sensitivity.csv "
      f"theta_sweep.csv perclass_recall_theta.csv")
print("=" * 72)
