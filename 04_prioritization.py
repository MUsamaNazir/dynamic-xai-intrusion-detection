"""
STEP 4 (v3) - Alert prioritization on the XGBoost backbone (flow-level).
Run:  %run 04_prioritization.py
Outputs: results/weights.txt, weight_sensitivity.csv, theta_sweep.csv,
         perclass_recall_theta.csv
"""
import numpy as np, pandas as pd, shap
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support
import models as M

va = np.load("data/val.npz"); te = np.load("data/test.npz")
trn = np.load("data/train_noSMOTE.npz")
meta = pd.read_parquet("data/meta.parquet")
classes = np.load("data/classes.npy", allow_pickle=True)
benign = int(np.where(classes == "Benign")[0][0])
nF = len(open("data/feature_names.txt").read().splitlines())

xgb = XGBClassifier(); xgb.load_model("results/xgb_seed42.json")

def terms(X, idx):
    prob = xgb.predict_proba(X)
    p_att = 1 - prob[:, benign]
    raw = np.array(shap.TreeExplainer(xgb).shap_values(X))
    if raw.ndim == 3 and raw.shape[0] == len(X):      # new shap: (n, feat, class)
        sv = np.abs(raw).mean(2)
    else:                                             # old shap: (class, n, feat)
        sv = np.abs(raw).mean(0)
    s = np.sort(sv, 1)[:, -10:].mean(1)
    s = np.nan_to_num((s - s.min()) / (np.ptp(s) + 1e-9))
    full = np.zeros((len(meta), nF), dtype="float32")
    for part in (trn, va, te): full[part["idx"]] = part["X"]
    G = np.nan_to_num(M.build_graph_embed(full, meta))
    c = G[idx][:, 2]
    c = np.nan_to_num((c - c.min()) / (np.ptp(c) + 1e-9))
    return prob, np.nan_to_num(p_att), s, c

print("computing terms (val)...")
prob_v, p_v, s_v, c_v = terms(va["X"], va["idx"])
print("computing terms (test)...")
prob_t, p_t, s_t, c_t = terms(te["X"], te["idx"])

def evaluate(a, b, g, theta, prob, p_att, s_shap, c_gcn, y):
    P = a * p_att + b * s_shap + g * c_gcn
    keep = P >= theta
    yhat = np.where(keep, prob.argmax(1), benign)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="macro",
                                                    zero_division=0)
    return (1 - keep.mean()) * 100, rc * 100, pr * 100, f1 * 100, keep, yhat

# ---- tune alpha/beta/gamma on validation (recall floor 90, fallback 85) ----
grid = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
rows, best = [], None
for a in grid:
    for b in grid:
        g = round(1 - a - b, 2)
        if g < -1e-9: continue
        g = max(g, 0.0)
        arr, rc, pr, f1, _, _ = evaluate(a, b, g, 0.6, prob_v, p_v, s_v, c_v, va["y"])
        rows.append(dict(alpha=a, beta=b, gamma=g, ARR=round(arr, 2),
                         recall=round(rc, 2)))
        for floor in (90, 85):
            if rc >= floor and (best is None or best[4] < floor or
                                (best[4] == floor and arr > best[0])):
                best = (arr, a, b, g, floor); break
pd.DataFrame(rows).to_csv("results/weight_sensitivity.csv", index=False)
assert best is not None, "no weight combo found - inspect weight_sensitivity.csv"
_, A, B, Gm, floor = best
open("results/weights.txt", "w").write(
    f"alpha={A} beta={B} gamma={Gm} (recall floor used: {floor})\n")
print(f"chosen weights: alpha={A} beta={B} gamma={Gm} (floor {floor})")

# ---- theta sweep + per-class recall on test ----
sweep, perclass = [], []
for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    arr, rc, pr, f1, keep, yhat = evaluate(A, B, Gm, th, prob_t, p_t, s_t, c_t, te["y"])
    sweep.append(dict(theta=th, presented=int(keep.sum()),
                      ARR=round(arr, 1), recall=round(rc, 2),
                      precision=round(pr, 2), f1=round(f1, 2)))
    for ci, cls in enumerate(classes):
        m = te["y"] == ci
        perclass.append(dict(theta=th, cls=cls,
                             recall=round(100 * (yhat[m] == ci).mean(), 1)))
pd.DataFrame(sweep).to_csv("results/theta_sweep.csv", index=False)
pd.DataFrame(perclass).pivot(index="cls", columns="theta", values="recall") \
    .to_csv("results/perclass_recall_theta.csv")
print(pd.DataFrame(sweep).to_string(index=False))
print("\nper-class recall vs theta:")
print(pd.DataFrame(perclass).pivot(index="cls", columns="theta", values="recall"))
