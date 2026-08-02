"""
STEP 3 — Train everything, 5 seeds, dump results.
Run:  python 03_experiments.py
Outputs:
  results/runs.csv            -> mean±std for Tables 3 and 8
  results/per_class_seed42.csv-> Table 5 and Table (per-class comparison)
  results/significance.txt    -> p-values for Table 3 caption
  results/probs_DEF.npy       -> used by 04_prioritization.py
"""
import os, numpy as np, pandas as pd, torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import ttest_rel, wilcoxon
from torch.utils.data import DataLoader, TensorDataset
import models as M

os.makedirs("results", exist_ok=True)
SEEDS = [1, 7, 13, 42, 101]

tr = np.load("data/train.npz"); trn = np.load("data/train_noSMOTE.npz")
va = np.load("data/val.npz");   te = np.load("data/test.npz")
meta = pd.read_parquet("data/meta.parquet")
classes = np.load("data/classes.npy", allow_pickle=True)
nC = len(classes)

def metrics(y, p, prob=None):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    cm = confusion_matrix(y, p)
    fpr = np.mean([(cm[:, c].sum() - cm[c, c]) / max(1, cm.sum() - cm[c].sum()) for c in range(nC)])
    auc = roc_auc_score(y, prob, multi_class="ovr", average="macro") if prob is not None else np.nan
    return dict(acc=accuracy_score(y, p)*100, prec=pr*100, rec=rc*100,
                f1=f1*100, fpr=fpr*100, auc=auc)

# sequence windows + graph context (built once from the FULL scaled matrix, split by idx)
Xall = np.vstack([trn["X"], va["X"], te["X"]])           # not used directly; windows need original order
# rebuild scaled full matrix in original order:
full_scaled = np.zeros((len(meta), trn["X"].shape[1]), dtype="float32")
for part in (trn, va, te): full_scaled[part["idx"]] = part["X"]
full_y = np.zeros(len(meta), dtype=int)
for part in (trn, va, te): full_y[part["idx"]] = part["y"]

W, Wy, Widx = M.build_windows(full_scaled, meta, full_y)
cw_pool = None  # set after split masks exist
G = M.build_graph_embed(full_scaled, meta)
split = np.full(len(meta), "tr"); split[va["idx"]] = "va"; split[te["idx"]] = "te"
wsplit = split[Widx]
_, cnt = np.unique(Wy[wsplit == "tr"], return_counts=True)
CW = (cnt.sum() / (len(cnt) * cnt))  # inverse-frequency class weights
def wl(mask, extra=None, bs=256, shuffle=False):
    tens = [torch.tensor(W[mask])]
    if extra is not None: tens.append(torch.tensor(extra[Widx][mask], dtype=torch.float32))
    tens.append(torch.tensor(Wy[mask]))
    return DataLoader(TensorDataset(*tens), batch_size=bs, shuffle=shuffle)

rows = []
for seed in SEEDS:
    np.random.seed(seed); torch.manual_seed(seed)
    # --- XGBoost ---
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        subsample=0.8, random_state=seed, tree_method="hist")
    xgb.fit(tr["X"], tr["y"])
    p, pr = xgb.predict(te["X"]), xgb.predict_proba(te["X"])
    rows.append(dict(model="XGBoost", seed=seed, **metrics(te["y"], p, pr)))
    if seed == 42:
        xgb.save_model("results/xgb_seed42.json")
        np.save("results/probs_XGB.npy", pr)
    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=seed, n_jobs=-1)
    rf.fit(tr["X"], tr["y"])
    rows.append(dict(model="RandomForest", seed=seed,
                     **metrics(te["y"], rf.predict(te["X"]), rf.predict_proba(te["X"]))))
    # --- LSTM ---
    lstm = M.train_torch(M.LSTMNet(W.shape[2], nC),
        {"train": wl(wsplit=="tr", shuffle=True), "val": wl(wsplit=="va")}, seed=seed, class_weight=CW)
    # --- Transformer only ---
    trm = M.train_torch(M.SeqTransformer(W.shape[2], nC),
        {"train": wl(wsplit=="tr", shuffle=True), "val": wl(wsplit=="va")}, seed=seed, class_weight=CW)
    # --- GCN only ---
    gcn = M.train_torch(M.GCNHead(nC),
        {"train": DataLoader(TensorDataset(torch.tensor(G[trn["idx"]]), torch.tensor(trn["y"])), 256, shuffle=True),
         "val":   DataLoader(TensorDataset(torch.tensor(G[va["idx"]]),  torch.tensor(va["y"])), 256)}, seed=seed, class_weight=CW)
    # --- Full DEF ---
    deff = M.train_torch(M.DEF(W.shape[2], nC),
        {"train": wl(wsplit=="tr", G, shuffle=True), "val": wl(wsplit=="va", G)}, seed=seed, class_weight=CW)

    for name, mdl, loader, needs_g in [("LSTM", lstm, wl(wsplit=="te"), False),
                                       ("Transformer", trm, wl(wsplit=="te"), False),
                                       ("DEF", deff, wl(wsplit=="te", G), True)]:
        mdl.eval(); P, PR = [], []
        with torch.no_grad():
            for b in loader:
                b = [x.to(M.DEVICE) for x in b]
                logit = mdl(*b[:-1])
                PR.append(torch.softmax(logit, 1).cpu()); P.append(logit.argmax(1).cpu())
        P, PR = torch.cat(P).numpy(), torch.cat(PR).numpy()
        rows.append(dict(model=name, seed=seed, **metrics(Wy[wsplit=="te"], P, PR)))
        if seed == 42 and name == "DEF":
            np.save("results/probs_DEF.npy", PR)
            np.save("results/def_test_widx.npy", Widx[wsplit=="te"])
            torch.save(deff.state_dict(), "results/def_seed42.pt")
            print(classification_report(Wy[wsplit=="te"], P, target_names=classes))
            pd.DataFrame(classification_report(Wy[wsplit=="te"], P,
                target_names=classes, output_dict=True)).T.to_csv("results/per_class_seed42.csv")
    # GCN-only eval (flow-level)
    gcn.eval()
    with torch.no_grad():
        lg = gcn(torch.tensor(G[te["idx"]]).to(M.DEVICE))
    rows.append(dict(model="GCN", seed=seed,
                     **metrics(te["y"], lg.argmax(1).cpu().numpy(),
                               torch.softmax(lg,1).cpu().numpy())))
    pd.DataFrame(rows).to_csv("results/runs.csv", index=False)
    print(f"seed {seed} done")

df = pd.DataFrame(rows)
summary = df.groupby("model")[["acc","prec","rec","f1","fpr","auc"]].agg(["mean","std"]).round(2)
summary.to_csv("results/summary_mean_std.csv")
print(summary)

# significance: both directions on run-level macro F1
with open("results/significance.txt", "w") as f:
    for ref in ["XGBoost", "DEF"]:
        a = df[df.model == ref].sort_values("seed")["f1"].values
        for m in ["XGBoost", "RandomForest", "LSTM", "Transformer", "GCN", "DEF"]:
            if m == ref:
                continue
            b = df[df.model == m].sort_values("seed")["f1"].values
            t = ttest_rel(a, b)
            w = wilcoxon(a, b) if not np.allclose(a, b) else None
            f.write(f"{ref} vs {m}: paired t p={t.pvalue:.4g}"
                    + (f", wilcoxon p={w.pvalue:.4g}\n" if w else "\n"))
print(open("results/significance.txt").read())