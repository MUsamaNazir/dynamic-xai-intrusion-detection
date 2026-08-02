"""
03b_experiments_unsw.py
-----------------------
STEP 3b - Train and evaluate on UNSW-NB15, reusing models.py unchanged.

Everything that could affect a number is identical to 03_experiments.py:
the same SEEDS, the same XGBoost/RandomForest hyperparameters, the same
models.py classes and the same train_torch() loop. Only the data paths,
the benign class name and the output folder differ.

Run in stages, because the neural models are slow on CPU:

    python 03b_experiments_unsw.py tabular    # XGBoost + RandomForest, ~10 min
    python 03b_experiments_unsw.py neural     # LSTM/Transformer/GCN/Fusion, hours
    python 03b_experiments_unsw.py all

Outputs (results/unsw/):
    runs_<stage>.csv          per-seed metrics
    runs.csv                  merged across stages
    summary_mean_std.csv      mean +/- std  -> cross-dataset table
    significance.txt          paired t / Wilcoxon, both directions
    per_class_seed42.csv      per-class precision/recall/F1
    probs_XGB.npy             backbone probabilities -> 04b prioritization
    xgb_seed42.json           saved backbone
    probs_DEF.npy             fusion probabilities (neural stage)
    def_test_widx.npy         window -> original row map
    def_seed42.pt             saved fusion model
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix,
                             classification_report)
from scipy.stats import ttest_rel, wilcoxon
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

import models as M

# ---------------------------------------------------------------- config
STAGE = (sys.argv[1] if len(sys.argv) > 1 else "tabular").lower()
assert STAGE in ("tabular", "neural", "all"), f"bad stage: {STAGE}"

# Optional: restrict models and seeds.
#   python 03b_experiments_unsw.py neural --models Transformer,DEF --seeds 13,42,101
def _arg(flag, default=None):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default

ONLY_MODELS = _arg("--models")
ONLY_MODELS = [m.strip() for m in ONLY_MODELS.split(",")] if ONLY_MODELS else None
ONLY_SEEDS = _arg("--seeds")
ONLY_SEEDS = [int(s) for s in ONLY_SEEDS.split(",")] if ONLY_SEEDS else None
RESUME = "--no-resume" not in sys.argv

DATA = "data/unsw"
RESULTS = "results/unsw"
BENIGN = "Normal"
SEEDS = [1, 7, 13, 42, 101]          # identical to 03_experiments.py
if ONLY_SEEDS:
    SEEDS = ONLY_SEEDS
os.makedirs(RESULTS, exist_ok=True)

LEDGER = f"{RESULTS}/runs_ledger.csv"


def load_ledger():
    """Every completed (model, seed) result written so far."""
    if os.path.exists(LEDGER):
        return pd.read_csv(LEDGER)
    # migrate any pre-existing per-stage files into the ledger
    frames = []
    for f in ("runs_tabular.csv", "runs_neural.csv", "runs_all.csv", "runs.csv"):
        p = f"{RESULTS}/{f}"
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if frames:
        led = (pd.concat(frames, ignore_index=True)
                 .drop_duplicates(subset=["model", "seed"], keep="last"))
        led.to_csv(LEDGER, index=False)
        print(f"[LEDGER] migrated {len(led)} existing results into "
              f"runs_ledger.csv")
        return led
    return pd.DataFrame(columns=["model", "seed"])


rows = []
LED = load_ledger()
DONE = set(zip(LED.get("model", []), LED.get("seed", [])))
if DONE and RESUME:
    print(f"[RESUME] {len(DONE)} (model, seed) results already complete")


def record(model, seed, m):
    """Append one result and flush to disk immediately."""
    global LED
    row = dict(model=model, seed=seed, **m)
    LED = pd.concat([LED[~((LED.model == model) & (LED.seed == seed))],
                     pd.DataFrame([row])], ignore_index=True)
    LED.to_csv(LEDGER, index=False)
    rows.append(row)


def skip(model, seed):
    if ONLY_MODELS and model not in ONLY_MODELS:
        return True
    if RESUME and (model, seed) in DONE:
        print(f"  {model:12s} already done for seed {seed}, skipping",
              flush=True)
        return True
    return False

tr = np.load(f"{DATA}/train.npz")
trn = np.load(f"{DATA}/train_noSMOTE.npz")
va = np.load(f"{DATA}/val.npz")
te = np.load(f"{DATA}/test.npz")
meta = pd.read_parquet(f"{DATA}/meta.parquet")
classes = np.load(f"{DATA}/classes.npy", allow_pickle=True)
nC = len(classes)

print("=" * 72)
print(f"  STEP 3b  UNSW-NB15 experiments  [stage: {STAGE}]")
print("=" * 72)
print(f"classes ({nC}): {list(classes)}")
print(f"benign class   : {BENIGN} (index "
      f"{int(np.where(classes == BENIGN)[0][0])})")
print(f"train (SMOTE)  : {tr['X'].shape}")
print(f"test           : {te['X'].shape}")
print(f"device         : {M.DEVICE}")
print(f"window/stride  : T={M.T_WIN}, stride={M.STRIDE}")
print("=" * 72, flush=True)


def metrics(y, p, prob=None):
    """Identical metric definitions to 03_experiments.py."""
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, p, average="macro", zero_division=0)
    cm = confusion_matrix(y, p, labels=np.arange(nC))
    fpr = np.mean([(cm[:, c].sum() - cm[c, c]) /
                   max(1, cm.sum() - cm[c].sum()) for c in range(nC)])
    try:
        auc = (roc_auc_score(y, prob, multi_class="ovr", average="macro")
               if prob is not None else np.nan)
    except ValueError:
        auc = np.nan          # a class may be absent from a window split
    return dict(acc=accuracy_score(y, p) * 100, prec=pr * 100, rec=rc * 100,
                f1=f1 * 100, fpr=fpr * 100, auc=auc)


# ------------------------------------------- sequence windows + host graph
need_neural = STAGE in ("neural", "all")
if need_neural:
    print("\nbuilding windows and host graph ...", flush=True)
    t0 = time.time()
    nF = tr["X"].shape[1]
    full_scaled = np.zeros((len(meta), nF), dtype="float32")
    full_y = np.zeros(len(meta), dtype=int)
    for part in (trn, va, te):
        full_scaled[part["idx"]] = part["X"]
        full_y[part["idx"]] = part["y"]

    W, Wy, Widx = M.build_windows(full_scaled, meta, full_y)
    G = M.build_graph_embed(full_scaled, meta)
    split = np.full(len(meta), "tr", dtype=object)
    split[va["idx"]] = "va"
    split[te["idx"]] = "te"
    wsplit = split[Widx]
    print(f"  windows {W.shape}  graph {G.shape}  "
          f"({time.time()-t0:.0f}s, {W.nbytes/1e6:.0f} MB)")
    print(f"  window split: tr={np.sum(wsplit=='tr'):,} "
          f"va={np.sum(wsplit=='va'):,} te={np.sum(wsplit=='te'):,}", flush=True)

    def wl(mask, extra=None, bs=256, shuffle=False):
        tens = [torch.tensor(W[mask])]
        if extra is not None:
            tens.append(torch.tensor(extra[Widx][mask], dtype=torch.float32))
        tens.append(torch.tensor(Wy[mask]))
        return DataLoader(TensorDataset(*tens), batch_size=bs, shuffle=shuffle)


# ---------------------------------------------------------------- run
rows = []  # defined above use in record()
for seed in SEEDS:
    t_seed = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"\n{'-'*72}\nseed {seed}\n{'-'*72}", flush=True)

    if STAGE in ("tabular", "all") and not skip("XGBoost", seed):
        # --- XGBoost (the DEF backbone) ---
        t0 = time.time()
        xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            subsample=0.8, random_state=seed,
                            tree_method="hist")
        xgb.fit(tr["X"], tr["y"])
        p, pr = xgb.predict(te["X"]), xgb.predict_proba(te["X"])
        m = metrics(te["y"], p, pr)
        record("XGBoost", seed, m)
        print(f"  XGBoost      acc {m['acc']:6.2f}  F1 {m['f1']:6.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if seed == 42:
            xgb.save_model(f"{RESULTS}/xgb_seed42.json")
            np.save(f"{RESULTS}/probs_XGB.npy", pr)
            pd.DataFrame(classification_report(
                te["y"], p, labels=np.arange(nC), target_names=classes,
                output_dict=True, zero_division=0)).T.to_csv(
                    f"{RESULTS}/per_class_XGB_seed42.csv")

    # --- Random Forest ---
    if STAGE in ("tabular", "all") and not skip("RandomForest", seed):
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                    random_state=seed, n_jobs=-1)
        rf.fit(tr["X"], tr["y"])
        m = metrics(te["y"], rf.predict(te["X"]), rf.predict_proba(te["X"]))
        record("RandomForest", seed, m)
        print(f"  RandomForest acc {m['acc']:6.2f}  F1 {m['f1']:6.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    if need_neural:
        pend = [m for m in ("LSTM", "Transformer", "GCN", "DEF")
                if not skip(m, seed)]
        if pend:
            loaders_seq = {"train": wl(wsplit == "tr", shuffle=True),
                           "val":   wl(wsplit == "va")}
            te_seq = wl(wsplit == "te")

        # -- sequence models: train, evaluate and CHECKPOINT one at a time --
        for name, ctor in (("LSTM", lambda: M.LSTMNet(W.shape[2], nC)),
                           ("Transformer",
                            lambda: M.SeqTransformer(W.shape[2], nC))):
            if name not in pend:
                continue
            t0 = time.time()
            mdl = M.train_torch(ctor(), loaders_seq, seed=seed)
            mdl.eval()
            P, PR = [], []
            with torch.no_grad():
                for b in te_seq:
                    b = [x.to(M.DEVICE) for x in b]
                    lg = mdl(*b[:-1])
                    PR.append(torch.softmax(lg, 1).cpu())
                    P.append(lg.argmax(1).cpu())
            m = metrics(Wy[wsplit == "te"],
                        torch.cat(P).numpy(), torch.cat(PR).numpy())
            record(name, seed, m)
            print(f"  {name:12s} acc {m['acc']:6.2f}  F1 {m['f1']:6.2f}  "
                  f"({time.time()-t0:.0f}s)  [saved]", flush=True)
            del mdl

        # -- topology-context encoder (flow-level) --
        if "GCN" in pend:
            t0 = time.time()
            gcn = M.train_torch(
                M.GCNHead(nC),
                {"train": DataLoader(TensorDataset(
                     torch.tensor(G[trn["idx"]]), torch.tensor(trn["y"])),
                     256, shuffle=True),
                 "val": DataLoader(TensorDataset(
                     torch.tensor(G[va["idx"]]), torch.tensor(va["y"])), 256)},
                seed=seed)
            gcn.eval()
            with torch.no_grad():
                lg = gcn(torch.tensor(G[te["idx"]]).to(M.DEVICE))
            m = metrics(te["y"], lg.argmax(1).cpu().numpy(),
                        torch.softmax(lg, 1).cpu().numpy())
            record("GCN", seed, m)
            print(f"  {'GCN':12s} acc {m['acc']:6.2f}  F1 {m['f1']:6.2f}  "
                  f"({time.time()-t0:.0f}s)  [saved]", flush=True)
            del gcn

        # -- Transformer + topology fusion --
        if "DEF" in pend:
            t0 = time.time()
            loaders_fus = {"train": wl(wsplit == "tr", G, shuffle=True),
                           "val":   wl(wsplit == "va", G)}
            deff = M.train_torch(M.DEF(W.shape[2], nC), loaders_fus, seed=seed)
            deff.eval()
            P, PR = [], []
            with torch.no_grad():
                for b in wl(wsplit == "te", G):
                    b = [x.to(M.DEVICE) for x in b]
                    lg = deff(*b[:-1])
                    PR.append(torch.softmax(lg, 1).cpu())
                    P.append(lg.argmax(1).cpu())
            P, PR = torch.cat(P).numpy(), torch.cat(PR).numpy()
            m = metrics(Wy[wsplit == "te"], P, PR)
            record("DEF", seed, m)
            print(f"  {'DEF':12s} acc {m['acc']:6.2f}  F1 {m['f1']:6.2f}  "
                  f"({time.time()-t0:.0f}s)  [saved]", flush=True)
            if seed == 42:
                np.save(f"{RESULTS}/probs_DEF.npy", PR)
                np.save(f"{RESULTS}/def_test_widx.npy", Widx[wsplit == "te"])
                torch.save(deff.state_dict(), f"{RESULTS}/def_seed42.pt")
                pd.DataFrame(classification_report(
                    Wy[wsplit == "te"], P, labels=np.arange(nC),
                    target_names=classes, output_dict=True,
                    zero_division=0)).T.to_csv(
                        f"{RESULTS}/per_class_seed42.csv")
            del deff, loaders_fus

    print(f"  seed {seed} done in {time.time()-t_seed:.0f}s", flush=True)

# ---------------------------------------------------------------- summarise
merged = pd.read_csv(LEDGER).drop_duplicates(subset=["model", "seed"],
                                             keep="last")
merged.to_csv(f"{RESULTS}/runs.csv", index=False)

summary = merged.groupby("model")[
    ["acc", "prec", "rec", "f1", "fpr", "auc"]].agg(["mean", "std"]).round(2)
summary.to_csv(f"{RESULTS}/summary_mean_std.csv")
print("\n" + "=" * 72)
print(summary.to_string())

counts = merged.groupby("model")["seed"].nunique()
incomplete = counts[counts < 5]
if len(incomplete):
    print("\nINCOMPLETE (need 5 seeds):")
    for mo, n in incomplete.items():
        got = sorted(merged[merged.model == mo]["seed"].tolist())
        missing = [s for s in [1, 7, 13, 42, 101] if s not in got]
        print(f"  {mo:12s} {n}/5   missing seeds: {missing}")
    print("\n  resume with:  python 03b_experiments_unsw.py neural")

present = [m for m in merged["model"].unique() if counts[m] == 5]
with open(f"{RESULTS}/significance.txt", "w") as f:
    for ref in [r for r in ("XGBoost", "DEF") if r in present]:
        a = merged[merged.model == ref].sort_values("seed")["f1"].values
        for mo in present:
            if mo == ref:
                continue
            b = merged[merged.model == mo].sort_values("seed")["f1"].values
            t = ttest_rel(a, b)
            w = wilcoxon(a, b) if not np.allclose(a, b) else None
            f.write(f"{ref} vs {mo}: paired t p={t.pvalue:.4g}"
                    + (f", wilcoxon p={w.pvalue:.4g}\n" if w else "\n"))
print("\n" + open(f"{RESULTS}/significance.txt").read())
print(f"ledger: {LEDGER}  ({len(merged)} results)")
print("=" * 72)
