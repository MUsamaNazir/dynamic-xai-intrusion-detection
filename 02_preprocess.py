"""
STEP 2 — Preprocess exactly as described in the paper.
Run:  python 02_preprocess.py
Outputs: data/train.npz, val.npz, test.npz, feature_names.txt (=> Appendix A),
         smote_counts.txt (before/after numbers for Section 4.2)
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

SEED = 42
df = pd.read_parquet("data/subset60k.parquet")

y_raw = df["Label"].values
meta_cols = [c for c in ["Label", "Timestamp", "Flow ID", "Src IP", "Dst IP", "Src Port", "__order"] if c in df.columns]
# CICIoT2023 has no IP/timestamp columns; use Protocol Type as the grouping key
meta = df[[c for c in ["Timestamp", "Src IP", "Dst IP", "Protocol Type", "__order"] if c in df.columns]].copy()
X = df.drop(columns=meta_cols, errors="ignore").apply(pd.to_numeric, errors="coerce")

# 1. clean
X = X.replace([np.inf, -np.inf], np.nan)
keep = X.isna().mean(axis=1) <= 0.5
X, y_raw, meta = X[keep], y_raw[keep.values], meta[keep.values]
X = X.fillna(X.median())
# drop constant and duplicate columns
X = X.loc[:, X.nunique() > 1]
X = X.loc[:, ~X.T.duplicated()]
# order the survivors by descending variance. All 37 are retained (the original
# head(48) was a no-op on count since 37 < 48), but the ordering is load-bearing:
# models.py build_graph_embed() uses column 0 as the local-activity proxy.
feats = X.var().sort_values(ascending=False).index.tolist()
X = X[feats]
open("data/feature_names.txt", "w").write("\n".join(feats))
print(f"[FEATURES] retained {len(feats)} (variance-ordered) after constant/duplicate removal")

le = LabelEncoder(); y = le.fit_transform(y_raw)
np.save("data/classes.npy", le.classes_)

# 2. split 70/15/15 stratified
idx = np.arange(len(X))
i_tr, i_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=SEED)

# 3. scale fitted on train only
sc = MinMaxScaler().fit(X.iloc[i_tr])
Xs = sc.transform(X)

# 4. SMOTE on train only
sm = SMOTE(random_state=SEED)
Xtr, ytr = sm.fit_resample(Xs[i_tr], y[i_tr])
open("data/smote_counts.txt", "w").write(
    f"train before SMOTE: {len(i_tr)}\ntrain after SMOTE: {len(Xtr)}\n")

np.savez("data/train.npz", X=Xtr, y=ytr)
np.savez("data/train_noSMOTE.npz", X=Xs[i_tr], y=y[i_tr], idx=i_tr)
np.savez("data/val.npz",  X=Xs[i_va], y=y[i_va], idx=i_va)
np.savez("data/test.npz", X=Xs[i_te], y=y[i_te], idx=i_te)
meta.reset_index(drop=True).to_parquet("data/meta.parquet")  # for windows + graphs
print("done. train:", Xtr.shape, "val:", len(i_va), "test:", len(i_te))
