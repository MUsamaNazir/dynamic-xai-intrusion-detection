"""
02b_preprocess_unsw.py - Preprocess the UNSW-NB15 subset using the SAME
protocol as 02_preprocess.py, so the cross-dataset comparison is like-for-like.

meta.parquet is written with columns "Src IP" / "Dst IP" / "Timestamp", which
activates the host-graph and true per-host-sequence branches of models.py.

Held out of the feature matrix (leakage control):
  Label, attack_cat  - direct target leaks
  srcip, dstip       - host identity (45-host testbed would be memorised)
  sport, dsport      - endpoint identity; CICIoT2023 has no ports either
  Stime, Ltime       - absolute time; attacks are time-clustered
`dur` is retained: a duration, not an absolute timestamp.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

SEED = 42
IN_PARQUET = "data/unsw_subset.parquet"
OUT = "data/unsw"
os.makedirs(OUT, exist_ok=True)

IDENTIFIERS = ["srcip", "dstip", "sport", "dsport", "Stime", "Ltime"]
LABEL_COLS = ["label", "attack_cat", "Label"]
CATEGORICALS = ["proto", "state", "service"]

print("=" * 70)
print("  STEP 2b  Preprocess UNSW-NB15")
print("=" * 70)

df = pd.read_parquet(IN_PARQUET)
print(f"loaded {IN_PARQUET}: {df.shape}")
print(f"chronological on Stime: {df['Stime'].is_monotonic_increasing}")

y_raw = df["label"].astype(str).values
le = LabelEncoder()
y = le.fit_transform(y_raw)
np.save(f"{OUT}/classes.npy", le.classes_)
print(f"\nclasses ({len(le.classes_)}): {list(le.classes_)}")

meta = pd.DataFrame({
    "Src IP":    df["srcip"].astype(str).values,
    "Dst IP":    df["dstip"].astype(str).values,
    "Timestamp": pd.to_datetime(df["Stime"].astype("int64"), unit="s"),
})
print(f"\nmeta: {meta.shape}  hosts: {meta['Src IP'].nunique()} src / "
      f"{meta['Dst IP'].nunique()} dst")
print(f"time span: {meta['Timestamp'].min()} .. {meta['Timestamp'].max()}")

drop = [c for c in IDENTIFIERS + LABEL_COLS + ["__order"] if c in df.columns]
X = df.drop(columns=drop, errors="ignore").copy()
print(f"\ndropped {len(drop)} held-aside columns: {drop}")

for c in CATEGORICALS:
    if c in X.columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str).fillna("none"))
        print(f"  label-encoded {c}: {X[c].nunique()} codes")

X = X.apply(pd.to_numeric, errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)
keep = X.isna().mean(axis=1) <= 0.5
X, y, meta, y_raw = X[keep], y[keep.values], meta[keep.values], y_raw[keep.values]
X = X.fillna(X.median())
print(f"\nrows surviving cleaning: {keep.sum():,} of {len(keep):,}")

n0 = X.shape[1]
X = X.loc[:, X.nunique() > 1]
n1 = X.shape[1]
X = X.loc[:, ~X.T.duplicated()]
n2 = X.shape[1]
feats = X.var().sort_values(ascending=False).index.tolist()
X = X[feats]
open(f"{OUT}/feature_names.txt", "w").write("\n".join(feats))
print(f"features: {n0} raw -> {n1} non-constant -> {n2} non-duplicate")
print(f"top-5 by variance: {feats[:5]}")

idx = np.arange(len(X))
i_tr, i_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp],
                              random_state=SEED)

sc = MinMaxScaler().fit(X.iloc[i_tr])
Xs = sc.transform(X).astype("float32")

tr_counts = pd.Series(y[i_tr]).value_counts()
k = int(max(1, min(5, tr_counts.min() - 1)))
print(f"\nsmallest training class: {tr_counts.min()} "
      f"({le.classes_[tr_counts.idxmin()]}) -> SMOTE k_neighbors={k}")
sm = SMOTE(random_state=SEED, k_neighbors=k)
Xtr, ytr = sm.fit_resample(Xs[i_tr], y[i_tr])
open(f"{OUT}/smote_counts.txt", "w").write(
    f"train before SMOTE: {len(i_tr)}\ntrain after SMOTE: {len(Xtr)}\n"
    f"k_neighbors: {k}\n")

np.savez(f"{OUT}/train.npz", X=Xtr, y=ytr)
np.savez(f"{OUT}/train_noSMOTE.npz", X=Xs[i_tr], y=y[i_tr], idx=i_tr)
np.savez(f"{OUT}/val.npz",  X=Xs[i_va], y=y[i_va], idx=i_va)
np.savez(f"{OUT}/test.npz", X=Xs[i_te], y=y[i_te], idx=i_te)
meta.reset_index(drop=True).to_parquet(f"{OUT}/meta.parquet")

print("\n" + "=" * 70)
print(f"train (SMOTE) : {Xtr.shape}")
print(f"train (raw)   : {len(i_tr):,}")
print(f"val           : {len(i_va):,}")
print(f"test          : {len(i_te):,}")
print(f"features      : {X.shape[1]}   (CICIoT2023 used 37)")
print(f"\nwrote {OUT}/")
print("=" * 70)
