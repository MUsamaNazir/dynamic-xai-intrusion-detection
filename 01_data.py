"""
STEP 1 - Build the 60k subset from CICIoT2023.
Data folder: D:/PHD_UTM/Datasets/wataiData/csv/CICIoT2023  (the part-*.csv files)
Run:  python 01_data.py
"""
import glob, os
import numpy as np
import pandas as pd

SEED = 42
CSV_DIR = r"D:\PHD_UTM\Datasets\wataiData\csv\CICIoT2023"
OUT = "./data"
os.makedirs(OUT, exist_ok=True)

# CICIoT2023 raw labels -> 8 paper classes
def map_label(l):
    l = str(l).strip()
    if l == "BenignTraffic":            return "Benign"
    if l.startswith("DDoS-"):           return "DDoS"
    if l.startswith("DoS-"):            return "DoS"
    if l.startswith("Mirai-"):          return "Mirai Botnet"
    if l.startswith("Recon-") or l == "VulnerabilityScan": return "Recon"
    if l in ("DNS_Spoofing", "MITM-ArpSpoofing"):          return "Spoofing"
    if l in ("XSS", "SQLInjection", "CommandInjection",
             "BrowserHijacking", "Uploading_Attack",
             "Backdoor_Malware"):       return "Web Attack"
    if l == "DictionaryBruteForce":     return "Brute Force"
    return None

TARGETS = {"Benign": 42000, "DDoS": 2700, "DoS": 2700, "Mirai Botnet": 2700,
           "Recon": 2700, "Spoofing": 2700, "Web Attack": 2700, "Brute Force": 2700}

rng = np.random.RandomState(SEED)
files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
assert files, f"No CSVs found in {CSV_DIR}"
print(f"{len(files)} csv files found")

# rare classes are scarce per file -> accumulate rare fully, sample common per-file
pools = {k: [] for k in TARGETS}
counts = {k: 0 for k in TARGETS}
offset = 0
for f in files:
    df = pd.read_csv(f, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["__order"] = np.arange(len(df)) + offset   # global chronological index
    offset += len(df)
    lab_col = "label" if "label" in df.columns else "Label"
    df["cls"] = df[lab_col].map(map_label)
    df = df.dropna(subset=["cls"]).drop(columns=[lab_col])
    for cls, g in df.groupby("cls"):
        need = TARGETS[cls] * 3 - counts[cls]          # keep ~3x target then downsample
        if need <= 0: continue
        take = g if len(g) <= need else g.sample(need, random_state=SEED)
        pools[cls].append(take); counts[cls] += len(take)
    print(os.path.basename(f), {k: counts[k] for k in counts})
    if all(counts[k] >= TARGETS[k] * 3 for k in TARGETS):
        print("enough of every class collected, stopping early"); break

sub = []
for cls, n in TARGETS.items():
    pool = pd.concat(pools[cls], ignore_index=True)
    take = min(n, len(pool))
    sub.append(pool.sample(take, random_state=SEED))
    print(f"{cls}: {take} sampled (pool {len(pool)})")
subset = pd.concat(sub, ignore_index=True).sort_values("__order").reset_index(drop=True)
subset = subset.rename(columns={"cls": "Label"})
subset.to_parquet(f"{OUT}/subset60k.parquet")
subset["Label"].value_counts().to_csv(f"{OUT}/class_counts.csv")
print("\nSaved data/subset60k.parquet - class_counts.csv is your new Table 2")
