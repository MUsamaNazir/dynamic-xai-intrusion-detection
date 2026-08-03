"""
def_unsw_loader.py
------------------
Loader + label harmonisation for UNSW-NB15, for the DEF cross-dataset experiment
(Scientific Reports revision, Reviewer 1 comment on single-dataset evaluation).

Handles BOTH official distributions:

  A) "full"  : UNSW-NB15_1.csv .. UNSW-NB15_4.csv
               49 columns, NO header row, HAS srcip / dstip / Stime / Ltime.
               ---> required for the true per-host sequence + flow-graph experiment.

  B) "split" : UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv
               ~45 columns, header row present, NO IPs, NO timestamps.
               ---> usable for the generalisation table only.

Usage
-----
    import def_unsw_loader as L

    layout, files = L.detect_layout("data/unswnb15")
    df   = L.load_unsw("data/unswnb15")
    sub  = L.stratified_subset(df, total=60000, benign_frac=0.70, seed=42)
    L.report(sub)
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Encoding. The official UNSW-NB15 CSVs were written on Windows and are
# NOT UTF-8: they contain cp1252 bytes such as 0x92 (curly apostrophe),
# which makes a default pd.read_csv() raise UnicodeDecodeError.
# We try cp1252 first (correct rendering), then fall back to latin-1,
# which cannot fail because every byte 0x00-0xFF maps to a codepoint.
# ----------------------------------------------------------------------
ENCODINGS = ("cp1252", "latin-1")


def _read_csv_robust(path, **kwargs):
    """pd.read_csv with encoding fallback for the cp1252-encoded source files."""
    last_err = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    # final safety net: replace undecodable bytes rather than dying
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **kwargs)


def scan_encoding(path, max_report=10):
    """Diagnostic: report byte offsets of non-ASCII bytes in a file.

    Useful for confirming *why* a file will not read as UTF-8 and which
    column the offending characters live in.
    """
    path = Path(path)
    hits = []
    with open(path, "rb") as fh:
        for lineno, raw in enumerate(fh, start=1):
            for off, b in enumerate(raw):
                if b > 127:
                    hits.append((lineno, off, hex(b), chr(b) if b < 256 else "?"))
                    break
            if len(hits) >= max_report:
                break
    if not hits:
        print(f"[ENC] {path.name}: pure ASCII, any encoding will work")
    else:
        print(f"[ENC] {path.name}: non-ASCII bytes found (first {len(hits)}):")
        for lineno, off, bhex, ch in hits:
            print(f"       line {lineno:>6}  byte offset {off:>4}  {bhex}  -> '{ch}' (cp1252)")
    return hits


def normalise_text(df, verbose=True):
    """Strip non-ASCII characters from categorical/text columns.

    Guarantees that label mapping and one-hot encoding are deterministic
    regardless of which fallback encoding was used to read the file.
    """
    df = df.copy()
    touched = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = df[col].astype("string")
        # Strip BOM first. UNSW-NB15_1.csv begins with a UTF-8 BOM (EF BB BF),
        # which decodes under cp1252 as "i>>?"-style mojibake. NFKD is NOT used,
        # because it would transliterate the leading char into an ASCII letter
        # and silently corrupt the first srcip value.
        cleaned = (s.str.replace("\ufeff", "", regex=False)
                    .str.replace("\u00ef\u00bb\u00bf", "", regex=False)
                    .str.encode("ascii", "ignore")
                    .str.decode("ascii")
                    .str.strip())
        if not cleaned.equals(s.str.strip()):
            touched.append(col)
        df[col] = cleaned
    if verbose and touched:
        print(f"[TEXT] stripped non-ASCII from: {', '.join(touched)}")
    return df

# ----------------------------------------------------------------------
# Column names for the 49-column "full" distribution.
# The official NUSW-NB15_features.csv lists these in order; hard-coded
# here so the pipeline does not depend on that auxiliary file being present.
# ----------------------------------------------------------------------
UNSW49_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "Label",
]

# Identifier / leakage columns: kept through loading (the sequence and graph
# modules need them) but must be excluded from the model feature matrix.
ID_COLUMNS = ["srcip", "dstip", "sport", "dsport", "Stime", "Ltime"]

# ----------------------------------------------------------------------
# Label harmonisation.
# In the raw files attack_cat is blank for benign traffic and carries
# inconsistent spellings / stray whitespace for several attack families.
# Without this map you silently end up with ~13 classes instead of 10.
# ----------------------------------------------------------------------
ATTACK_CAT_CANON = {
    "": "Normal",
    "nan": "Normal",
    "none": "Normal",
    "normal": "Normal",
    "analysis": "Analysis",
    "backdoor": "Backdoor",
    "backdoors": "Backdoor",          # both spellings occur
    "dos": "DoS",
    "exploits": "Exploits",
    "fuzzers": "Fuzzers",
    "generic": "Generic",
    "reconnaissance": "Reconnaissance",
    "shellcode": "Shellcode",
    "worms": "Worms",
}

CANONICAL_CLASSES = [
    "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
    "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
]


# ======================================================================
# Layout detection
# ======================================================================
def detect_layout(data_dir="data/unswnb15", verbose=True):
    """Return ('full' | 'split' | 'unknown', [Path, ...])."""
    data_dir = Path(data_dir)
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {data_dir.resolve()}")

    full, split = [], []
    for f in csvs:
        low = f.name.lower()
        if "features" in low:                 # NUSW-NB15_features.csv
            continue
        if "training-set" in low or "testing-set" in low or "train" in low or "test" in low:
            split.append(f)
        else:
            ncols = _read_csv_robust(f, nrows=2, header=None, low_memory=False).shape[1]
            (full if ncols >= 47 else split).append(f)

    if full:
        layout, files = "full", full
    elif split:
        layout, files = "split", split
    else:
        layout, files = "unknown", csvs

    if verbose:
        print(f"[LAYOUT] detected '{layout}' from {len(files)} file(s):")
        for f in files:
            print(f"         {f.name}  ({f.stat().st_size / 1e6:,.1f} MB)")
        if layout == "split":
            print("[LAYOUT] WARNING: this distribution has NO srcip/dstip/Stime.")
            print("         Cross-dataset generalisation: OK.")
            print("         True per-host sequences / flow graph: NOT possible.")
    return layout, files


# ======================================================================
# Quirk cleaning
# ======================================================================
def _to_num(series):
    """Coerce a messy column to numeric.

    Handles the known UNSW-NB15 quirks:
      * sport / dsport contain hex strings ('0x000b') and '-' placeholders
      * ct_ftp_cmd contains ' ' (space) for 'not applicable'
    """
    s = series.astype(str).str.strip()
    out = pd.to_numeric(s, errors="coerce")
    hexmask = out.isna() & s.str.lower().str.startswith("0x")
    if hexmask.any():
        out.loc[hexmask] = s[hexmask].apply(
            lambda v: int(v, 16) if v not in ("", "-") else np.nan
        )
    return pd.to_numeric(out, errors="coerce")


def clean_quirks(df, verbose=True):
    """Fix the documented dtype/value problems in the raw UNSW-NB15 CSVs."""
    df = df.copy()
    fixed = []

    for col in ["sport", "dsport", "ct_ftp_cmd"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = _to_num(df[col])
            fixed.append(col)

    # is_ftp_login is documented as binary but contains 2 and 4 in places
    if "is_ftp_login" in df.columns:
        df["is_ftp_login"] = (_to_num(df["is_ftp_login"]).fillna(0) > 0).astype(int)
        fixed.append("is_ftp_login")

    # These count columns are NaN where the protocol does not apply -> 0
    for col in ["ct_flw_http_mthd", "ct_ftp_cmd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if verbose and fixed:
        print(f"[CLEAN] coerced to numeric: {', '.join(sorted(set(fixed)))}")
    return df


def harmonise_labels(df, verbose=True):
    """Map attack_cat to the 10 canonical classes; store as 'label'."""
    df = df.copy()
    src = "attack_cat" if "attack_cat" in df.columns else "label"
    # fillna BEFORE astype(str): under pandas>=3 NA survives astype(str) and
    # would otherwise be reported as an unmapped category.
    raw = df[src].fillna("").astype(str).str.strip().str.lower()

    unmapped = sorted(v for v in raw.unique() if v not in ATTACK_CAT_CANON)
    if unmapped and verbose:
        print(f"[LABELS] WARNING unmapped attack_cat values: {unmapped}")

    df["label"] = raw.map(ATTACK_CAT_CANON).fillna("Normal")

    if verbose:
        n_before = raw.nunique()
        print(f"[LABELS] {n_before} raw value(s) -> {df['label'].nunique()} canonical class(es)")
    return df


# ======================================================================
# Loaders
# ======================================================================
def load_unsw_full(files, nrows=None, verbose=True, dedup=True):
    """Load the 49-column headerless distribution (keeps IPs and timestamps)."""
    frames = []
    for f in files:
        df = _read_csv_robust(
            f, header=None, names=UNSW49_COLUMNS,
            low_memory=False, nrows=nrows,
            na_values=["", " ", "-"], skipinitialspace=True,
        )
        if verbose:
            print(f"[LOAD] {f.name}: {df.shape}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = normalise_text(df, verbose=verbose)
    df = clean_quirks(df, verbose=verbose)
    df = harmonise_labels(df, verbose=verbose)
    if dedup:
        df = deduplicate(df, verbose=verbose)
    return df


def load_unsw_split(files, verbose=True):
    """Load the pre-split distribution (has header, no IPs, no timestamps)."""
    frames = []
    for f in files:
        df = _read_csv_robust(f, low_memory=False)
        if verbose:
            print(f"[LOAD] {f.name}: {df.shape}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in ["id"] if c in df.columns])
    df = normalise_text(df, verbose=verbose)
    df = clean_quirks(df, verbose=verbose)
    df = harmonise_labels(df, verbose=verbose)
    return df


def load_unsw(data_dir="data/unswnb15", nrows=None, verbose=True, dedup=True):
    """Dispatch on detected layout. Returns a DataFrame with a 'label' column."""
    layout, files = detect_layout(data_dir, verbose=verbose)
    if layout == "full":
        df = load_unsw_full(files, nrows=nrows, verbose=verbose, dedup=dedup)
    elif layout == "split":
        df = load_unsw_split(files, verbose=verbose)
    else:
        raise ValueError(f"Could not identify UNSW-NB15 layout in {data_dir}")
    df.attrs["layout"] = layout
    df.attrs["has_host_time"] = all(c in df.columns for c in ["srcip", "dstip", "Stime"])
    if verbose:
        print(f"[LOAD] total {df.shape}; host/time fields available: "
              f"{df.attrs['has_host_time']}")
    return df


# ======================================================================
# Stratified subset, matched to the CICIoT2023 protocol
# ======================================================================
def stratified_subset(df, total=60900, benign_frac=0.6897, benign_name="Normal",
                      seed=42, preserve_time_order=True, match_ratio=True,
                      verbose=True):
    """Draw a class-stratified subset matched to the CICIoT2023 protocol.

    Benign is capped at `benign_frac` of `total`. The remaining budget is
    distributed across attack families by WATER-FILLING: the per-class cap is
    raised iteratively, classes whose pool is smaller than the current cap are
    taken in full, and their unused allowance is redistributed to the classes
    that still have capacity. This keeps the benign share (and therefore the
    class prior that FPR and precision depend on) matched across datasets even
    though UNSW-NB15 has classes as small as Worms (174 records).
    """
    rng = np.random.RandomState(seed)
    n_benign = int(round(total * benign_frac))
    attack_classes = sorted(c for c in df["label"].unique() if c != benign_name)

    avail = {c: int((df["label"] == c).sum()) for c in attack_classes}
    budget = total - n_benign
    alloc, pending = {}, set(attack_classes)

    while pending:
        cap = budget / len(pending)
        small = {c for c in pending if avail[c] <= cap}
        if not small:                          # every remaining pool exceeds cap
            per = int(budget // len(pending))
            for c in sorted(pending):
                alloc[c] = per
            break
        for c in small:                        # take small pools in full
            alloc[c] = avail[c]
            budget -= avail[c]
        pending -= small

    # If attack pools ran short, scale benign down so the class PRIOR is
    # preserved rather than the absolute count. The benign/attack ratio is what
    # FPR and macro precision depend on, so matching it is what makes the
    # cross-dataset comparison like-for-like.
    n_attack = sum(alloc.values())
    if match_ratio and n_attack < (total - n_benign):
        n_benign = int(round(n_attack * benign_frac / (1.0 - benign_frac)))
        if verbose:
            print(f"[SUBSET] attack pools short ({n_attack:,} of "
                  f"{total - int(round(total*benign_frac)):,}); "
                  f"benign scaled to {n_benign:,} to hold the "
                  f"{100*benign_frac:.1f}% prior")

    parts = []
    for cls, take in [(benign_name, min(n_benign, int((df["label"] == benign_name).sum())))] \
                     + sorted(alloc.items()):
        pool = df.index[df["label"] == cls]
        take = min(take, len(pool))
        if take > 0:
            parts.append(rng.choice(pool, size=take, replace=False))

    idx = np.concatenate(parts)
    if preserve_time_order and "Stime" in df.columns:
        sub = df.loc[idx].sort_values("Stime", kind="mergesort")
    else:
        sub = df.loc[np.sort(idx)]
    sub = sub.reset_index(drop=True)

    if verbose:
        got_benign = int((sub["label"] == benign_name).sum())
        print(f"[SUBSET] {len(sub):,} rows (target {total:,}); "
              f"{benign_name} {got_benign:,} = {100*got_benign/len(sub):.1f}% "
              f"(target {100*benign_frac:.1f}%)")
        exhausted = [c for c in attack_classes if alloc.get(c, 0) >= avail[c]]
        if exhausted:
            print(f"[SUBSET] pools exhausted (all rows taken): "
                  f"{', '.join(f'{c}={avail[c]:,}' for c in exhausted)}")
    return sub



def deduplicate(df, verbose=True):
    """Remove exact duplicate records.

    The full UNSW-NB15 CSVs contain a substantial number of byte-identical
    rows (~19% of the corpus). Because every field including srcip, dsport,
    Stime and Ltime is repeated, these are true duplicate records rather
    than distinct flows that happen to share a feature vector. Left in
    place they leak between train and test folds under any random split.
    """
    n0 = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    if verbose:
        removed = n0 - len(out)
        print(f"[DEDUP] removed {removed:,} exact duplicate rows "
              f"({100*removed/max(n0,1):.1f}%) -> {len(out):,} unique records")
    return out


def report(df):
    """Print the class distribution table for the manuscript."""
    counts = df["label"].value_counts()
    total = len(df)
    print(f"\n{'Class':<16}{'Samples':>10}{'Share':>9}")
    print("-" * 35)
    for cls in CANONICAL_CLASSES:
        if cls in counts:
            n = counts[cls]
            print(f"{cls:<16}{n:>10,}{100 * n / total:>8.1f}%")
    print("-" * 35)
    print(f"{'Total':<16}{total:>10,}{100.0:>8.1f}%")
    if "srcip" in df.columns:
        print(f"\nUnique source hosts:      {df['srcip'].nunique():,}")
        print(f"Unique destination hosts: {df['dstip'].nunique():,}")
        span = df["Stime"].max() - df["Stime"].min()
        print(f"Capture span:             {span:,.0f} s "
              f"({span / 3600:.1f} h)")


if __name__ == "__main__":
    d = load_unsw("data/unswnb15")
    s = stratified_subset(d)
    report(s)