# Dynamic Explainable Framework (DEF) for Intrusion Detection

Code and processed data for:

> M. U. Nazir and A. bin Ngadi, "A Dynamic Explainable AI Framework for
> Real-Time Intrusion Detection and Automated Alert Prioritization in
> Security Operation Centers," *Scientific Reports* (under review).

The DEF couples an XGBoost detection backbone with inference-time SHAP and
LIME explanations and an explanation-aware alert prioritization layer. It is
evaluated on two benchmarks under one protocol: **CICIoT2023** (8 classes)
and **UNSW-NB15** (10 classes).

---

## Headline results

| | CICIoT2023 | UNSW-NB15 |
|---|---|---|
| Subset | 60,900 flows, 8 classes | 60,889 flows, 10 classes |
| Features retained | 37 | 41 |
| Accuracy | 94.05 ± 0.08 % | 88.26 ± 0.07 % |
| Macro F1 | 88.49 ± 0.13 % | 68.22 ± 0.22 % |
| False-positive rate | 1.29 % | 1.25 % |
| Macro AUC-ROC | 0.99 | 0.98 |
| Alert Reduction Rate (θ=0.5) | 69.7 % | 68.2 % |
| Baseline macro recall retained | 99.41 % | 99.67 % |

SHAP TreeExplainer 1.8 ms/alert (fidelity ρ = 0.79); LIME 146.2 ms
(top-10 Jaccard stability 0.62); combined detection + attribution 2.68 ms,
373 alerts/s, 0.90 GB process RSS, CPU only.

All figures are means over five seeds `{1, 7, 13, 42, 101}`.

---

## Repository layout

```
.
├── 01_data.py                    CICIoT2023: build the 60,900-flow subset
├── 02_preprocess.py              CICIoT2023: clean, prune, split, scale, SMOTE
├── 03_experiments.py             CICIoT2023: 6 configurations x 5 seeds
├── 04_prioritization.py          CICIoT2023: weight tuning + theta sweep
├── 05_xai_runtime_figures.py     CICIoT2023: XAI metrics, runtime, figures
│
├── def_unsw_loader.py            UNSW-NB15: loader, dedup, stratified subset
├── 02b_preprocess_unsw.py        UNSW-NB15: preprocessing (same protocol)
├── 03b_experiments_unsw.py       UNSW-NB15: experiments (resumable)
├── 04b_prioritization_unsw.py    UNSW-NB15: prioritization
│
├── models.py                     shared: windows, host graph, torch models
├── verify_manuscript.py          audit: artifacts vs reported values
│
├── data/                         CICIoT2023 subset, splits, feature list
│   ├── unsw/                     UNSW-NB15 subset, splits, feature list
│   └── unswnb15/                 raw UNSW-NB15 CSVs (NOT committed)
├── results/                      CICIoT2023 metrics, sweeps, saved models
│   └── unsw/                     UNSW-NB15 metrics, sweeps, run ledger
└── Figures/                      manuscript figures
```

Paths are relative and match what the scripts read and write: the
CICIoT2023 pipeline uses `data/` and `results/` directly, while the
UNSW-NB15 pipeline uses `data/unsw/` and `results/unsw/`.

**Run every script from the repository root**, not from a subdirectory.
All paths are relative to the root.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11. No GPU required; all reported runtimes are CPU-only.

---

## Obtaining the source data

Neither raw dataset is redistributed here. Both are freely available for
academic use.

### CICIoT2023
Download from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/iotdataset-2023.html).
Place the 169 `part-*.csv` files anywhere and set `CSV_DIR` at the top of
`01_data.py`.

### UNSW-NB15
Download from the [UNSW-NB15 project page](https://research.unsw.edu.au/projects/unsw-nb15-dataset).

> **Take the four `UNSW-NB15_1.csv` … `UNSW-NB15_4.csv` files** from the
> `UNSW-NB15 - CSV Files` folder (49 columns, ~586 MB total, 2,540,047
> records). These retain `srcip`, `dstip`, `Stime` and `Ltime`.
>
> **Do not use** the `a part of training and testing set` subfolder. Those
> files have 45 columns with no host addresses and no timestamps, and cannot
> reproduce the sequence or graph experiments of Section 5.5.

Place them in `data/unswnb15/`.

---

## Reproducing the results

### CICIoT2023

```bash
python 01_data.py                  # ~30 min, reads ~13.8 GB
python 02_preprocess.py
python 03_experiments.py           # 6 configurations x 5 seeds
python 04_prioritization.py
python 05_xai_runtime_figures.py
```

### UNSW-NB15

```bash
python -c "import def_unsw_loader as L; \
           d = L.load_unsw('data/unswnb15'); \
           s = L.stratified_subset(d, total=60900, benign_frac=0.6897, benign_name='Normal'); \
           L.report(s); s.to_parquet('data/unsw_subset.parquet')"

python 02b_preprocess_unsw.py
python 03b_experiments_unsw.py tabular    # XGBoost + RF, ~10 min
python 03b_experiments_unsw.py neural     # LSTM/Transformer/GCN/Fusion, several hours on CPU
python 04b_prioritization_unsw.py
```

`03b_experiments_unsw.py` is **resumable**. It records each (model, seed)
result to `results/unsw/runs_ledger.csv` immediately on completion and skips
anything already done, so an interrupted run can simply be restarted. To run
one model at a time:

```bash
python 03b_experiments_unsw.py neural --models Transformer --seeds 13,42,101
```

### Verifying against the manuscript

```bash
python verify_manuscript.py
```

Compares every artifact on disk against the values reported in the paper and
flags mismatches.

---

## Implementation notes

Four details are necessary for exact reproduction and are easy to get wrong.

**1. Feature ordering is load-bearing (CICIoT2023).**
After removing constant and duplicate columns, 37 of the 46 numerical
features remain. These are retained *in descending variance order*. No
top-*k* selection is applied — all 37 are used — but the order matters,
because `models.build_graph_embed()` uses column 0 as the local-activity
proxy when host identifiers are absent. Preserving the 37 features in their
original column order instead of variance order reduces accuracy from 94 %
to roughly 7 %, because each of the model's input slots then receives the
wrong feature.

**2. UNSW-NB15 must be deduplicated.**
The four source files contain **480,633 byte-identical records (18.9 %)** in
which every field, including host addresses, ports and both timestamps, is
repeated. Under any random split these appear in both the training and test
folds. `def_unsw_loader.load_unsw()` removes them by default. The effect is
strongly class-dependent: Generic falls from 215,481 to 25,378 records
(88.2 % duplicated) and DoS from 16,353 to 5,665 (65.4 %). Results computed
without deduplication will be higher than those reported here.

**3. UNSW-NB15 source files are not UTF-8.**
They are cp1252-encoded and `UNSW-NB15_1.csv` begins with a UTF-8 BOM. A
default `pd.read_csv()` raises `UnicodeDecodeError` on byte `0x92`. The
loader tries cp1252, then latin-1, then replacement, and strips BOM
characters before ASCII normalisation — note that Unicode NFKD normalisation
must *not* be applied, since it transliterates the BOM's leading character
into an ASCII letter and corrupts the first `srcip` value.

**4. Identifier hold-out (UNSW-NB15).**
`srcip`, `dstip`, `sport`, `dsport`, `Stime` and `Ltime` are excluded from
the feature matrix and supplied only to the sequence and graph constructors,
via `data/unsw/meta.parquet` with the column names `Src IP`, `Dst IP` and
`Timestamp`. Those names are what activate the host-graph and true per-host
sequence branches of `models.py`; no change to `models.py` is required.
Including them as features would leak the label, since the testbed has few
hosts and attacks are time-clustered.

---

## Changes since the first submission

- Second benchmark (UNSW-NB15) added throughout, under a matched protocol.
- `models.build_graph_embed()`: fixed the in-degree lookup, which indexed a
  destination-keyed series by source address and was therefore identically
  zero whenever the host-graph branch was taken.
- `03_experiments.py`: significance tests now computed in both directions,
  so the reported comparisons match the claims made in the paper.
- `05_xai_runtime_figures.py`: the runtime profile now measures the deployed
  path (XGBoost detection followed by TreeExplainer attribution) and reports
  detection, attribution and combined latency separately. The previous
  version timed only a model forward pass.
- `def_pipeline.py` has been **removed**. It was an early scaffold that ran
  on synthetic data and emitted placeholder figures; it produced no result
  reported in the paper.

---

## Citing

If you use this code, please cite the paper (see `CITATION.cff`) and the
source datasets:

- Neto et al. (2023), *CICIoT2023*, Sensors 23(13):5941.
- Moustafa and Slay (2015), *UNSW-NB15*, MilCIS.
- Moustafa and Slay (2016), *Information Security Journal* 25(1–3):18–31.

## License

Code released under the MIT License (see `LICENSE`). The datasets remain
subject to the terms of their respective providers.
