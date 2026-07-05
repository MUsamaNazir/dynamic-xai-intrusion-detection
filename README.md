# Dynamic Explainable Framework (DEF) — Code and Data

Code, processed data, and results for **"A Dynamic Explainable AI Framework
for Real-Time Intrusion Detection and Automated Alert Prioritization in
Security Operation Centers"** (under review, *Scientific Reports*).

## Repository layout
```
01_data.py                  build the stratified 60k CICIoT2023 subset
02_preprocess.py            cleaning, scaling, splits, SMOTE
models.py                   Transformer / context encoder / fusion / LSTM
03_experiments.py           all models x 5 seeds + significance tests
04_prioritization.py        alpha/beta/gamma tuning + theta sweep
05_xai_runtime_figures.py   SHAP/LIME metrics, runtime profile, figures
data/                       processed subset + feature list (see data/README)
results/                    per-seed metrics and derived tables (as reported)
Figures/                    figures as they appear in the manuscript
```

## Data
Experiments use a stratified 60,000-flow subset of **CICIoT2023**
(Neto et al., *Sensors* 2023, doi:10.3390/s23135941), available from the
Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/iotdataset-2023.html

The exact subset used in the paper is `data/subset60k.parquet`. To rebuild
it from the raw CSVs, set `CSV_DIR` in `01_data.py` and run step 1.

## Reproduction
```
pip install -r requirements.txt
python 01_data.py            # optional if using the included subset
python 02_preprocess.py
python 03_experiments.py     # 6 configurations x 5 seeds; several hours
python 04_prioritization.py
python 05_xai_runtime_figures.py
```
Outputs are written to `results/` and `Figures/`. The committed copies are
the runs reported in the paper (seeds 1, 7, 13, 42, 101).

## Key results (test set, n = 9,135)
- XGBoost backbone: **94.05 +/- 0.08%** accuracy, **88.49 +/- 0.13%** macro F1,
  **1.29%** FPR, 0.99 AUC (5 seeds; paired t-test vs. all other models p < 0.01)
- Transformer/context fusion: 71.82% macro F1 (reported negative result;
  the released CICIoT2023 features lack host/timestamp identifiers)
- Prioritization (theta = 0.5; alpha = 0.5, beta = 0.4, gamma = 0.1):
  **69.7%** alert reduction, 88.05% recall, 90.09% precision
- SHAP TreeExplainer 2.0 ms/alert (fidelity rho = 0.79); LIME 149.2 ms
  (top-10 Jaccard stability 0.63); end-to-end 0.19 ms/alert,
  5,221 alerts/s, 1.54 GB RAM, CPU-only

## Archive
A frozen copy of this release is archived on Zenodo:
DOI: [INSERT ZENODO DOI AFTER CREATING THE RELEASE]

## License
MIT — see LICENSE.
