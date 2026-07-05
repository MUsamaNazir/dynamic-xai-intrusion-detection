"""
STEP 5 — XAI timing + quality metrics, runtime profile, and all figures.
Run:  python 05_xai_runtime_figures.py
Outputs:
  results/xai_metrics.txt   -> SHAP/LIME times, fidelity rho, LIME Jaccard stability
  results/runtime.txt       -> latency mean/p95, throughput, memory (Section 5.9)
  Figures/*.png             -> beeswarm, bar, LIME example, confusion matrices
"""
import time, numpy as np, pandas as pd, torch, shap, psutil, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import models as M

os.makedirs("Figures", exist_ok=True)
te = np.load("data/test.npz"); tr = np.load("data/train_noSMOTE.npz")
feat = open("data/feature_names.txt").read().splitlines()
classes = np.load("data/classes.npy", allow_pickle=True)

xgb = XGBClassifier(); xgb.load_model("results/xgb_seed42.json")
out = open("results/xai_metrics.txt", "w")

# --- SHAP timing (TreeExplainer, backbone) ---
ex = shap.TreeExplainer(xgb)
sub = te["X"][:500]
t0 = time.perf_counter(); sv = ex.shap_values(sub); t1 = time.perf_counter()
raw = np.array(sv)
if raw.ndim == 3 and raw.shape[0] == len(sub):      # new shap: (n, feat, class)
    sv_pc = [raw[:, :, c] for c in range(raw.shape[2])]
else:                                                # old shap: (class, n, feat)
    sv_pc = [raw[c] for c in range(raw.shape[0])]
out.write(f"SHAP TreeExplainer: {(t1-t0)/len(sub)*1000:.1f} ms/sample\n")

# --- SHAP timing for full DEF (GradientExplainer) ---
deff = M.DEF(len(feat), len(classes)); deff.load_state_dict(torch.load("results/def_seed42.pt", map_location=M.DEVICE)); deff.to(M.DEVICE).eval()
# explain the fusion model wrt the last flow in each window (simplest faithful setup)
meta = pd.read_parquet("data/meta.parquet")
full = np.zeros((len(meta), len(feat)), dtype="float32")
for p in (tr, np.load("data/val.npz"), te): full[p["idx"]] = p["X"]
W, Wy, Widx = M.build_windows(full, meta, np.zeros(len(meta), dtype=int))
G = M.build_graph_embed(full, meta)
bw = torch.tensor(W[:64]).to(M.DEVICE); bg = torch.tensor(G[Widx[:64]], dtype=torch.float32).to(M.DEVICE)
gex = shap.GradientExplainer(deff, [bw, bg])
t0 = time.perf_counter(); _ = gex.shap_values([bw[:32], bg[:32]]); t1 = time.perf_counter()
out.write(f"SHAP GradientExplainer (full DEF): {(t1-t0)/32*1000:.1f} ms/sample\n")

# --- fidelity: mean|SHAP| vs permutation importance ---
pi = permutation_importance(xgb, te["X"][:2000], te["y"][:2000], n_repeats=5, random_state=42)
mshap = np.abs(np.stack(sv_pc)).mean((0, 1))   # (feat,)
rho = spearmanr(mshap, pi.importances_mean).statistic
out.write(f"Fidelity (Spearman SHAP vs permutation importance): {rho:.2f}\n")

# --- LIME timing + stability ---
expl = LimeTabularExplainer(tr["X"], feature_names=feat, class_names=list(classes),
                            discretize_continuous=True)
times, jac = [], []
for i in range(30):
    x = te["X"][i]
    tops = []
    for rep in range(5):
        t0 = time.perf_counter()
        e = expl.explain_instance(x, xgb.predict_proba, num_features=10)
        times.append(time.perf_counter() - t0)
        tops.append({f for f, _ in e.as_list()})
    for a in range(5):
        for b in range(a + 1, 5):
            jac.append(len(tops[a] & tops[b]) / len(tops[a] | tops[b]))
out.write(f"LIME: {np.mean(times)*1000:.1f} ms/instance, "
          f"top-10 Jaccard stability {np.mean(jac):.2f}\n")
out.close(); print(open("results/xai_metrics.txt").read())

# --- runtime profile (Section 5.9) ---
lat = []
proc = psutil.Process()
with torch.no_grad():
    for i in range(0, 3200, 32):
        b = torch.tensor(W[i:i+32]).to(M.DEVICE)
        g = torch.tensor(G[Widx[i:i+32]], dtype=torch.float32).to(M.DEVICE)
        t0 = time.perf_counter(); deff(b, g); 
        if M.DEVICE == "cuda": torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) / 32 * 1000)
with open("results/runtime.txt", "w") as f:
    f.write(f"latency mean {np.mean(lat):.2f} ms, p95 {np.percentile(lat,95):.2f} ms/alert\n")
    f.write(f"throughput {1000/np.mean(lat)*1:.0f} alerts/s ({M.DEVICE})\n")
    f.write(f"RAM {proc.memory_info().rss/1e9:.2f} GB")
    if M.DEVICE == "cuda":
        f.write(f", GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print(open("results/runtime.txt").read())

# --- figures with REAL feature names ---
att = int(np.argmax(classes != "Benign"))   # first attack class
shap.summary_plot(sv_pc[att], sub, feature_names=feat, show=False, max_display=15)
plt.tight_layout(); plt.savefig("Figures/shap_beeswarm_CICIoT2023.png", dpi=200); plt.close()
shap.summary_plot(sv_pc, sub, feature_names=feat, plot_type="bar", show=False, max_display=15, class_names=list(classes))
plt.tight_layout(); plt.savefig("Figures/shap_bar_CICIoT2023.png", dpi=200); plt.close()
e = expl.explain_instance(te["X"][0], xgb.predict_proba, num_features=10)
e.as_pyplot_figure(); plt.tight_layout()
plt.savefig("Figures/lime_local_CICIoT2023.png", dpi=200); plt.close()
pd.DataFrame(e.as_list(), columns=["condition", "weight"]).to_csv("results/lime_table7.csv", index=False)

for name, pred in [("DEF", np.load("results/probs_DEF.npy").argmax(1)),
                   ("XGBoost", xgb.predict(te["X"]))]:
    yy = Wy if name == "DEF" else te["y"]
    yy = np.load("data/test.npz")["y"] if name == "XGBoost" else None
    # DEF preds are over test windows; align:
    if name == "DEF":
        widx = np.load("results/def_test_widx.npy")
        full_y = np.zeros(len(meta), dtype=int)
        for p in (tr, np.load("data/val.npz"), te): full_y[p["idx"]] = p["y"]
        yy = full_y[widx]; pred = pred[:len(yy)]
    cm = confusion_matrix(yy, pred, normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        xticks_rotation=45, values_format=".2f", colorbar=False)
    plt.tight_layout(); plt.savefig(f"Figures/confmat_{name}_CICIoT2023.png", dpi=200); plt.close()
print("figures written to Figures/")
