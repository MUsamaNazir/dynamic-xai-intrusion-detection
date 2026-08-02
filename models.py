"""
models.py — shared model definitions (imported by 03/04/05). No need to run directly.
Transformer windows: grouped by Src IP if present, else by Dst Port, sorted by Timestamp.
GCN: manual implementation (no torch_geometric needed), 5-min windows,
nodes=IPs, edge weight=log(1+bytes) if available else flow count.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn

T_WIN, STRIDE = 32, 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- sequence windows ----------
def build_windows(X, meta, y):
    key = ("Src IP" if "Src IP" in meta.columns
           else "Protocol Type" if "Protocol Type" in meta.columns
           else "Dst Port" if "Dst Port" in meta.columns else None)
    if "Timestamp" in meta.columns:
        ts = pd.to_datetime(meta["Timestamp"], errors="coerce")
        order = np.argsort(ts.values.astype("int64"))
    elif "__order" in meta.columns:
        order = np.argsort(meta["__order"].values)   # chronological
    else:
        order = np.arange(len(X))
    Xo, yo = X[order], y[order]
    ko = meta[key].values[order] if key else np.zeros(len(X))
    wins, labs, last_idx = [], [], []
    df = pd.DataFrame({"k": ko, "i": np.arange(len(Xo))})
    for _, g in df.groupby("k"):
        ii = g["i"].values
        for s in range(0, max(1, len(ii) - 1), STRIDE):
            w = ii[s:s + T_WIN]
            pad = T_WIN - len(w)
            seq = Xo[w]
            if pad: seq = np.vstack([np.zeros((pad, Xo.shape[1])), seq])
            wins.append(seq); labs.append(yo[w[-1]]); last_idx.append(order[w[-1]])
    return (np.stack(wins).astype("float32"), np.array(labs),
            np.array(last_idx))  # last_idx maps window -> original row

# ---------- graph features (betweenness proxy + neighborhood embedding) ----------
def build_graph_embed(X, meta, hidden=16):
    """Per-flow graph/neighborhood context. With IPs+timestamps: degree stats in
    5-min windows. Without (CICIoT2023): rolling neighborhood stats within the
    protocol group - group size, local rate mean/std, position - as topology proxy."""
    out = np.zeros((len(X), 4), dtype="float32")
    if "Src IP" not in meta.columns or "Timestamp" not in meta.columns:
        key = meta["Protocol Type"].values if "Protocol Type" in meta.columns else np.zeros(len(X))
        base = X[:, 0]  # first (highest-variance) feature as activity proxy
        d = pd.DataFrame({"k": key, "v": base, "i": np.arange(len(X))})
        for _, g in d.groupby("k"):
            r = g["v"].rolling(64, min_periods=1)
            out[g["i"].values, 0] = np.log1p(len(g))
            out[g["i"].values, 1] = r.mean().values
            out[g["i"].values, 2] = r.std().fillna(0).values
            out[g["i"].values, 3] = np.arange(len(g)) / max(1, len(g))
        return out
    ts = pd.to_datetime(meta["Timestamp"], errors="coerce")
    slot = (ts.values.astype("int64") // (5 * 60 * 10**9))
    d = pd.DataFrame({"s": meta["Src IP"].values, "d": meta.get("Dst IP", meta["Src IP"]).values,
                      "slot": slot, "i": np.arange(len(X))})
    for _, g in d.groupby("slot"):
        deg_out = g.groupby("s").size(); deg_in = g.groupby("d").size()
        fanout = g.groupby("s")["d"].nunique()
        for _, r in g.iterrows():
            out[r["i"]] = [deg_out.get(r["s"], 0), deg_in.get(r["d"], 0),
                           fanout.get(r["s"], 0), len(g)]
    return np.log1p(out)

# ---------- torch models ----------
class SeqTransformer(nn.Module):
    def __init__(self, d_in, n_cls, d=128, heads=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        pe = torch.zeros(T_WIN, d)
        pos = torch.arange(T_WIN).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-np.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        enc = nn.TransformerEncoderLayer(d, heads, 256, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, n_cls)
    def embed(self, x):
        h = self.enc(self.proj(x) + self.pe)
        return h.mean(1)
    def forward(self, x): return self.head(self.embed(x))

class GCNHead(nn.Module):
    """2-layer MLP over per-flow graph-context features (degree/fan-out stats)."""
    def __init__(self, n_cls, d_g=4, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_g, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.head = nn.Linear(h, n_cls)
    def embed(self, g): return self.f(g)
    def forward(self, g): return self.head(self.embed(g))

class DEF(nn.Module):
    def __init__(self, d_in, n_cls, d_g=4):
        super().__init__()
        self.tr = SeqTransformer(d_in, n_cls)
        self.gc = GCNHead(n_cls, d_g)
        self.head = nn.Sequential(nn.Linear(128 + 64, 128), nn.ReLU(), nn.Linear(128, n_cls))
    def forward(self, x, g):
        return self.head(torch.cat([self.tr.embed(x), self.gc.embed(g)], dim=1))

class LSTMNet(nn.Module):
    def __init__(self, d_in, n_cls):
        super().__init__()
        self.l = nn.LSTM(d_in, 128, 2, batch_first=True)
        self.head = nn.Linear(128, n_cls)
    def forward(self, x): return self.head(self.l(x)[0][:, -1])

def train_torch(model, loaders, epochs=50, lr=1e-3, patience=10, seed=42, class_weight=None):
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(weight=None if class_weight is None
                 else torch.tensor(class_weight, dtype=torch.float32).to(DEVICE))
    best, wait, best_state = -1, 0, None
    for ep in range(epochs):
        model.train()
        for batch in loaders["train"]:
            batch = [b.to(DEVICE) for b in batch]
            opt.zero_grad()
            out = model(*batch[:-1])
            loss = lossf(out, batch[-1]); loss.backward(); opt.step()
        # val macro-F1
        from sklearn.metrics import f1_score
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for batch in loaders["val"]:
                batch = [b.to(DEVICE) for b in batch]
                P.append(model(*batch[:-1]).argmax(1).cpu()); Y.append(batch[-1].cpu())
        f1 = f1_score(torch.cat(Y), torch.cat(P), average="macro")
        if f1 > best: best, wait, best_state = f1, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    return model
