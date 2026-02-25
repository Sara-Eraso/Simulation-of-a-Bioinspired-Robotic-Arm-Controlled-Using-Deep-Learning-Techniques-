# -- coding: utf-8 --
# ============================================================
# NARX EMG - SOLO DOBLE ENTRADA (CH1 + CH2)
# + Features lentas por ventanas (basic/spectral)
# + Autoregresivo estable: cont/class/mix + EMA + dy_max + clip
#
# - Emparejamiento por número o por orden
# - Lectura robusta CSV/TXT (coma/; tab), decimales con coma
# - Normalización por archivo opcional (--per-file-norm)
# - Scaler ajustado SOLO con TRAIN
# - Split por par de archivos (GroupShuffleSplit)
# - Ventanas: --win-ms / --hop-ms
# - Features:
#     basic:   RMS, MAV, WL, ZC, SSC
#     spectral: + bandpower 20-60, 60-120, median frequency
# - Autoregresivo opcional: --eval-autoreg
# - Reporte por par en test: figs/test_metrics_by_pair.txt
# ============================================================

import argparse, importlib, re
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, confusion_matrix
from sklearn.neural_network import MLPRegressor
import joblib
import matplotlib.pyplot as plt


# ---------------- Defaults ----------------
ANGLES = [30.0, 60.0, 90.0, 120.0, 150.0]

TEST_SIZE = 0.2
VAL_SPLIT = 0.1

SEED = 2025
np.random.seed(SEED)

# MLP defaults
HIDDEN_SIZES = (128, 64, 32)
ALPHA = 1e-3
MAX_ITER = 600

# AR defaults
N_U = 12
N_Y = 1


# -------------- util --------------
def _key_by_number(p: Path):
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else p.stem

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def _robust_z_per_file(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    z = (x - med) / (1.4826 * mad)
    return np.clip(z, -10.0, 10.0)

def read_any(path: Path) -> pd.DataFrame:
    """Lee TXT/CSV (auto-separador). Devuelve columnas: t, amp"""
    tried = []
    df = None
    for sep in [None, ",", ";", "\t", r"\s+"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", header=None)
            if df.shape[1] >= 2:
                break
        except Exception as e:
            tried.append(str(e))
            df = None

    if df is None or df.shape[1] < 2:
        raise ValueError(f"No pude leer 2 columnas en {path.name}. Intentos={len(tried)}")

    df = df.iloc[:, :2].copy()
    df.columns = ["t", "amp"]

    # t puede venir como string; intenta parsear
    if df["t"].dtype == object:
        df["t"] = pd.to_numeric(df["t"].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    if df["amp"].dtype == object:
        df["amp"] = pd.to_numeric(df["amp"].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    df = df.dropna()
    return df

def assign_blocks(n: int, angles: List[float]) -> np.ndarray:
    """Etiqueta por bloques iguales (asume protocolo por bloques)."""
    m = len(angles)
    q = n // m
    y = np.empty(n, float)
    for i, ang in enumerate(angles):
        beg = i * q
        end = n if i == m - 1 else (i + 1) * q
        y[beg:end] = ang
    return y

def _angles_to_classes(y: np.ndarray, angles: List[float]) -> np.ndarray:
    A = np.asarray(angles, float).reshape(1, -1)
    y = np.asarray(y, float).reshape(-1, 1)
    idx = np.argmin(np.abs(y - A), axis=1)
    return np.asarray(angles, float)[idx]

def _acc_tolerancia(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 5.0) -> float:
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    return float(np.mean(np.abs(y_true - y_pred) <= tol))

def scaler_factory(name: str):
    if name == "robust": return RobustScaler()
    if name == "minmax": return MinMaxScaler()
    return StandardScaler()

def _figdir(base: Path) -> Path:
    d = base / "figs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------- Window features ----------------
def _infer_fs(t: np.ndarray) -> float:
    t = np.asarray(t, float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 1.0
    return float(1.0 / np.median(dt))

def _bandpower(mag2: np.ndarray, freqs: np.ndarray, f1: float, f2: float) -> float:
    idx = (freqs >= f1) & (freqs <= f2)
    if not np.any(idx):
        return 0.0
    return float(np.trapz(mag2[idx], freqs[idx]))

def _median_freq(mag2: np.ndarray, freqs: np.ndarray) -> float:
    total = float(np.trapz(mag2, freqs))
    if total <= 0:
        return 0.0
    cdf = np.cumsum(mag2)
    cdf = cdf / (cdf[-1] + 1e-12)
    k = int(np.searchsorted(cdf, 0.5))
    k = min(max(k, 0), len(freqs) - 1)
    return float(freqs[k])

def window_features(x: np.ndarray, t: np.ndarray,
                    win_ms: float, hop_ms: float,
                    feat_set: str = "basic",
                    zc_th: float = 0.02,
                    ssc_th: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve:
      F: (W, D) features por ventana
      tc: (W,) tiempo centro de ventana
    """
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    fs = _infer_fs(t)

    win = max(8, int(round(win_ms * 1e-3 * fs)))
    hop = max(1, int(round(hop_ms * 1e-3 * fs)))

    W = 1 + max(0, (len(x) - win) // hop)
    if W <= 0:
        return np.empty((0, 1)), np.empty((0,))

    feats = []
    t_centers = []

    for w in range(W):
        i0 = w * hop
        i1 = i0 + win
        seg = x[i0:i1]
        if len(seg) < win:
            break

        tc = float(t[i0 + win // 2])
        t_centers.append(tc)

        # ---- basic time-domain ----
        rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
        mav = float(np.mean(np.abs(seg)))
        wl = float(np.sum(np.abs(np.diff(seg))))

        # Zero-crossings con umbral
        s = seg.copy()
        s[np.abs(s) < zc_th] = 0.0
        zc = float(np.sum((s[:-1] * s[1:]) < 0))

        # Slope sign changes con umbral
        d1 = np.diff(seg)
        ssc = float(np.sum(((d1[:-1] * d1[1:]) < 0) & (np.abs(d1[:-1] - d1[1:]) > ssc_th)))

        row = [rms, mav, wl, zc, ssc]

        if feat_set == "spectral":
            seg0 = seg - np.mean(seg)
            spec = np.fft.rfft(seg0)
            mag2 = (np.abs(spec) ** 2) / (len(seg0) + 1e-12)
            freqs = np.fft.rfftfreq(len(seg0), d=1.0 / fs)

            bp1 = _bandpower(mag2, freqs, 20.0, 60.0)
            bp2 = _bandpower(mag2, freqs, 60.0, 120.0)
            mf = _median_freq(mag2, freqs)
            row += [bp1, bp2, mf]

        feats.append(row)

    F = np.asarray(feats, float)
    tc = np.asarray(t_centers, float)
    return F, tc


# ---------------- NARX feature builder ----------------
def make_narx_features_dual(U1: np.ndarray, U2: np.ndarray, y: np.ndarray, n_u: int, n_y: int):
    """
    U1: (T, d1), U2: (T, d2), y: (T,)
    X(t) = [U1(t-1..t-n_u), U2(t-1..t-n_u), y(t-1..t-n_y)]
    """
    U1 = np.asarray(U1, float)
    U2 = np.asarray(U2, float)
    y = np.asarray(y, float)

    T = len(y)
    start = max(n_u, n_y) + 1
    d1 = U1.shape[1]
    d2 = U2.shape[1]
    if T <= start:
        return np.empty((0, (d1 + d2) * n_u + n_y)), np.empty((0,))

    Xs, Ys = [], []
    for t in range(start, T):
        feats = []
        for k in range(1, n_u + 1):
            feats.extend(U1[t - k].tolist())
        for k in range(1, n_u + 1):
            feats.extend(U2[t - k].tolist())
        feats.extend([y[t - k] for k in range(1, n_y + 1)])
        Xs.append(feats)
        Ys.append(y[t])

    return np.asarray(Xs, float), np.asarray(Ys, float)


# ---------------- Dataset builder (dual only) ----------------
def collect_dual_windows(
    dir1: Path, dir2: Path, pat1: str, pat2: str,
    angles: List[float],
    per_file_norm: bool,
    n_u: int, n_y: int,
    pair_mode: str,
    win_ms: float, hop_ms: float,
    feat_set: str,
    zc_th: float, ssc_th: float,
    y_noise: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Returns: X, y, groups, n_pairs_used, feat_dim_per_channel
    """
    if pair_mode == "number":
        f1 = sorted(dir1.glob(pat1), key=_key_by_number)
        f2 = sorted(dir2.glob(pat2), key=_key_by_number)
    else:
        f1 = sorted(dir1.glob(pat1))
        f2 = sorted(dir2.glob(pat2))

    n1, n2 = len(f1), len(f2)
    n = min(n1, n2)
    if n == 0:
        raise FileNotFoundError("Sin pares en dual.")
    if n1 != n2:
        print(f"[WARN] ch1={n1} ch2={n2} -> uso n={n}")

    print(f"[DUAL] ch1={n1}  ch2={n2}  -> pares usados={n}")
    print("[DUAL] primeros 5 pares:", [(f1[i].name, f2[i].name) for i in range(min(5, n))])

    X_list, y_list, g_list = [], [], []
    used = 0
    feat_dim = None

    for i in range(n):
        try:
            df1 = read_any(f1[i])
            df2 = read_any(f2[i])
        except Exception as e:
            print(f"[SKIP] {f1[i].name} | {f2[i].name}: {e}")
            continue

        a1 = df1["amp"].values
        a2 = df2["amp"].values
        t1 = df1["t"].values
        t2 = df2["t"].values

        L = min(len(a1), len(a2), len(t1), len(t2))
        if L < 10:
            continue
        a1, a2 = a1[:L], a2[:L]
        t1, t2 = t1[:L], t2[:L]

        if per_file_norm:
            a1 = _robust_z_per_file(a1)
            a2 = _robust_z_per_file(a2)

        # Features por ventana
        F1, tc1 = window_features(a1, t1, win_ms=win_ms, hop_ms=hop_ms, feat_set=feat_set, zc_th=zc_th, ssc_th=ssc_th)
        F2, tc2 = window_features(a2, t2, win_ms=win_ms, hop_ms=hop_ms, feat_set=feat_set, zc_th=zc_th, ssc_th=ssc_th)

        W = min(len(F1), len(F2))
        if W <= max(n_u, n_y) + 2:
            continue
        F1, F2 = F1[:W], F2[:W]

        # y por bloques en espacio de ventanas
        y_win = assign_blocks(W, angles)

        # "teacher-forcing robustness": ruido sobre lags de y (solo en construcción de features)
        if y_noise and y_noise > 0:
            y_for_features = y_win + np.random.normal(0.0, y_noise, size=len(y_win))
        else:
            y_for_features = y_win

        if feat_dim is None:
            feat_dim = int(F1.shape[1])

        Xf, yf = make_narx_features_dual(F1, F2, y_for_features, n_u=n_u, n_y=n_y)
        if Xf.size == 0:
            continue

        # Nota: target sigue siendo y real (no ruidosa)
        # alineación: make_narx_features_dual devuelve yf = y_for_features[t]
        # queremos la etiqueta real: reconstruimos con y_win:
        _, y_real = make_narx_features_dual(F1, F2, y_win, n_u=n_u, n_y=n_y)

        gid = f"{f1[i].name}__{f2[i].name}"
        X_list.append(Xf)
        y_list.append(y_real)
        g_list.append(np.array([gid] * len(y_real), dtype=object))
        used += 1

    if not X_list:
        raise RuntimeError("Sin pares válidos (dual).")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(g_list)

    if feat_dim is None:
        feat_dim = 1

    return X, y, groups, used, feat_dim


def subsample_by_group(X: np.ndarray, y: np.ndarray, groups: np.ndarray, max_samples: int, random_state: int = 2025):
    rng = np.random.default_rng(random_state)
    if len(X) <= max_samples:
        return X, y, groups
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx], y[idx], groups[idx]


# ---------------- Models ----------------
def build_model_sklearn(hidden, alpha, max_iter) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=tuple(hidden),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=float(alpha),
        tol=1e-4,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        max_iter=int(max_iter),
        shuffle=True,
        random_state=SEED,
        verbose=False
    )

def build_model_tf(input_dim: int, dropout: float = 0.2):
    tf = importlib.import_module("tensorflow")
    keras = importlib.import_module("tensorflow.keras")
    layers = importlib.import_module("tensorflow.keras.layers")
    reg = importlib.import_module("tensorflow.keras.regularizers")

    tf.random.set_seed(SEED)
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu", kernel_regularizer=reg.l2(1e-4)),
        layers.Dropout(dropout),
        layers.Dense(128, activation="relu", kernel_regularizer=reg.l2(1e-4)),
        layers.Dropout(dropout),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


# ---------------- Plots & Reports ----------------
def plot_loss_sklearn(model, outdir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(getattr(model, "loss_curve_", []), label="train loss")
    if hasattr(model, "validation_scores_") and getattr(model, "validation_scores_", None) is not None:
        ax2 = ax.twinx()
        ax2.plot(model.validation_scores_, alpha=0.7, label="val score")
        ax2.set_ylabel("val score")
        ax2.legend(loc="lower right")
    ax.set_xlabel("iteración")
    ax.set_ylabel("loss")
    ax.set_title("Curva de entrenamiento (sklearn)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "loss_sklearn.png", dpi=150)
    plt.close(fig)

def plot_history_tf(history, outdir: Path):
    if history is None:
        return
    hist = history.history
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist.get("loss", []), label="loss")
    if "val_loss" in hist:
        ax.plot(hist["val_loss"], label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss (TensorFlow)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_tf.png", dpi=150)
    plt.close(fig)

def report_test_by_group(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, outdir: Path, tol: float = 5.0):
    figdir = _figdir(outdir)
    df = pd.DataFrame({
        "group": np.asarray(groups, object),
        "y_true": np.asarray(y_true, float),
        "y_pred": np.asarray(y_pred, float),
    })
    df["abs_err"] = np.abs(df["y_true"] - df["y_pred"])
    df["ok"] = (df["abs_err"] <= tol).astype(int)

    mae_all = df["abs_err"].mean()
    acc_all = df["ok"].mean()

    lines = [
        f"[GLOBAL]\tN={len(df)}\tMAE={mae_all:.3f}°\tACC±{tol:.0f}°={acc_all:.3f}",
        "-" * 80
    ]

    agg = df.groupby("group").agg(
        N=("abs_err", "size"),
        MAE=("abs_err", "mean"),
        ACC=("ok", "mean"),
    ).reset_index()

    for _, r in agg.iterrows():
        lines.append(f"{r['group']}\tN={int(r['N'])}\tMAE={r['MAE']:.3f}°\tACC±{tol:.0f}°={r['ACC']:.3f}")

    out = figdir / "test_metrics_by_pair.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[TEST] Métricas por par guardadas en: {out}")


# ---------------- Save/Load ----------------
def save_artifacts(save_dir: Path, backend: str, scaler_X, model, meta: str):
    ensure_dir(save_dir)

    # limpiar artefacto previo correcto
    try:
        old = save_dir / ("model.keras" if backend == "tf" else "model.pkl")
        if old.exists():
            old.unlink()
    except Exception as e:
        print(f"[WARN] limpieza artefacto previo: {e}")

    joblib.dump(scaler_X, save_dir / "scaler.pkl")
    (save_dir / "backend.txt").write_text(backend, encoding="utf-8")

    if backend == "tf":
        keras_models = importlib.import_module("tensorflow.keras.models")
        keras_models.save_model(model, save_dir / "model.keras")
    else:
        joblib.dump(model, save_dir / "model.pkl")

    (save_dir / "meta.txt").write_text(meta, encoding="utf-8")
    print(f"[SAVE] Artefactos en: {save_dir}")

def load_artifacts(load_dir: str):
    d = Path(load_dir)
    scaler_X = joblib.load(d / "scaler.pkl")
    backend = (d / "backend.txt").read_text(encoding="utf-8").strip() if (d / "backend.txt").exists() else "sklearn"

    if (d / "model.keras").exists():
        keras_models = importlib.import_module("tensorflow.keras.models")
        model = keras_models.load_model(d / "model.keras")
        backend = "tf"
    else:
        model = joblib.load(d / "model.pkl")
        backend = "sklearn"

    print(f"[LOAD] desde {d} (backend={backend})")
    return scaler_X, model, backend, d


# ---------------- Autoregresivo ----------------
def _ema(x, k):
    if k <= 1:
        return x
    y = np.zeros_like(x, float)
    a = 2.0 / (k + 1.0)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1 - a) * y[i - 1]
    return y

def _clip(y, y_clip: Optional[Tuple[float, float]]):
    if y_clip is None:
        return y
    lo, hi = y_clip
    return float(np.clip(y, lo, hi))

def _apply_dy_limit(y_new: float, y_prev: float, dy_max: float) -> float:
    if dy_max <= 0:
        return y_new
    dy = y_new - y_prev
    if dy > dy_max:
        return y_prev + dy_max
    if dy < -dy_max:
        return y_prev - dy_max
    return y_new

def _feed_value(y_hat: float, angles: List[float], ar_feed: str, mix_beta: float, y_prev: float,
                dy_max: float, y_clip: Optional[Tuple[float, float]]) -> float:
    """
    ar_feed:
      cont: usa y_hat
      class: cuantiza a clase
      mix: y_feed = beta*y_hat + (1-beta)*y_class
    además aplica dy_max y clip
    """
    angles_arr = np.asarray(angles, float)
    y_class = float(angles_arr[np.argmin(np.abs(angles_arr - y_hat))])

    if ar_feed == "class":
        y_feed = y_class
    elif ar_feed == "mix":
        b = float(np.clip(mix_beta, 0.0, 1.0))
        y_feed = b * float(y_hat) + (1.0 - b) * y_class
    else:
        y_feed = float(y_hat)

    y_feed = _apply_dy_limit(y_feed, y_prev=y_prev, dy_max=dy_max)
    y_feed = _clip(y_feed, y_clip)
    return y_feed


def eval_autoreg_dual_windows(
    dir1: Path, pat1: str, dir2: Path, pat2: str,
    angles: List[float], outdir: Path,
    scaler_X, model, backend: str,
    pair_mode: str,
    per_file_norm: bool,
    win_ms: float, hop_ms: float,
    feat_set: str,
    zc_th: float, ssc_th: float,
    ar_feed: str, mix_beta: float,
    ema_k: int,
    seed_angle: Optional[float],
    dy_max: float,
    y_clip: Optional[Tuple[float, float]],
):
    # listar pares
    if pair_mode == "number":
        f1 = sorted(dir1.glob(pat1), key=_key_by_number)
        f2 = sorted(dir2.glob(pat2), key=_key_by_number)
    else:
        f1 = sorted(dir1.glob(pat1))
        f2 = sorted(dir2.glob(pat2))
    n = min(len(f1), len(f2))
    if n == 0:
        print("[EVAL-AR] Sin pares para evaluar.")
        return

    figdir = _figdir(outdir)
    all_true, all_pred = [], []

    for i in range(n):
        try:
            df1 = read_any(f1[i])
            df2 = read_any(f2[i])
        except Exception as e:
            print(f"[SKIP] {f1[i].name} | {f2[i].name}: {e}")
            continue

        a1 = df1["amp"].values
        a2 = df2["amp"].values
        t1 = df1["t"].values
        t2 = df2["t"].values

        L = min(len(a1), len(a2), len(t1), len(t2))
        if L < 10:
            continue
        a1, a2 = a1[:L], a2[:L]
        t1, t2 = t1[:L], t2[:L]

        if per_file_norm:
            a1 = _robust_z_per_file(a1)
            a2 = _robust_z_per_file(a2)

        F1, _ = window_features(a1, t1, win_ms=win_ms, hop_ms=hop_ms, feat_set=feat_set, zc_th=zc_th, ssc_th=ssc_th)
        F2, _ = window_features(a2, t2, win_ms=win_ms, hop_ms=hop_ms, feat_set=feat_set, zc_th=zc_th, ssc_th=ssc_th)

        W = min(len(F1), len(F2))
        if W <= max(N_U, N_Y) + 2:
            continue
        F1, F2 = F1[:W], F2[:W]

        y_true_full = assign_blocks(W, angles)

        # autoreg
        seed = float(seed_angle) if seed_angle is not None else float(angles[0])
        ybuf = [seed] * N_Y
        preds = []

        # start index consistent con make_narx_features_dual
        start = max(N_U, N_Y) + 1
        for t in range(start, W):
            feats = []
            for k in range(1, N_U + 1):
                feats.extend(F1[t - k].tolist())
            for k in range(1, N_U + 1):
                feats.extend(F2[t - k].tolist())
            feats.extend(ybuf)

            X_t = np.asarray(feats, float).reshape(1, -1)
            X_t = scaler_X.transform(X_t)

            if backend == "tf":
                y_hat = float(model.predict(X_t, verbose=0).ravel()[0])
            else:
                y_hat = float(model.predict(X_t)[0])

            preds.append(y_hat)

            # feedback estable
            y_prev = ybuf[0] if len(ybuf) else seed
            y_feed = _feed_value(
                y_hat=y_hat, angles=angles, ar_feed=ar_feed, mix_beta=mix_beta,
                y_prev=y_prev, dy_max=dy_max, y_clip=y_clip
            )
            ybuf = [y_feed] + ybuf[:-1]

        y_pred = np.asarray(preds, float)
        if ema_k and ema_k > 1:
            y_pred = _ema(y_pred, ema_k)

        # alinear con y_true
        y_true = y_true_full[start:start + len(y_pred)]
        if len(y_true) != len(y_pred) or len(y_true) == 0:
            continue

        all_true.append(y_true)
        all_pred.append(y_pred)

    if not all_true:
        print("[EVAL-AR] No hubo series válidas.")
        return

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    mae = mean_absolute_error(y_true, y_pred)
    acc5 = _acc_tolerancia(y_true, y_pred, tol=5.0)

    (figdir / "metrics_autoreg.txt").write_text(
        f"MAE={mae:.3f}°\nACC±5°={acc5:.3f}\nN={len(y_true)}\n",
        encoding="utf-8"
    )

    # Confusion por clases (cuantiza)
    y_true_cls = _angles_to_classes(y_true, angles)
    y_pred_cls = _angles_to_classes(y_pred, angles)
    labels = np.asarray(angles, float)
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Matriz de confusión (autoregresivo)")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels.astype(int))
    ax.set_yticklabels(labels.astype(int))
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, cm[r, c], ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figdir / "confusion_matrix_autoreg.png", dpi=150)
    plt.close(fig)

    # Accuracy por clase
    accs = []
    for a in labels:
        idx = (y_true_cls == a)
        accs.append(float(np.mean((y_pred_cls[idx] == a))) if np.any(idx) else np.nan)

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(labels, accs, width=6)
    for x, v in zip(labels, accs):
        if not np.isnan(v):
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Clase (°)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy por clase (autoregresivo)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "acc_per_class_autoreg.png", dpi=150)
    plt.close(fig)

    # Accuracy vs tolerancia
    tols = [1, 2, 3, 5, 7, 10, 12, 15]
    vals = [_acc_tolerancia(y_true, y_pred, t) for t in tols]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(tols, vals, marker="o")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Tolerancia (°)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs tolerancia (autoregresivo)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "acc_vs_tol_autoreg.png", dpi=150)
    plt.close(fig)

    print(f"[EVAL-AR] MAE={mae:.3f}°  ACC±5°={acc5:.3f}  (N={len(y_true)})")
    print(f"[EVAL-AR] Guardadas figuras en {figdir}")


# ---------------- Main ----------------
def main():
    global ANGLES, N_U, N_Y

    p = argparse.ArgumentParser(description="NARX EMG dual-only con features por ventanas + AR estable")

    # dual I/O
    p.add_argument("--data-ch1", type=str, required=True)
    p.add_argument("--data-ch2", type=str, required=True)
    p.add_argument("--pattern-ch1", type=str, default="*.csv")
    p.add_argument("--pattern-ch2", type=str, default="*.csv")
    p.add_argument("--pair-mode", choices=["zip", "number"], default="number")

    # angles
    p.add_argument("--angles", type=str, default="", help="Ej: 30,60,90,120,150")

    # model backend
    p.add_argument("--model", choices=["sklearn", "tf"], default="sklearn")
    p.add_argument("--hidden", type=str, default="", help="Ej: 128,64,32")
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--max-iter", type=int, default=MAX_ITER)
    p.add_argument("--dropout", type=float, default=0.2)

    # NARX lags
    p.add_argument("--n-u", type=int, default=12)
    p.add_argument("--n-y", type=int, default=1)

    # preprocessing
    p.add_argument("--per-file-norm", action="store_true")
    p.add_argument("--scaler", choices=["standard", "robust", "minmax"], default="robust")

    # window features
    p.add_argument("--win-ms", type=float, default=250.0)
    p.add_argument("--hop-ms", type=float, default=50.0)
    p.add_argument("--feat-set", choices=["basic", "spectral"], default="basic")
    p.add_argument("--zc-th", type=float, default=0.02)
    p.add_argument("--ssc-th", type=float, default=0.02)

    # robustness train-side (teacher-forcing noise on y_lags)
    p.add_argument("--y-noise", type=float, default=0.0)

    # speed
    p.add_argument("--max-samples", type=int, default=0)

    # save/load
    p.add_argument("--save-dir", type=str, default="")
    p.add_argument("--load-dir", type=str, default="")

    # autoregresivo
    p.add_argument("--eval-autoreg", action="store_true")
    p.add_argument("--ar-feed", choices=["cont", "class", "mix"], default="mix")
    p.add_argument("--mix-beta", type=float, default=0.85)
    p.add_argument("--ema", type=int, default=5)
    p.add_argument("--seed-angle", type=float, default=None)

    # AR stabilizers
    p.add_argument("--dy-max", type=float, default=8.0)
    p.add_argument("--y-clip", type=str, default="", help="Ej: '20,160'")

    args = p.parse_args()

    # angles override
    if args.angles.strip():
        ANGLES = [float(a) for a in args.angles.split(",") if a.strip()]

    # lags
    N_U = int(args.n_u)
    N_Y = int(args.n_y)

    # parse hidden
    hidden = HIDDEN_SIZES
    if args.hidden.strip():
        hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    # parse y_clip
    y_clip = None
    if args.y_clip.strip():
        parts = [x.strip() for x in args.y_clip.split(",")]
        if len(parts) != 2:
            raise ValueError("--y-clip debe ser 'lo,hi' (ej: --y-clip 20,160)")
        y_clip = (float(parts[0]), float(parts[1]))

    # output dir
    target_dir = Path(args.save_dir) if args.save_dir else (Path.home() / "OneDrive" / "Escritorio" / "narx_model_dual_windows")
    ensure_dir(target_dir)
    figs = _figdir(target_dir)

    # load or train
    if args.load_dir.strip():
        scaler_X, model, backend, target_dir = load_artifacts(args.load_dir)
        figs = _figdir(target_dir)
        if backend == "sklearn" and hasattr(model, "loss_curve_"):
            plot_loss_sklearn(model, figs)
    else:
        X, y, groups, npairs, feat_dim = collect_dual_windows(
            Path(args.data_ch1), Path(args.data_ch2),
            args.pattern_ch1, args.pattern_ch2,
            ANGLES,
            per_file_norm=bool(args.per_file_norm),
            n_u=N_U, n_y=N_Y,
            pair_mode=args.pair_mode,
            win_ms=float(args.win_ms),
            hop_ms=float(args.hop_ms),
            feat_set=str(args.feat_set),
            zc_th=float(args.zc_th),
            ssc_th=float(args.ssc_th),
            y_noise=float(args.y_noise),
        )

        if args.max_samples and args.max_samples > 0:
            X, y, groups = subsample_by_group(X, y, groups, max_samples=int(args.max_samples), random_state=SEED)

        print(f"Pares usados: {npairs} | X={X.shape} y={y.shape} | input_dim={X.shape[1]} "
              f"| feats/canal={feat_dim} => (2*{feat_dim}*N_U + N_Y)")

        # split por par
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
        tr_idx, te_idx = next(gss.split(X, y, groups=groups))
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        g_te = groups[te_idx]

        scaler_X = scaler_factory(args.scaler)
        X_tr = scaler_X.fit_transform(X_tr)
        X_te = scaler_X.transform(X_te)

        backend = args.model
        history = None

        if backend == "tf":
            try:
                tf = importlib.import_module("tensorflow")
                model = build_model_tf(X_tr.shape[1], dropout=float(args.dropout))
                cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
                history = model.fit(
                    X_tr, y_tr,
                    validation_split=VAL_SPLIT,
                    epochs=25,
                    batch_size=128,
                    callbacks=[cb],
                    verbose=1
                )
                pred_te = model.predict(X_te, verbose=0).ravel()
            except Exception as e:
                print("[WARN] TensorFlow falló, uso sklearn:", e)
                backend = "sklearn"
                model = build_model_sklearn(hidden, args.alpha, args.max_iter)
                model.fit(X_tr, y_tr)
                pred_te = model.predict(X_te)
        else:
            model = build_model_sklearn(hidden, args.alpha, args.max_iter)
            model.fit(X_tr, y_tr)
            pred_te = model.predict(X_te)

        mae = mean_absolute_error(y_te, pred_te)
        acc5 = _acc_tolerancia(y_te, pred_te, tol=5.0)
        print(f"[TEST] MAE: {mae:.3f}° | ACC±5°: {acc5:.3f}")

        meta = (
            f"angles={','.join(map(str, ANGLES))}\n"
            f"backend={backend}\n"
            f"n_u={N_U}\n"
            f"n_y={N_Y}\n"
            f"win_ms={args.win_ms}\n"
            f"hop_ms={args.hop_ms}\n"
            f"feat_set={args.feat_set}\n"
            f"zc_th={args.zc_th}\n"
            f"ssc_th={args.ssc_th}\n"
            f"per_file_norm={bool(args.per_file_norm)}\n"
            f"scaler={args.scaler}\n"
            f"y_noise={args.y_noise}\n"
            f"max_samples={args.max_samples}\n"
            f"ar_feed={args.ar_feed}\n"
            f"mix_beta={args.mix_beta}\n"
            f"ema={args.ema}\n"
            f"dy_max={args.dy_max}\n"
            f"y_clip={args.y_clip}\n"
            f"feats_per_channel={feat_dim}\n"
            f"input_dim={X.shape[1]}\n"
        )

        save_artifacts(target_dir, backend, scaler_X, model, meta=meta)

        if backend == "sklearn":
            plot_loss_sklearn(model, figs)
        else:
            plot_history_tf(history, figs)

        report_test_by_group(y_te, pred_te, g_te, target_dir, tol=5.0)

    # autoreg optional
    if args.eval_autoreg:
        eval_autoreg_dual_windows(
            Path(args.data_ch1), args.pattern_ch1,
            Path(args.data_ch2), args.pattern_ch2,
            ANGLES, target_dir,
            scaler_X, model, backend,
            pair_mode=args.pair_mode,
            per_file_norm=bool(args.per_file_norm),
            win_ms=float(args.win_ms),
            hop_ms=float(args.hop_ms),
            feat_set=str(args.feat_set),
            zc_th=float(args.zc_th),
            ssc_th=float(args.ssc_th),
            ar_feed=str(args.ar_feed),
            mix_beta=float(args.mix_beta),
            ema_k=int(args.ema),
            seed_angle=args.seed_angle,
            dy_max=float(args.dy_max),
            y_clip=y_clip
        )


if _name_ == "_main_":
    main()