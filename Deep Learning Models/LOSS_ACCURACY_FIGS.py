# narx_make_figs.py
# Genera: loss_*.png, confusion_matrix.png, accuracy_by_class.png, accuracy_vs_tolerance.png, metrics.txt

import argparse, importlib
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RMS_WIN = 32
N_U = 32
N_Y = 4
SEED = 2025
np.random.seed(SEED)

# --------- utilidades de datos (idénticas a tu entrenamiento) ----------
def read_emg_txt(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python", header=None)
    except Exception:
        df = pd.read_csv(path, sep="\t", header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Archivo sin 2 columnas: {path}")
    df = df.iloc[:, :2]
    df.columns = ["f", "amp"]
    return df

def emg_rms_envelope(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, float)
    sq = np.abs(x) ** 2
    kern = np.ones(win, float) / win
    mv = np.convolve(sq, kern, mode="same")
    return np.sqrt(mv + 1e-12)

def assign_blocks(n: int, angles: List[float]) -> np.ndarray:
    m = len(angles)
    q = n // m
    y = np.empty(n, float)
    for i, ang in enumerate(angles):
        beg = i * q
        end = n if i == m - 1 else (i + 1) * q
        y[beg:end] = ang
    return y

def make_narx_features(u: np.ndarray, y: np.ndarray, n_u: int, n_y: int) -> Tuple[np.ndarray, np.ndarray]:
    T = len(y); start = max(n_u, n_y) + 1
    if T <= start: return np.empty((0, n_u+n_y)), np.empty((0,))
    Xs, Ys = [], []
    for t in range(start, T):
        u_lags = [u[t-k] for k in range(1, n_u+1)]
        y_lags = [y[t-k] for k in range(1, n_y+1)]
        Xs.append(u_lags + y_lags); Ys.append(y[t])
    return np.asarray(Xs, float), np.asarray(Ys, float)

def collect_dataset(base: Path, pattern: str, angles: List[float]) -> Tuple[np.ndarray, np.ndarray, int]:
    files = sorted(base.glob(pattern))
    X_list, y_list = [], []
    for fp in files:
        try:
            df = read_emg_txt(fp)
        except Exception as e:
            print(f"[SKIP] {fp.name}: {e}"); continue
        u_env = emg_rms_envelope(df["amp"].values, RMS_WIN)
        y_gt = assign_blocks(len(u_env), angles)
        X_f, y_f = make_narx_features(u_env, y_gt, N_U, N_Y)
        if X_f.size == 0:
            print(f"[SKIP] {fp.name}: segmento corto"); continue
        X_list.append(X_f); y_list.append(y_f)
    if not X_list: raise RuntimeError("Sin archivos válidos.")
    X = np.vstack(X_list); y = np.concatenate(y_list)
    return X, y, len(files)

# ------------------- métricas y gráficos -------------------
def angles_to_classes(y: np.ndarray, angles: List[float]) -> np.ndarray:
    A = np.asarray(angles, float).reshape(1, -1)
    y = np.asarray(y, float).reshape(-1, 1)
    idx = np.argmin(np.abs(y - A), axis=1)
    return np.asarray(angles, float)[idx]

def acc_tolerancia(y_true: np.ndarray, y_pred: np.ndarray, tol: float) -> float:
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    return float(np.mean(np.abs(y_true - y_pred) <= tol))

def plot_confmat(y_true_cls, y_pred_cls, angles, out: Path):
    labels = np.asarray(angles, float)
    # matriz manual para conservar orden de labels
    L = len(labels)
    cm = np.zeros((L, L), dtype=int)
    # map valor->índice
    pos = {v:i for i,v in enumerate(labels)}
    for yt, yp in zip(y_true_cls, y_pred_cls):
        cm[pos[yt], pos[yp]] += 1

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Matriz de confusión (clases en °)")
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_xticks(range(L)); ax.set_yticks(range(L))
    ax.set_xticklabels(labels.astype(int)); ax.set_yticklabels(labels.astype(int))
    for i in range(L):
        for j in range(L):
            ax.text(j, i, cm[i,j], ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)

def plot_loss_sklearn(model, out: Path):
    if not hasattr(model, "loss_curve_"): return
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(model.loss_curve_, label="train loss")
    if hasattr(model, "validation_scores_") and getattr(model, "validation_scores_", None) is not None:
        ax2 = ax.twinx()
        ax2.plot(model.validation_scores_, alpha=0.7, label="val score (sklearn)")
        ax2.set_ylabel("val score"); ax2.legend(loc="lower right")
    ax.set_xlabel("iteración"); ax.set_ylabel("loss"); ax.set_title("Curva de entrenamiento (sklearn)")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out / "loss_sklearn.png", dpi=150); plt.close(fig)

def plot_accuracy_by_class(y_true_cls, y_pred_cls, angles, out: Path):
    labels = np.asarray(angles, float)
    pos = {v:i for i,v in enumerate(labels)}
    tot = np.zeros(len(labels), int)
    ok  = np.zeros(len(labels), int)
    for yt, yp in zip(y_true_cls, y_pred_cls):
        i = pos[yt]; tot[i] += 1; ok[i] += int(yt == yp)
    acc = np.divide(ok, np.maximum(tot, 1))
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(labels.astype(int), acc)
    ax.set_ylim(0,1); ax.set_xlabel("Clase (°)"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy por clase")
    ax.grid(True, axis="y", alpha=0.3)
    for x,v in zip(labels.astype(int), acc):
        ax.text(x, v+0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "accuracy_by_class.png", dpi=150); plt.close(fig)

def plot_accuracy_vs_tolerance(y_true, y_pred, out: Path, tols=(1,2,3,5,7,10,12,15)):
    vals = [acc_tolerancia(y_true, y_pred, t) for t in tols]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(list(tols), vals, marker="o")
    ax.set_xlabel("Tolerancia (°)"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs tolerancia")
    ax.set_ylim(0,1); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "accuracy_vs_tolerance.png", dpi=150); plt.close(fig)

# ------------------- carga artefactos + predicción -------------------
def load_artifacts(model_dir: Path):
    backend = "sklearn"
    if (model_dir / "backend.txt").exists():
        backend = (model_dir / "backend.txt").read_text(encoding="utf-8").strip()
    scaler = joblib.load(model_dir / "scaler.pkl")
    if (model_dir / "model.keras").exists():
        keras_models = importlib.import_module("tensorflow.keras.models")
        model = keras_models.load_model(model_dir / "model.keras")
        backend = "tf"
    else:
        model = joblib.load(model_dir / "model.pkl")
        backend = "sklearn"
    return scaler, model, backend

def predict(model, backend: str, X: np.ndarray) -> np.ndarray:
    if backend == "tf":
        return model.predict(X, verbose=0).ravel()
    return model.predict(X).ravel()

# ------------------- main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-dir", required=True, help="Carpeta del modelo (scaler.pkl, model.pkl/.keras)")
    ap.add_argument("--data", required=True, help="Carpeta con H*.txt seleccionados")
    ap.add_argument("--pattern", default="*.txt")
    ap.add_argument("--angles", required=True, help="Ángulos CSV, ej: '10,20,30,40,50,60,70,80,90'")
    args = ap.parse_args()

    model_dir = Path(args.load_dir)
    data_dir  = Path(args.data)
    angles = [float(a) for a in args.angles.split(",")]

    out = model_dir / "figs"
    out.mkdir(parents=True, exist_ok=True)

    scaler, model, backend = load_artifacts(model_dir)
    X, y, nfiles = collect_dataset(data_dir, args.pattern, angles)
    Xn = scaler.transform(X)

    y_pred = predict(model, backend, Xn)

    # clases
    y_cls     = angles_to_classes(y, angles)
    y_predcls = angles_to_classes(y_pred, angles)

    # guardar métricas de resumen
    acc_cls = float(np.mean(y_cls == y_predcls))
    acc_5   = acc_tolerancia(y, y_pred, 5.0)
    (out / "metrics.txt").write_text(
        f"backend={backend}\nfiles={nfiles}\nclass_accuracy={acc_cls:.3f}\nacc_±5deg={acc_5:.3f}\n",
        encoding="utf-8"
    )

    # figuras
    if backend == "sklearn":
        plot_loss_sklearn(model, out)
    plot_confmat(y_cls, y_predcls, angles, out)
    plot_accuracy_by_class(y_cls, y_predcls, angles, out)
    plot_accuracy_vs_tolerance(y, y_pred, out)

if _name_ == "_main_":
    main()