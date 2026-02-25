# -- coding: utf-8 --
"""
LIVE EMG -> NARX (dual) -> UDP (MATLAB / Kinova)

FIXES:
1) Un solo CAPTURE por ciclo (con watchdog) -> evita capturar doble.
2) Activity gate bien puesto dentro del while (arregla errores de continue).
3) Spectral siempre devuelve 8 feats/canal (5 basic + 3 spectral).
4) Per-file-norm en entrenamiento => en live usa --per-seg-norm para aproximarlo.
5) capture-sec=3 y wait-sec=3 (cycle-sec = capture + wait).
6) np.trapezoid (quita deprecation warning).
"""

import argparse
import time
import socket
from pathlib import Path
from collections import deque
import re

import numpy as np
import joblib

try:
    import serial  # pyserial
except Exception:
    serial = None


# -------------------- util --------------------
def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    z = (x - med) / (1.4826 * mad)
    return np.clip(z, -10.0, 10.0)

def angles_to_classes(y: np.ndarray, angles):
    A = np.asarray(angles, float).reshape(1, -1)
    y = np.asarray(y, float).reshape(-1, 1)
    idx = np.argmin(np.abs(y - A), axis=1)
    return np.asarray(angles, float)[idx]

def infer_feat_dim_per_channel(expected_features: int, n_u: int, n_y: int) -> int:
    rem = expected_features - n_y
    denom = 2 * n_u
    if rem <= 0 or rem % denom != 0:
        raise ValueError(
            f"No puedo inferir feat_dim: expected={expected_features}, n_u={n_u}, n_y={n_y} "
            f"=> (expected-n_y) debe ser múltiplo de (2*n_u)."
        )
    return rem // denom

def parse_line_to_two_floats(line: str):
    parts = re.split(r"[,\;\t\s]+", line.strip())
    parts = [p for p in parts if p != ""]
    if len(parts) < 2:
        return None
    try:
        a = float(parts[0].replace(",", "."))
        b = float(parts[1].replace(",", "."))
        return a, b
    except:
        return None


# -------------------- features (basic/spectral) --------------------
def bandpower(mag2: np.ndarray, freqs: np.ndarray, f1: float, f2: float) -> float:
    idx = (freqs >= f1) & (freqs <= f2)
    if not np.any(idx):
        return 0.0
    return float(np.trapezoid(mag2[idx], freqs[idx]))

def median_freq(mag2: np.ndarray, freqs: np.ndarray) -> float:
    total = float(np.trapezoid(mag2, freqs))
    if total <= 0:
        return 0.0
    cdf = np.cumsum(mag2)
    cdf = cdf / (cdf[-1] + 1e-12)
    k = int(np.searchsorted(cdf, 0.5))
    k = min(max(k, 0), len(freqs) - 1)
    return float(freqs[k])

def window_features_from_signal(x: np.ndarray, fs: float, win_samp: int, hop_samp: int,
                                feat_set: str, zc_th: float, ssc_th: float):
    x = np.asarray(x, float)
    W = 1 + max(0, (len(x) - win_samp) // hop_samp)
    feats = []

    for w in range(W):
        i0 = w * hop_samp
        i1 = i0 + win_samp
        seg = x[i0:i1]
        if len(seg) < win_samp:
            break

        # basic (5)
        rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
        mav = float(np.mean(np.abs(seg)))
        wl = float(np.sum(np.abs(np.diff(seg))))

        s = seg.copy()
        s[np.abs(s) < zc_th] = 0.0
        zc = float(np.sum((s[:-1] * s[1:]) < 0))

        d1 = np.diff(seg)
        ssc = float(np.sum(((d1[:-1] * d1[1:]) < 0) & (np.abs(d1[:-1] - d1[1:]) > ssc_th)))

        row = [rms, mav, wl, zc, ssc]

        if feat_set == "spectral":
            seg0 = seg - np.mean(seg)
            spec = np.fft.rfft(seg0)
            mag2 = (np.abs(spec) ** 2) / (len(seg0) + 1e-12)
            freqs = np.fft.rfftfreq(len(seg0), d=1.0 / fs)

            bp1 = bandpower(mag2, freqs, 20.0, 60.0)
            bp2 = bandpower(mag2, freqs, 60.0, 120.0)
            mf  = median_freq(mag2, freqs)
            row += [bp1, bp2, mf]  # +3 => total 8

        feats.append(row)

    return np.asarray(feats, float)  # (W, D)


# -------------------- NARX helpers --------------------
def ema_filter(x: np.ndarray, k: int):
    if k <= 1:
        return x
    if x is None or len(x) == 0:
        return x
    y = np.zeros_like(x, float)
    a = 2.0 / (k + 1.0)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1 - a) * y[i - 1]
    return y

def apply_dy_limit(y_new: float, y_prev: float, dy_max: float) -> float:
    if dy_max <= 0:
        return y_new
    dy = y_new - y_prev
    if dy > dy_max:
        return y_prev + dy_max
    if dy < -dy_max:
        return y_prev - dy_max
    return y_new

def clip_y(y: float, y_clip):
    if y_clip is None:
        return y
    lo, hi = y_clip
    return float(np.clip(y, lo, hi))

def feed_value(y_hat: float, angles, ar_feed: str, mix_beta: float, y_prev: float, dy_max: float, y_clip):
    angles_arr = np.asarray(angles, float)
    y_class = float(angles_arr[np.argmin(np.abs(angles_arr - y_hat))])

    if ar_feed == "class":
        y_feed = y_class
    elif ar_feed == "mix":
        b = float(np.clip(mix_beta, 0.0, 1.0))
        y_feed = b * float(y_hat) + (1.0 - b) * y_class
    else:
        y_feed = float(y_hat)

    y_feed = apply_dy_limit(y_feed, y_prev=y_prev, dy_max=dy_max)
    y_feed = clip_y(y_feed, y_clip)
    return y_feed


# -------------------- activity gate --------------------
def segment_activity_score_raw(x1: np.ndarray, x2: np.ndarray, mode: str = "rms") -> float:
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    if mode == "mav":
        a1 = float(np.mean(np.abs(x1)))
        a2 = float(np.mean(np.abs(x2)))
    else:
        a1 = float(np.sqrt(np.mean(x1**2) + 1e-12))
        a2 = float(np.sqrt(np.mean(x2**2) + 1e-12))
    return 0.5 * (a1 + a2)


# -------------------- OOD stats --------------------
def scaler_ood_stats(scaler, X_row: np.ndarray):
    X_row = np.asarray(X_row, float).reshape(1, -1)

    center = None
    scale = None

    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        center = np.asarray(scaler.mean_, float)
        scale = np.asarray(scaler.scale_, float)
    elif hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
        center = np.asarray(scaler.center_, float)
        scale = np.asarray(scaler.scale_, float)

    if center is None or scale is None:
        return None

    scale = np.where(scale == 0, 1.0, scale)
    z = (X_row.ravel() - center) / scale
    az = np.abs(z)
    return {
        "z_abs_max": float(np.max(az)),
        "pct_abs_gt5": float(np.mean(az > 5.0)),
        "pct_abs_gt10": float(np.mean(az > 10.0)),
    }


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser("LIVE dual-EMG -> NARX -> UDP (fixed)")

    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--com", type=str, required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--fs", type=float, default=1000.0)

    ap.add_argument("--angles", type=str, default="30,60,90,120,150")
    ap.add_argument("--n-u", type=int, default=50)
    ap.add_argument("--n-y", type=int, default=1)

    ap.add_argument("--win-ms", type=float, default=400.0)
    ap.add_argument("--hop-ms", type=float, default=50.0)
    ap.add_argument("--feat-set", choices=["basic", "spectral"], default="spectral")
    ap.add_argument("--zc-th", type=float, default=0.02)
    ap.add_argument("--ssc-th", type=float, default=0.02)

    ap.add_argument("--per-seg-norm", action="store_true")
    ap.add_argument("--seed-angle", type=float, default=30.0)

    ap.add_argument("--ar-feed", choices=["cont", "class", "mix"], default="mix")
    ap.add_argument("--mix-beta", type=float, default=0.8)
    ap.add_argument("--ema", type=int, default=3)
    ap.add_argument("--dy-max", type=float, default=12.0)
    ap.add_argument("--y-clip", type=str, default="20,160")

    # ✅ pedido: 3s captura + 3s espera
    ap.add_argument("--capture-sec", type=float, default=3.0)
    ap.add_argument("--wait-sec", type=float, default=3.0)

    ap.add_argument("--decision", choices=["median", "vote"], default="median")
    ap.add_argument("--reset-each-cycle", action="store_true")

    ap.add_argument("--activity-th", type=float, default=0.0)
    ap.add_argument("--activity-mode", choices=["rms", "mav"], default="rms")
    ap.add_argument("--idle-send", choices=["seed", "last", "none"], default="last")
    ap.add_argument("--idle-reset", action="store_true")

    ap.add_argument("--udp-host", type=str, default="127.0.0.1")
    ap.add_argument("--udp-port", type=int, default=5005)

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--print-every", type=int, default=1)

    args = ap.parse_args()

    if serial is None:
        raise RuntimeError("pyserial no está instalado. Instala: python -m pip install pyserial")

    angles = [float(x.strip()) for x in args.angles.split(",") if x.strip()]
    model_dir = Path(args.model_dir)

    scaler = joblib.load(model_dir / "scaler.pkl")
    model = joblib.load(model_dir / "model.pkl")

    expected = int(getattr(scaler, "n_features_in_", 0))
    if expected <= 0:
        raise RuntimeError("No pude leer scaler.n_features_in_")

    feat_dim = infer_feat_dim_per_channel(expected, args.n_u, args.n_y)

    y_clip = None
    if args.y_clip.strip():
        lo, hi = [float(x.strip()) for x in args.y_clip.split(",")]
        y_clip = (lo, hi)

    win_samp = int(round(args.win_ms * 1e-3 * args.fs))
    hop_samp = int(round(args.hop_ms * 1e-3 * args.fs))
    need_samples = int(round(args.capture_sec * args.fs))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ser = serial.Serial(args.com, args.baud, timeout=1)

    ybuf = deque([args.seed_angle] * args.n_y, maxlen=args.n_y)
    last_sent = float(args.seed_angle)

    cycle_i = 0
    t0_all = time.time()

    print(f"[LIVE] expected={expected} feat_dim/canal={feat_dim} feat_set={args.feat_set}")
    print(f"[LIVE] capture={args.capture_sec}s ({need_samples} samples) + wait={args.wait_sec}s  | fs={args.fs}")
    print("Ctrl+C para parar.\n")

    while True:
        cycle_i += 1
        t_cycle0 = time.time()

        do_print = (cycle_i % max(1, args.print_every)) == 0
        if do_print:
            print(f"\n[CYCLE {cycle_i}] start | t={t_cycle0 - t0_all:.2f}s")
            print(f"[CYCLE {cycle_i}] capturing {args.capture_sec:.2f}s ... (need_samples={need_samples})")

        # -------- 1) CAPTURE (watchdog) --------
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        t_cap0 = time.time()
        max_wait = max(5.0, args.capture_sec * 5.0)
        min_ok = max(50, int(0.2 * need_samples))

        ch1, ch2 = [], []
        n_lines = 0
        n_bad = 0
        raw_samples = []

        while len(ch1) < need_samples:
            if time.time() - t_cap0 > max_wait:
                print(f"[ERR] Capture timeout: valid={len(ch1)}/{need_samples} lines={n_lines} bad={n_bad}")
                if raw_samples:
                    print("[ERR] RAW samples:")
                    for s in raw_samples[:5]:
                        print("   ", repr(s))
                break

            line_bytes = ser.readline()
            if not line_bytes:
                continue

            try:
                line = line_bytes.decode(errors="ignore")
            except Exception:
                line = str(line_bytes)

            n_lines += 1
            if args.debug and len(raw_samples) < 5:
                raw_samples.append(line)

            v = parse_line_to_two_floats(line)
            if v is None:
                n_bad += 1
                continue

            a, b = v
            ch1.append(a)
            ch2.append(b)

        if len(ch1) < min_ok:
            if do_print:
                print(f"[WARN] muestras insuficientes ({len(ch1)}/{need_samples}). lines={n_lines} bad={n_bad}")
            time.sleep(max(0.0, args.wait_sec))
            continue

        x1_raw = np.asarray(ch1, float)
        x2_raw = np.asarray(ch2, float)

        if do_print:
            print(f"[CYCLE {cycle_i}] CAPTURE OK: valid={len(ch1)}/{need_samples} | lines={n_lines} bad={n_bad}")

        # -------- 2) ACTIVITY GATE --------
        act = segment_activity_score_raw(x1_raw, x2_raw, mode=args.activity_mode)
        is_idle = (args.activity_th > 0) and (act < args.activity_th)

        if is_idle:
            if args.idle_reset:
                ybuf = deque([args.seed_angle] * args.n_y, maxlen=args.n_y)

            if args.idle_send == "seed":
                y_out = float(args.seed_angle)
                last_sent = y_out
                sock.sendto(f"{y_out:.2f}".encode("utf-8"), (args.udp_host, args.udp_port))
            elif args.idle_send == "last":
                y_out = float(last_sent)
                sock.sendto(f"{y_out:.2f}".encode("utf-8"), (args.udp_host, args.udp_port))

            if do_print:
                print(f"[CYCLE {cycle_i}] IDLE act={act:.6f} (<{args.activity_th})")

            time.sleep(max(0.0, args.wait_sec))
            continue

        # -------- 3) PREPROCESS --------
        x1 = x1_raw.copy()
        x2 = x2_raw.copy()

        if args.per_seg_norm:
            x1 = robust_z(x1)
            x2 = robust_z(x2)

        if args.reset_each_cycle:
            ybuf = deque([args.seed_angle] * args.n_y, maxlen=args.n_y)

        # -------- 4) FEATURES --------
        F1 = window_features_from_signal(x1, args.fs, win_samp, hop_samp, args.feat_set, args.zc_th, args.ssc_th)
        F2 = window_features_from_signal(x2, args.fs, win_samp, hop_samp, args.feat_set, args.zc_th, args.ssc_th)

        W = min(len(F1), len(F2))
        F1, F2 = F1[:W], F2[:W]

        if F1.ndim != 2 or F2.ndim != 2 or W == 0:
            if do_print:
                print(f"[ERR] ventanas inválidas: F1={F1.shape} F2={F2.shape}")
            time.sleep(max(0.0, args.wait_sec))
            continue

        if F1.shape[1] != feat_dim or F2.shape[1] != feat_dim:
            if do_print:
                print(f"[ERR] feat_dim mismatch: F1={F1.shape}, F2={F2.shape}, esperado d={feat_dim}")
            time.sleep(max(0.0, args.wait_sec))
            continue

        start = max(args.n_u, args.n_y) + 1
        if W <= start:
            if do_print:
                print(f"[WARN] Muy pocas ventanas W={W} para n_u={args.n_u}. Sube capture-sec o baja n_u.")
            time.sleep(max(0.0, args.wait_sec))
            continue

        # -------- 5) PREDICT --------
        preds = []
        n_clip_hi = 0
        n_clip_lo = 0
        last_X = None

        for t in range(start, W):
            feats = []
            for k in range(1, args.n_u + 1):
                feats.extend(F1[t - k].tolist())
            for k in range(1, args.n_u + 1):
                feats.extend(F2[t - k].tolist())
            feats.extend(list(ybuf))

            X = np.asarray(feats, float).reshape(1, -1)
            last_X = X

            if X.shape[1] != expected:
                if do_print:
                    print(f"[ERR] X features={X.shape[1]} esperado={expected} (n_u/hop/win/feat_set deben coincidir)")
                preds = []
                break

            Xs = scaler.transform(X)
            y_hat = float(model.predict(Xs)[0])
            preds.append(y_hat)

            if y_clip is not None:
                if y_hat > y_clip[1]:
                    n_clip_hi += 1
                if y_hat < y_clip[0]:
                    n_clip_lo += 1

            y_prev = ybuf[0] if len(ybuf) else args.seed_angle
            y_feed = feed_value(y_hat, angles, args.ar_feed, args.mix_beta, y_prev, args.dy_max, y_clip)
            ybuf.appendleft(y_feed)

        if not preds:
            time.sleep(max(0.0, args.wait_sec))
            continue

        y_pred = np.asarray(preds, float)
        if args.ema and args.ema > 1:
            y_pred = ema_filter(y_pred, args.ema)

        # -------- 6) DECISIÓN --------
        if args.decision == "median":
            y_out = float(np.median(y_pred))
            y_out = float(angles_to_classes(np.array([y_out]), angles)[0])
        else:
            y_cls = angles_to_classes(y_pred, angles)
            vals, counts = np.unique(y_cls, return_counts=True)
            y_out = float(vals[np.argmax(counts)])

        last_sent = float(y_out)

        # -------- 7) UDP SEND --------
        sock.sendto(f"{y_out:.2f}".encode("utf-8"), (args.udp_host, args.udp_port))

        # -------- 8) DEBUG --------
        if do_print:
            ood = None
            if last_X is not None:
                try:
                    ood = scaler_ood_stats(scaler, last_X)
                except Exception:
                    ood = None

            msg = (f"[CYCLE {cycle_i}] ACTIVE -> y_out={y_out:.2f} | "
                   f"pred(min/med/max)={np.min(y_pred):.2f}/{np.median(y_pred):.2f}/{np.max(y_pred):.2f} "
                   f"std={np.std(y_pred):.2f} | Npred={len(y_pred)} W={W} | "
                   f"clip_hi={n_clip_hi} clip_lo={n_clip_lo} | act={act:.6f}")
            if ood is not None:
                msg += f" | OOD zmax={ood['z_abs_max']:.2f} pct>|5|={100*ood['pct_abs_gt5']:.1f}%"
            print(msg)

        # -------- 9) WAIT --------
        time.sleep(max(0.0, args.wait_sec))


if _name_ == "_main_":
    main()