#!/usr/bin/env python3
import argparse, glob, os, re, json
import numpy as np

def parse_fname(path):
    # expects: modX_snrY_cfoZ.c64
    base = os.path.basename(path)
    m = re.search(r"mod(\d+)_snr(-?\d+)_cfo(-?\d+)", base)
    if not m:
        raise ValueError(f"Filename not parseable: {base}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

def rms_norm(x):
    # compute in float64 to avoid overflow from float32 squaring
    mag2 = (np.real(x).astype(np.float64)**2 + np.imag(x).astype(np.float64)**2)
    p = np.mean(mag2)
    if not np.isfinite(p) or p < 1e-18:
        return None  # signal to skip
    rms = np.sqrt(p)
    y = (x.astype(np.complex64) / np.float32(rms))
    if not np.all(np.isfinite(y)):
        return None
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_glob", default="data/raw/*.c64")
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--skip", type=int, default=10000)  # skip RRC transient
    ap.add_argument("--max_windows_per_file", type=int, default=2000)
    ap.add_argument("--out_npz", default="data/modclf_windows_v1.npz")
    ap.add_argument("--labels_json", default="models/labels.json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.in_glob))
    if not files:
        raise SystemExit("No input files found.")

    labels = ["BPSK", "QPSK", "8PSK", "16QAM"]
    os.makedirs(os.path.dirname(args.labels_json), exist_ok=True)
    with open(args.labels_json, "w") as f:
        json.dump(labels, f, indent=2)

    X_list, y_list = [], []
    snr_list, cfo_list, mod_list = [], [], []

    for fp in files:
        mod, snr, cfo = parse_fname(fp)
        x = np.fromfile(fp, dtype=np.complex64)
        if len(x) < args.skip + args.win:
            continue
        x = x[args.skip:]

        count = 0
        for start in range(0, len(x) - args.win + 1, args.hop):
            w = rms_norm(x[start:start + args.win])
            if w is None:
                continue
            X = np.stack([np.real(w), np.imag(w)], axis=0).astype(np.float32)

            X_list.append(X)
            y_list.append(mod)
            mod_list.append(mod)
            snr_list.append(snr)
            cfo_list.append(cfo)

            count += 1
            if count >= args.max_windows_per_file:
                break

        print(f"{os.path.basename(fp)} -> windows: {count}")

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N,2,win)
    y = np.array(y_list, dtype=np.int64)

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        X=X,
        y=y,
        mod=np.array(mod_list, dtype=np.int64),
        snr_db=np.array(snr_list, dtype=np.int64),
        cfo_hz=np.array(cfo_list, dtype=np.int64),
        win=np.array([args.win], dtype=np.int64),
        hop=np.array([args.hop], dtype=np.int64),
    )

    print("\nSaved:", args.out_npz)
    print("X:", X.shape, "y:", y.shape)

if __name__ == "__main__":
    main()

