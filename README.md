# sdr-ModRec — Modulation Recognition Using GNU Radio

A **SDR + ML** project that performs **real-time modulation classification** in **GNU Radio**.

- **Modulations:** BPSK, QPSK, 8PSK, 16QAM  
- **Channel impairments:** AWGN (SNR), CFO (frequency offset)  
- **Training:** small 1D CNN (PyTorch, CPU-friendly)  
- **Deployment:** real-time inference inside GNU Radio via **ONNXRuntime**  
- **Demo UI:** QT GUI controls + a big-text **popup** showing `BPSK/QPSK/8PSK/16QAM` (plus optional message/console output)

---

## Workflow

1. **DSP chain in GNU Radio** generates synthetic IQ (BPSK/QPSK/8PSK/16QAM).
2. A **Channel Model** adds controlled impairments (AWGN SNR, CFO).
3. Raw IQ is saved to disk (`.c64` complex64).
4. Python builds an ML dataset by slicing IQ into fixed-length windows and normalizing.
5. A small **1D CNN** is trained (CPU OK) and exported to **ONNX**.
6. GNU Radio runs **real-time inference** with ONNXRuntime and shows the predicted label live.

RF hardware is optional — everything is reproducible in loopback.

---

## Repo structure

```text
sdr-ModRec/
├── grc/
│   ├── modclf_dataset_gen.grc      # headless IQ generator (No GUI)
│   └── modclf_live_infer.grc       # QT GUI live inference demo
├── python/
│   ├── modgen.py                   # compiled/derived generator script (CLI params)
│   ├── make_windows.py             # .c64 -> windowed dataset .npz
│   └── train_export_onnx.py         # train CNN + export models/modclf.onnx
├── models/
│   ├── modclf.onnx                 # exported model used by GNU Radio
│   └── labels.json                 # class order (index -> name)
├── results/
│   ├── confusion_matrix.png
│   └── training_curves.png
└── data/                           # generated locally, NOT tracked in git
    ├── raw/                        # raw IQ .c64 files (generated)
    └── modclf_windows_v1.npz        # windowed ML dataset (generated, large)
```

---

## Requirements

- Linux recommended (tested workflow)
- Conda / Miniconda
- GNU Radio **3.10** (via conda-forge)
- ONNXRuntime (CPU)
- PyTorch (CPU)

> **No GPU required.** Training is CPU-friendly (small CNN + moderate dataset).

---

## Install (Conda, GNU Radio 3.10)

Create a dedicated conda env so you avoid “externally-managed-environment” (PEP668) issues with system Python.

```bash
# Create env (Python 3.10 is a safe match for GNU Radio 3.10)
conda create -n gr310 python=3.10 -c conda-forge -y
conda activate gr310

# Install GNU Radio + ML/runtime deps
conda install -c conda-forge -y \
  gnuradio=3.10 numpy matplotlib scikit-learn tqdm onnx onnxruntime

# CPU PyTorch
conda install -c pytorch -c conda-forge -y pytorch
```

Verify:

```bash
python -c "import gnuradio; import onnxruntime as ort; print('GNURadio OK, ORT', ort.__version__)"
python -c "from PyQt5 import QtWidgets; print('PyQt OK')"
gnuradio-config-info --version
```

**Important:** Always launch GRC from the conda env:

```bash
conda activate gr310
gnuradio-companion
```

---

## End-to-end pipeline

### Step 1 — Generate raw IQ files (GNU Radio loopback)

This uses `grc/modclf_dataset_gen.grc` (Generate Options: **No GUI**) to synthesize IQ and save `.c64` files.

#### 1A) Compile the dataset generator `.grc` into a runnable Python script

```bash
conda activate gr310
cd sdr-ModRec
grcc grc/modclf_dataset_gen.grc
```

This produces (or updates) a runnable script. In this repo the runnable generator is `python/modgen.py`.
(Depending on your setup, `grcc` may output a `.py` in `grc/`; if so, copy/rename it to `python/modgen.py` or run it directly.)

#### 1B) Generate a sweep (recommended starter dataset)

This makes **24 files**: 4 mods × 3 SNR × 2 CFO.

```bash
mkdir -p data/raw

for mod in 0 1 2 3; do
  for snr in 0 10 20; do
    for cfo in 0 500; do
      out="data/raw/mod${mod}_snr${snr}_cfo${cfo}.c64"
      echo "Generating $out"
      python3 python/modgen.py \
        --mod-sel ${mod} \
        --snr-db ${snr} \
        --cfo-hz ${cfo} \
        --n-samps 300000 \
        --out-file "${out}"
    done
  done
done
```
---

### Step 2 — Convert raw IQ to a windowed dataset (.npz)

The CNN expects fixed-length windows of IQ.

Windowing settings (defaults used here):
- `win = 1024` samples
- `hop = 256` samples
- per-window RMS normalization (computed safely in float64)
- input tensor shape for CNN: `(2, win)` = `[I; Q]`

Generate the large dataset file (not tracked in git):

```bash
conda activate gr310
cd sdr-ModRec

python3 python/make_windows.py \
  --in_glob "data/raw/*.c64" \
  --win 1024 --hop 256 \
  --max_windows_per_file 2000 \
  --out_npz data/modclf_windows_v1.npz
```

---

### Step 3 — Train the CNN (CPU) and export ONNX

This step:
- loads `data/modclf_windows_v1.npz`
- trains a small 1D CNN on CPU
- saves:
  - `results/training_curves.png`
  - `results/confusion_matrix.png`
  - `models/modclf.onnx`
  - `models/labels.json`

Run training + export:

```bash
conda activate gr310
cd sdr-ModRec
python3 python/train_export_onnx.py
```

---

### Step 4 — Run real-time inference in GNU Radio (QT GUI demo)

Open the live demo flowgraph:

```bash
conda activate gr310
cd sdr-ModRec
gnuradio-companion grc/modclf_live_infer.grc
```

Click **Run** inside GRC.

#### What the live flowgraph does
- generates IQ for the selected modulation (QT GUI chooser)
- applies SNR + CFO impairments in a Channel Model
- runs ONNX inference inside an Embedded Python Block
- outputs:
  - numeric class index (0..3) to a QT time sink
  - a popup window that displays the class name (`BPSK`, `QPSK`, `8PSK`, `16QAM`)
  - (optional) message/console output of label changes
---

## Model details

### Input representation
For each window of length `L=1024` complex samples:

- Normalize window by RMS power
- Build tensor:
  - `X[0, :] = I` (real)
  - `X[1, :] = Q` (imag)

Final model input shape:
- **(batch, 2, 1024)** float32

### Architecture (1D CNN)
The training script uses a compact CNN:

- Conv1d(2 → 32, k=7) + ReLU + MaxPool(2)
- Conv1d(32 → 64, k=5) + ReLU + MaxPool(2)
- Conv1d(64 → 128, k=3) + ReLU + AdaptiveAvgPool(1)
- Flatten
- Linear(128 → 4 classes)

This is intentionally small so it trains quickly on CPU and runs fast in real-time inference.

### Output classes
Class index mapping is stored in:

- `models/labels.json`

Default order:
- 0: BPSK
- 1: QPSK
- 2: 8PSK
- 3: 16QAM

---

## Dataset format

`data/modclf_windows_v1.npz` contains:

- `X`: float32, shape `(N, 2, 1024)`
- `y`: int64, shape `(N,)`  (class index 0..3)
- `snr_db`: int64, shape `(N,)`  (for analysis/debug)
- `cfo_hz`: int64, shape `(N,)`
- `win`: int64, shape `(1,)`
- `hop`: int64, shape `(1,)`
