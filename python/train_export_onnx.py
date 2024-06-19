#!/usr/bin/env python3
import os, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DATA_NPZ = "data/modclf_windows_v1.npz"
LABELS_JSON = "models/labels.json"
OUT_ONNX = "models/modclf.onnx"
OUT_CURVES = "results/training_curves.png"
OUT_CM = "results/confusion_matrix.png"

EPOCHS = 10
BATCH = 256
LR = 1e-3
SEED = 0

class IQDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # (N,2,L) float32
        self.y = torch.from_numpy(y)  # (N,) int64

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class ModCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def eval_loader(model, loader, crit, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
    return loss_sum / total, correct / total

def main():
    set_seed(SEED)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    d = np.load(DATA_NPZ)
    X = d["X"]  # (N,2,1024)
    y = d["y"]

    with open(LABELS_JSON, "r") as f:
        labels = json.load(f)

    # stratified split: train/val/test = 70/15/15
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp
    )

    train_loader = DataLoader(IQDataset(X_train, y_train), batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(IQDataset(X_val, y_val), batch_size=BATCH, shuffle=False)
    test_loader  = DataLoader(IQDataset(X_test, y_test), batch_size=BATCH, shuffle=False)

    device = "cpu"  # you don't have GPU; keep it explicit
    model = ModCNN(n_classes=len(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    train_loss_hist, val_loss_hist, val_acc_hist = [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)

        tr_loss = running / n
        va_loss, va_acc = eval_loader(model, val_loader, crit, device)

        train_loss_hist.append(tr_loss)
        val_loss_hist.append(va_loss)
        val_acc_hist.append(va_acc)

        print(f"epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.4f}")

    # Plot training curves
    plt.figure()
    plt.plot(train_loss_hist, label="train loss")
    plt.plot(val_loss_hist, label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(OUT_CURVES, dpi=200, bbox_inches="tight")
    plt.close()

    # Confusion matrix on test set
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.append(pred)
            y_true.append(yb.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig = plt.figure()
    disp.plot(values_format="d")
    plt.savefig(OUT_CM, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Export ONNX
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, 2, X.shape[2], dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy,
        OUT_ONNX,
        input_names=["iq"],
        output_names=["logits"],
        opset_version=13,
    )

    print("Saved:", OUT_ONNX)
    print("Saved:", OUT_CURVES)
    print("Saved:", OUT_CM)

if __name__ == "__main__":
    main()

