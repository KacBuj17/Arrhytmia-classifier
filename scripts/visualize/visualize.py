import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import wfdb
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from dataset.dataset import ECGDataset
from model.model import LSTMClassifier
from utils.project_paths import from_root

WINDOW_SIZE = 360
HALF_WINDOW = WINDOW_SIZE // 2

DATA_DIR = from_root("resources/data/raw/mit-bih-arrhythmia-database-1.0.0")
CKPT_PATH = from_root("resources/checkpoints/best-lstm-epoch=36-val_loss=0.4582.ckpt")
ENCODER_PATH = from_root("resources/checkpoints/label_encoder.pkl")

BATCH_SIZE = 256
PLOTS_DIR = from_root("resources/plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def list_available_records(data_dir):
    return sorted([f.split(".")[0] for f in os.listdir(data_dir) if f.endswith(".dat")])


def extract_windows(record, annotation):
    signal = record.p_signal[:, 0]
    windows = []
    labels = []

    for idx, sym in zip(annotation.sample, annotation.symbol):
        start = idx - HALF_WINDOW
        end = idx + HALF_WINDOW
        if start < 0 or end > len(signal):
            continue
        window = signal[start:end]
        windows.append(window[:, np.newaxis])
        labels.append(sym)

    return np.array(windows), np.array(labels)


def load_data():
    records = list_available_records(DATA_DIR)
    X_all, y_all = [], []

    for rec in records:
        record = wfdb.rdrecord(os.path.join(DATA_DIR, rec))
        annotation = wfdb.rdann(os.path.join(DATA_DIR, rec), "atr")
        X, y = extract_windows(record, annotation)
        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.hstack(y_all)

    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    mask = np.isin(y_all, le.classes_)
    X_all = X_all[mask]
    y_all = y_all[mask]

    y_encoded = le.transform(y_all)

    dataset = ECGDataset(X_all, y_encoded)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    return loader, le, X_all, y_encoded


def evaluate(model, loader, le, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            p = torch.argmax(logits, dim=1)
            preds.extend(p.cpu().numpy())
            targets.extend(y.numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(targets, preds, target_names=le.classes_))


def safe_label(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label)


def plot_ecg_correct_anomalies_all_records(model, le, device, data_dir=DATA_DIR, window_size=360, normal_class='N'):
    half_window = window_size // 2
    seen_classes = set()
    records = list_available_records(data_dir)

    for rec_name in records:
        record = wfdb.rdrecord(os.path.join(data_dir, rec_name))
        annotation = wfdb.rdann(os.path.join(data_dir, rec_name), "atr")
        signal = record.p_signal[:, 0]

        windows, r_positions, true_labels = [], [], []

        for idx, sym in zip(annotation.sample, annotation.symbol):
            start = max(0, idx - half_window)
            end = min(len(signal), idx + half_window)
            window = signal[start:end]
            if len(window) < window_size:
                continue
            windows.append(window[:, np.newaxis])
            r_positions.append(idx)
            true_labels.append(sym)

        if not windows:
            continue

        windows = np.stack(windows)
        X_tensor = torch.tensor(
            (windows - windows.mean(axis=1, keepdims=True)) /
            (windows.std(axis=1, keepdims=True) + 1e-8),
            dtype=torch.float32
        ).to(device)

        model.eval()
        with torch.no_grad():
            logits = model(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        for i in range(len(r_positions)):
            true_label = true_labels[i]
            pred_label = le.classes_[preds[i]]

            if true_label in seen_classes:
                continue

            if true_label != normal_class and true_label == pred_label:
                start = r_positions[i] - half_window
                end = r_positions[i] + half_window
                fragment = signal[start:end]

                plt.figure(figsize=(8, 3))
                plt.plot(fragment, color='black', lw=1)
                plt.scatter(half_window, fragment[half_window], color='red', s=50)
                plt.title(f"Anomaly | True = Pred = {true_label}", color='red')
                plt.xlabel("Próbka w oknie")
                plt.ylabel("Amplituda")
                plt.savefig(os.path.join(PLOTS_DIR, f"{safe_label(true_label)}_anomaly.png"))
                plt.close()
                seen_classes.add(true_label)

            elif true_label == normal_class and normal_class not in seen_classes:
                start = r_positions[i] - half_window
                end = r_positions[i] + half_window
                fragment = signal[start:end]

                plt.figure(figsize=(8, 3))
                plt.plot(fragment, color='black', lw=1)
                plt.scatter(half_window, fragment[half_window], color='green', s=50)
                plt.title(f"Normal | True = Pred = {normal_class}", color='green')
                plt.xlabel("Próbka w oknie")
                plt.ylabel("Amplituda")
                plt.savefig(os.path.join(PLOTS_DIR, f"{safe_label(normal_class)}_normal.png"))
                plt.close()
                seen_classes.add(normal_class)

        if len(seen_classes) >= len(le.classes_):
            break

    if not seen_classes:
        print("Nie znaleziono poprawnie wykrytych klas.")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader, le, X_all, y_all = load_data()

    model = LSTMClassifier.load_from_checkpoint(
        CKPT_PATH,
        input_size=1,
        hidden_size=128,
        num_classes=len(le.classes_),
        strict=False
    )

    model.to(device)

    evaluate(model, loader, le, device)

    plot_ecg_correct_anomalies_all_records(model, le, device)


if __name__ == "__main__":
    main()