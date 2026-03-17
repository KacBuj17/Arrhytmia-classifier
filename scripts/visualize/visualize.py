import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import wfdb
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from torch.utils.data import DataLoader, Dataset

WINDOW_SIZE = 360
HALF_WINDOW = WINDOW_SIZE // 2

DATA_DIR = "data/mit-bih-arrhythmia-database-1.0.0"
BATCH_SIZE = 256

CKPT_PATH = "lightning_logs/version_6/checkpoints/epoch=34-step=49245.ckpt"


def list_available_records(data_dir):
    records = []

    for file in os.listdir(data_dir):

        if file.endswith(".dat"):
            records.append(file.split(".")[0])

    records.sort()

    return records


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

        window = (window - np.mean(window)) / np.std(window)

        windows.append(window[:, np.newaxis])
        labels.append(sym)

    return np.array(windows), np.array(labels)


class ECGDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMClassifier(pl.LightningModule):

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.fc1 = torch.nn.Linear(hidden_size, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)

        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.dropout(torch.relu(self.fc1(out)))

        out = self.fc2(out)

        return out


def load_test_dataset():
    records = list_available_records(DATA_DIR)

    X_all = []
    y_all = []

    for rec in records:
        record = wfdb.rdrecord(os.path.join(DATA_DIR, rec))
        annotation = wfdb.rdann(os.path.join(DATA_DIR, rec), "atr")

        X, y = extract_windows(record, annotation)

        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.hstack(y_all)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)

    dataset = ECGDataset(X_all, y_encoded)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    return loader, le, X_all, y_encoded


def predict(model, loader, device):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)

            logits = model(X)

            p = torch.argmax(logits, dim=1)

            preds.extend(p.cpu().numpy())
            targets.extend(y.numpy())

    return np.array(preds), np.array(targets)


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.show()


def plot_class_distribution(y, labels):
    unique, counts = np.unique(y, return_counts=True)

    plt.figure(figsize=(8, 5))

    plt.bar(labels[unique], counts)

    plt.title("Class Distribution")

    plt.xticks(rotation=45)

    plt.show()


def plot_example_signals(X, y_true, y_pred, labels, n=5):
    idx = np.random.choice(len(X), n)

    plt.figure(figsize=(12, 8))

    for i, id_ in enumerate(idx):
        plt.subplot(n, 1, i + 1)

        plt.plot(X[id_])

        plt.title(
            f"True: {labels[y_true[id_]]} | Pred: {labels[y_pred[id_]]}"
        )

    plt.tight_layout()

    plt.show()


def plot_roc_curves(model, loader, num_classes, device):
    model.eval()

    probs = []
    targets = []

    with torch.no_grad():

        for X, y in loader:
            X = X.to(device)

            logits = model(X)

            p = torch.softmax(logits, dim=1)

            probs.append(p.cpu().numpy())
            targets.append(y.numpy())

    probs = np.vstack(probs)
    targets = np.hstack(targets)

    y_bin = label_binarize(targets, classes=np.arange(num_classes))

    plt.figure(figsize=(8, 6))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"class {i} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.title("ROC Curves")

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.legend()

    plt.show()


def plot_tsne(model, loader, device):
    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)

            out, _ = model.lstm(X)

            emb = out[:, -1, :]

            features.append(emb.cpu().numpy())
            labels.append(y.numpy())

    features = np.vstack(features)
    labels = np.hstack(labels)

    tsne = TSNE(n_components=2)

    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        s=5
    )

    plt.title("t-SNE embeddings")

    plt.show()


def plot_confidence_histogram(model, loader, device):
    model.eval()

    confs = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)

            logits = model(X)

            probs = torch.softmax(logits, dim=1)

            conf, _ = torch.max(probs, dim=1)

            confs.extend(conf.cpu().numpy())

    plt.figure(figsize=(8, 5))

    plt.hist(confs, bins=50)

    plt.title("Prediction confidence")

    plt.xlabel("confidence")

    plt.ylabel("count")

    plt.show()


def plot_wrong_predictions(X, y_true, y_pred, labels, n=5):
    wrong = np.where(y_true != y_pred)[0]

    idx = np.random.choice(wrong, min(n, len(wrong)))

    plt.figure(figsize=(12, 8))

    for i, id_ in enumerate(idx):
        plt.subplot(len(idx), 1, i + 1)

        plt.plot(X[id_])

        plt.title(
            f"True: {labels[y_true[id_]]} | Pred: {labels[y_pred[id_]]}"
        )

    plt.tight_layout()

    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader, le, X_all, y_all = load_test_dataset()

    model = LSTMClassifier.load_from_checkpoint(
        CKPT_PATH,
        strict=False
    )

    model.to(device)

    preds, targets = predict(model, loader, device)

    print(classification_report(targets, preds, target_names=le.classes_))

    plot_confusion_matrix(targets, preds, le.classes_)

    plot_class_distribution(targets, le.classes_)

    plot_example_signals(X_all.squeeze(), targets, preds, le.classes_)

    plot_roc_curves(model, loader, len(le.classes_), device)

    plot_tsne(model, loader, device)

    plot_confidence_histogram(model, loader, device)

    plot_wrong_predictions(X_all.squeeze(), targets, preds, le.classes_)


if __name__ == "__main__":
    main()
