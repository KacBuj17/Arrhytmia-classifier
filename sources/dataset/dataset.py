import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from utils.project_paths import from_root

BATCH_SIZE = 64
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
NUM_WORKERS = 4
MIN_SAMPLES_PER_CLASS = 50

DATA_PATH = from_root('resources/data/processed/data.csv')


class ECGDataset(Dataset):
    def __init__(self, X, y):
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_dataloaders():
    df = pd.read_csv(DATA_PATH)

    windows = df.groupby(['record_id', 'R_idx']).apply(
        lambda x: x.sort_values('window_idx')['ecg'].values
    )
    labels = df.groupby(['record_id', 'R_idx'])['symbol'].first()

    X_all = np.stack(windows.values)[:, :, np.newaxis]
    y_all = labels.values

    counts = pd.Series(y_all).value_counts()
    valid_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    mask = np.isin(y_all, valid_classes)
    X_all = X_all[mask]
    y_all = y_all[mask]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)

    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all,
        y_encoded,
        test_size=TEST_SPLIT,
        stratify=y_encoded,
        random_state=42
    )
    val_size = VAL_SPLIT / (1 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        stratify=y_temp,
        random_state=42
    )

    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)

    class_counts = np.bincount(y_train)
    weights = 1.0 / class_counts
    sample_weights = weights[y_train]

    sample_weights = 0.5 * sample_weights + 0.5
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader, le, class_weights
