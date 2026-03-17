import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, random_split

from utils.project_paths import from_root

BATCH_SIZE = 64
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
NUM_WORKERS = 4
DATA_PATH = from_root('resources/data/processed/data.csv')


class ECGDataset(Dataset):
    def __init__(self, X, y):
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

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_encoded), y=y_encoded)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    dataset = ECGDataset(X_all, y_encoded)

    total_len = len(dataset)
    test_len = int(total_len * TEST_SPLIT)
    val_len = int(total_len * VAL_SPLIT)
    train_len = total_len - val_len - test_len

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True)

    return train_loader, val_loader, test_loader, le, class_weights
