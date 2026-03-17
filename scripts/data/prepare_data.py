import os

import pandas as pd
import wfdb

from utils.project_paths import from_root

WINDOW_SIZE = 360
HALF_WINDOW = WINDOW_SIZE // 2
DATA_DIR = from_root('resources/data/raw/mit-bih-arrhythmia-database-1.0.0')
OUT_DATA_PATH = from_root('resources/data/processed/data.csv')


def list_available_records(data_dir):
    records = []
    for file in os.listdir(data_dir):
        if file.endswith('.dat'):
            records.append(file.split('.')[0])
    records.sort()
    return records


def extract_windows_df(record, annotation, record_id):
    signal = record.p_signal[:, 0]
    rows = []
    for idx, sym in zip(annotation.sample, annotation.symbol):
        start = idx - HALF_WINDOW
        end = idx + HALF_WINDOW
        if start < 0 or end > len(signal):
            continue
        window = signal[start:end]
        for i, val in enumerate(window):
            rows.append({
                'ecg': val,
                'window_idx': i,
                'R_idx': idx,
                'symbol': sym,
                'record_id': record_id
            })
    df = pd.DataFrame(rows)
    return df


def prepare_dataset_df(data_dir):
    records = list_available_records(data_dir)
    print(f"Founded records: {records}")
    df_all = pd.DataFrame()
    for rec in records:
        print(f"Processing record: {rec}...")
        record = wfdb.rdrecord(os.path.join(data_dir, rec))
        annotation = wfdb.rdann(os.path.join(data_dir, rec), 'atr')
        df = extract_windows_df(record, annotation, rec)
        df_all = pd.concat([df_all, df], ignore_index=True)
    print(f"Read df: {df_all.shape} rows, columns: {df_all.columns.tolist()}")
    return df_all


def main():
    df = prepare_dataset_df(DATA_DIR)
    print(df.head())
    df.to_csv(OUT_DATA_PATH, index=False)


if __name__ == "__main__":
    main()
