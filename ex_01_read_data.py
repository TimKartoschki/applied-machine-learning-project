import numpy as np
import pandas as pd
from pathlib import Path


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load and preprocess data from a CSV file. Remove rows with unlabeled data.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    data = pd.read_csv(data_path)
    data.dropna(inplace=True)
    data = remove_unlabeled_data(data)

    if data.empty:
        raise ValueError("No labeled data remaining after preprocessing.")

    return data


def remove_unlabeled_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with unlabeled data (where labels == -1).
    """
    return data[data['labels'] != -1]


def convert_to_np(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert DataFrame to numpy arrays, separating labels, experiment IDs, and features.
    """
    voltages_names = data.columns[data.columns.str.startswith('V')]  # alle Voltages namen holen
    current_names = data.columns[data.columns.str.startswith('I')]  # alle Currents namen holen

    # Umwandlung aller daten in Numpy arrays
    voltage_data = data[voltages_names].to_numpy()
    current_data = data[current_names].to_numpy()
    labels = data['labels'].to_numpy()
    experiment_ids = data['exp_ids'].to_numpy()
    # zusammenfügen von den current und voltage data
    data = np.stack([current_data, voltage_data], axis=2)
    return labels, experiment_ids, data


def create_sliding_windows_first_dim(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Create sliding windows over the first dimension of a 3D array.
    """
    n_samples, timesteps, features = data.shape
    if n_samples < sequence_length:
        raise ValueError("Not enough samples to create a single sequence.")

    windows = [
        data[i:i + sequence_length].reshape(sequence_length * timesteps, features)
        for i in range(n_samples - sequence_length + 1)
    ]
    return np.stack(windows)


def convert_to_np(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    voltages_names = data.columns[data.columns.str.startswith('V')]
    current_names = data.columns[data.columns.str.startswith('I')]

    # Prüfe jede Spalte mit versuchter Konvertierung zu float
    for col in voltages_names.union(current_names):
        try:
            _ = data[col].astype(float)
        except ValueError as e:
            raise ValueError(f"Non-numeric data found in column '{col}': {e}")

    voltage_data = data[voltages_names].astype(float).to_numpy()
    current_data = data[current_names].astype(float).to_numpy()
    labels = data['labels'].to_numpy()
    experiment_ids = data['exp_ids'].to_numpy()

    if voltage_data.shape != current_data.shape:
        raise ValueError(f"Voltage and current data have mismatched shapes: {voltage_data.shape} vs {current_data.shape}")

    data_combined = np.stack([current_data, voltage_data], axis=2)

    return labels, experiment_ids, data_combined


def convert_to_np(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    voltages_names = data.columns[data.columns.str.startswith('V')]
    current_names = data.columns[data.columns.str.startswith('I')]

    # Prüfe jede Spalte mit versuchter Konvertierung zu float
    for col in voltages_names.union(current_names):
        try:
            _ = data[col].astype(float)
        except ValueError as e:
            raise ValueError(f"Non-numeric data found in column '{col}': {e}")

    voltage_data = data[voltages_names].astype(float).to_numpy()
    current_data = data[current_names].astype(float).to_numpy()
    labels = data['labels'].to_numpy()
    experiment_ids = data['exp_ids'].to_numpy()

    if voltage_data.shape != current_data.shape:
        raise ValueError(f"Voltage and current data have mismatched shapes: {voltage_data.shape} vs {current_data.shape}")

    data_combined = np.stack([current_data, voltage_data], axis=2)

    return labels, experiment_ids, data_combined
