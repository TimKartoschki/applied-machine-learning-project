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


def get_welding_data(path: Path, n_samples: int | None = None, return_sequences: bool = False,
                     sequence_length: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load welding data from CSV or cached numpy files.
    """
    features_cache = path.with_suffix('.features.npy')
    labels_cache = path.with_suffix('.labels.npy')
    exp_ids_cache = path.with_suffix('.exp_ids.npy')

    if features_cache.exists() and labels_cache.exists() and exp_ids_cache.exists():
        data_np = np.load(features_cache)
        labels = np.load(labels_cache)
        exp_ids = np.load(exp_ids_cache)
    else:
        data = load_data(path)
        labels, exp_ids, data_np = convert_to_np(data)
        np.save(features_cache, data_np)
        np.save(labels_cache, labels)
        np.save(exp_ids_cache, exp_ids)

    if n_samples is not None:
        indices = np.random.choice(len(data_np), n_samples, replace=False)
        data_np = data_np[indices]
        labels = labels[indices]
        exp_ids = exp_ids[indices]

    if return_sequences:
        data_np = data_np.reshape(-1, 1, data_np.shape[1])  # Reshape to (samples, timesteps=1, features)
        data_np = create_sliding_windows_first_dim(data_np, sequence_length)
        labels = labels[sequence_length - 1:]
        exp_ids = exp_ids[sequence_length - 1:]

    return data_np, labels, exp_ids
