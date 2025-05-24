import numpy as np
import pandas as pd
from pathlib import Path
import os


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"The file {data_path} does not exist.")

    data = pd.read_csv(data_path)

    # Check for non-numeric values in feature columns
    feature_columns = [col for col in data.columns if col not in ['labels', 'exp_ids']]
    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"Column {col} contains non-numeric values.")

    data = remove_unlabeled_data(data)
    data = data.dropna()

    if data.empty:
        raise ValueError("The data is empty after removing unlabeled data and dropping NaN values.")

    return data


def remove_unlabeled_data(data: pd.DataFrame) -> pd.DataFrame:
    return data[data['labels'] != -1]


def convert_to_np(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    n_samples, timesteps, features = data.shape
    windows = []

    for i in range(n_samples - sequence_length + 1):
        window = data[i:i + sequence_length]
        # Reshape to (sequence_length*timesteps, features)
        window = window.reshape(-1, features)
        windows.append(window)

    return np.array(windows)


def get_welding_data(path: Path, n_samples: int | None = None, return_sequences: bool = False,
                     sequence_length: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load welding data from CSV or cached numpy files.

    Args:
        path (Path): Path to the CSV data file.
        n_samples (int | None): Number of samples to sample from the data. If None, all data is returned.
        return_sequences (bool): If True, return sequences of length `sequence_length`.
        sequence_length (int): Length of sequences to return.

    Returns:
        tuple: (features, labels, exp_ids), where:
            - features: np.ndarray of shape (n_samples, timesteps, 2) if return_sequences else (n_samples, n_features)
            - labels: np.ndarray
            - exp_ids: np.ndarray
    """

    # Define cache file paths
    base_path = path.parent
    features_cache = base_path / "features.npy"
    labels_cache = base_path / "labels.npy"
    exp_ids_cache = base_path / "exp_ids.npy"

    # Check if cache exists
    if all(os.path.exists(f) for f in [features_cache, labels_cache, exp_ids_cache]):
        features = np.load(features_cache)
        labels = np.load(labels_cache)
        exp_ids = np.load(exp_ids_cache)
    else:
        # Load from CSV and create cache
        df = load_data(path)
        labels, exp_ids, features = convert_to_np(df)

        np.save(features_cache, features)
        np.save(labels_cache, labels)
        np.save(exp_ids_cache, exp_ids)

    # Optional sampling
    if n_samples is not None and n_samples < len(features):
        indices = np.random.choice(len(features), n_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
        exp_ids = exp_ids[indices]

    if return_sequences:
        # Validate shape
        if features.shape[1] % 2 != 0:
            raise ValueError("Unexpected feature shape: number of feature columns must be even.")

        # Create sliding windows
        features = create_sliding_windows_first_dim(features, sequence_length)
        labels = np.array([
            labels[i:i + sequence_length]
            for i in range(len(labels) - sequence_length + 1)
        ])
        exp_ids = np.array([
            exp_ids[i:i + sequence_length]
            for i in range(len(exp_ids) - sequence_length + 1)
        ])

    return features, labels, exp_ids