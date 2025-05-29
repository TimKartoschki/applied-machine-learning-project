import numpy as np
import pandas as pd
from scipy.signal import detrend, windows


def find_dominant_frequencies(x: np.ndarray, fs: int) -> np.ndarray:
    """
    Calculates the dominant frequencies of multiple input signals using FFT.

    Args:
        x (np.ndarray): Input signals, shape: (num_samples, seq_len)
        fs (int): Sampling frequency

    Returns:
        np.ndarray: Dominant frequencies per sample, shape: (num_samples,)
    """
    num_samples, seq_len = x.shape
    window = windows.hann(seq_len)
    freqs = np.fft.rfftfreq(seq_len, d=1/fs)
    dominant_freqs = np.zeros(num_samples)

    for i in range(num_samples):
        signal = detrend(x[i]) * window
        fft_vals = np.abs(np.fft.rfft(signal))
        dominant_freqs[i] = freqs[np.argmax(fft_vals)]

    return dominant_freqs


def extract_features(data: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    Extracts 20 features from voltage and current time series.

    Args:
        data (np.ndarray): Time series data (n_samples, n_timesteps, 2)
        labels (np.ndarray): Class labels

    Returns:
        pd.DataFrame: Features as DataFrame
    """
    n_samples = data.shape[0]
    features = []

    for i in range(n_samples):
        voltage = data[i, :, 0]
        current = data[i, :, 1]

        feat = {
            # Voltage features
            "volt_mean": np.mean(voltage),
            "volt_std": np.std(voltage),
            "volt_max": np.max(voltage),
            "volt_min": np.min(voltage),
            "volt_skew": pd.Series(voltage).skew,
            "volt_kurt": pd.Series(voltage).kurt,
            "volt_median": np.median(voltage),
            "volt_range": np.ptp(voltage),
            "volt_energy": np.sum(voltage**2),
            "volt_dominant_freq": find_dominant_frequencies(voltage[np.newaxis, :], fs=1000)[0],

            # Current features
            "curr_mean": np.mean(current),
            "curr_std": np.std(current),
            "curr_max": np.max(current),
            "curr_min": np.min(current),
            "curr_skew": pd.Series(current).skew,
            "curr_kurt": pd.Series(current).kurt,
            "curr_median": np.median(current),
            "curr_range": np.ptp(current),
            "curr_energy": np.sum(current**2),
            "curr_dominant_freq": find_dominant_frequencies(current[np.newaxis, :], fs=1000)[0],

            # Label
            "label": labels[i]
        }

        features.append(feat)

    return pd.DataFrame(features)

