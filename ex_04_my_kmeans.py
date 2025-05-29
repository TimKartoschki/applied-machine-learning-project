import numpy as np
import pandas as pd
import dtaidistance.dtw as dtw
from tqdm import tqdm

class MyKMeans:
    def __init__(self, k=3, max_iter=100, distance_metric="euclidean"):
        if distance_metric not in {"euclidean", "manhattan", "dtw"}:
            raise ValueError(f"Ungültige Distanzmetrik: {distance_metric}")
        self.k = k
        self.max_iter = max_iter
        self.distance_metric = distance_metric
        self.centroids = None
        self.inertia_ = None

    def _validate_input(self, x):
        if not isinstance(x, (np.ndarray, pd.DataFrame)):
            raise ValueError("Input muss ein numpy array oder pandas DataFrame sein.")
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy(dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            raise ValueError("Input data must be at least a 2D array.")
        if x.ndim not in {2, 3}:
            raise ValueError("Input muss ein 2D oder 3D Array sein.")
        return x

    def _dtw(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        n_samples, n_features = x.shape[0], centroids.shape[0]
        distances = np.zeros((n_samples, n_features))

        for i in tqdm(range(n_samples), desc="DTW computation"):
            for j in range(n_features):
                if x[i].ndim == 2 and centroids[j].ndim == 2:
                    d = np.mean([dtw.distance(x[i][:, d], centroids[j][:, d]) for d in range(x[i].shape[1])])
                else:
                    d = dtw.distance(x[i], centroids[j])
                distances[i, j] = d
        return distances

    def _initialize_centroids(self, x):
        n_samples = x.shape[0]
        return x[np.random.choice(n_samples, self.k, replace=False)]

    def _compute_distance(self, x, centroids):
        x = np.asarray(x, dtype=np.float64)
        centroids = np.asarray(centroids, dtype=np.float64)

        if self.distance_metric == "euclidean":
            return np.linalg.norm(x[:, np.newaxis] - centroids, axis=-1)
        elif self.distance_metric == "manhattan":
            return np.abs(x[:, np.newaxis] - centroids).sum(axis=-1)
        elif self.distance_metric == "dtw":
            return self._dtw(x, centroids)
        else:
            raise ValueError("Ungültige Distanzmetrik angegeben.")

    def fit(self, x):
        x = self._validate_input(x)
        padded_x = self._pad_sequences(x) if x.ndim == 3 else x

        self.centroids = self._initialize_centroids(padded_x)

        for _ in range(self.max_iter):
            distances = self._compute_distance(padded_x, self.centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                np.mean(padded_x[labels == i], axis=0) if np.any(labels == i) else self.centroids[i]
                for i in range(self.k)
            ], dtype=np.float64)

            if np.allclose(self.centroids, new_centroids.reshape(self.centroids.shape), atol=1e-6):
                break

            self.centroids = new_centroids

        self.inertia_ = np.sum([np.sum((padded_x[labels == i] - self.centroids[i]) ** 2) for i in range(self.k)])

    def predict(self, x):
        x = self._validate_input(x)
        distances = self._compute_distance(x, self.centroids)
        return np.argmin(distances, axis=1)

    def fit_predict(self, x):
        self.fit(x)
        return self.predict(x)

    def _pad_sequences(self, x):
        max_length = max(seq.shape[0] for seq in x)
        return np.array([
            np.pad(seq, ((0, max_length - seq.shape[0]), (0, 0)), 'constant') if seq.shape[0] < max_length else seq
            for seq in x
        ], dtype=np.float64)
