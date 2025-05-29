import numpy as np
import pandas as pd
import dtaidistance.dtw as dtw

class MyKMeans:
    def __init__(self, k=3, max_iter=100, distance_metric="euclidean"):
        self.k = k
        self.max_iter = max_iter
        self.distance_metric = distance_metric
        self.centroids = None
        self.inertia_ = None

    def _validate_input(self, x):
        if not isinstance(x, (np.ndarray, pd.DataFrame)):
            raise ValueError("Input data must be a numpy array or a pandas DataFrame")
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy(dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)  # Stelle sicher, dass die Daten keine 'object'-Arrays sind
        if x.ndim not in {2, 3}:
            raise ValueError("Input data must be a 2D or 3D array")
        return x

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
            return np.array([[dtw.distance(np.squeeze(x_i), np.squeeze(c_i)) for c_i in centroids] for x_i in x], dtype=np.float64)
        else:
            raise ValueError("Ungültige Distanzmetrik angegeben.")

    def fit(self, x):
        x = self._validate_input(x)

        if x.ndim == 3:  # Falls Sequenzen unterschiedlich lang sind, angleichen
            max_length = max(seq.shape[0] for seq in x)
            padded_x = np.array([
                np.pad(seq, ((0, max_length - seq.shape[0]), (0, 0)), 'constant') if seq.shape[0] < max_length else seq
                for seq in x
            ], dtype=np.float64)
        else:
            padded_x = x

        self.centroids = self._initialize_centroids(padded_x)

        for _ in range(self.max_iter):
            distances = self._compute_distance(padded_x, self.centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                np.mean(padded_x[labels == i], axis=0) if np.any(labels == i) else self.centroids[i]
                for i in range(self.k)
            ], dtype=np.float64)

            if np.allclose(self.centroids, new_centroids):  # Korrekte Überprüfung auf Konvergenz
                break

            self.centroids = new_centroids

        self.inertia_ = np.sum([np.sum((padded_x[labels == i] - self.centroids[i]) ** 2) for i in range(self.k)])

    def predict(self, x):
        x = self._validate_input(x)
        distances = self._compute_distance(x, self.centroids)
        return np.argmin(distances, axis=1)

    def fit_predict(self, x: np.ndarray):
        """
        Fit the K-means model to the data and return the predicted labels.
        """
        if isinstance(x, pd.DataFrame):
            x = x.values
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise ValueError("Input data must be a numpy array or a pandas DataFrame")
        self.fit(x)
        return self.predict(x)