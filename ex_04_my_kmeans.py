import numpy as np
import pandas as pd
import dtaidistance.dtw as dtw

class MyKMeans:
    def __init__(self, k=3, max_iter=100, distance_metric="euclidean"):
        self.k = k
        self.max_iter = max_iter
        self.distance_metric = distance_metric
        self.centroids = None
        self.inertia_ = None  # Initialisiert zur Fehlervermeidung

    def _validate_input(self, x):
        if not isinstance(x, (np.ndarray, pd.DataFrame)):
            raise ValueError("Input data must be a numpy array or a pandas DataFrame")
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        if x.ndim not in {2, 3}:
            raise ValueError("Input data must be a 2D or 3D array")
        return x

    def _initialize_centroids(self, x):
        n_samples = x.shape[0]
        return x[np.random.choice(n_samples, self.k, replace=False)]

    def _compute_distance(self, x, centroids):
        if self.distance_metric == "euclidean":
            return np.linalg.norm(x[:, np.newaxis] - centroids, axis=-1)
        elif self.distance_metric == "manhattan":
            return np.abs(x[:, np.newaxis] - centroids).sum(axis=-1)
        elif self.distance_metric == "dtw":
            return np.array([
                [dtw.distance(np.squeeze(x_i), np.squeeze(c_i)) for c_i in centroids]
                for x_i in x
            ])
        else:
            raise ValueError("Ungültige Distanzmetrik angegeben.")

    def fit(self, x):
        x = self._validate_input(x)
        self.centroids = self._initialize_centroids(x)

        for _ in range(self.max_iter):
            distances = self._compute_distance(x, self.centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                np.mean(x[labels == i], axis=0) if np.any(labels == i) else self.centroids[i]
                for i in range(self.k)
            ])

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        self.inertia_ = np.sum([
            np.sum((x[labels == i] - self.centroids[i]) ** 2) for i in range(self.k)
        ])

    def predict(self, x):
        x = self._validate_input(x)
        distances = self._compute_distance(x, self.centroids)
        return np.argmin(distances, axis=1)

    def fit_predict(self, x):
        """Neue Methode zur Kombination von fit() und predict()"""
        self.fit(x)
        return self.predict(x)
