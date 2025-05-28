from typing import Literal
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from dtaidistance import dtw

DISTANCE_METRICS = Literal["euclidean", "manhattan", "dtw"]
INIT_METHOD = Literal["random", "kmeans++"]

class MyKMeans:
    """
    Custom K-means clustering implementation with support for multiple distance metrics.
    
    Args:
        k (int): Number of clusters.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        distance_metric (str, optional): Distance metric to use. Options are "euclidean", 
                                         "manhattan", or "dtw". Defaults to "euclidean".
        init_method (str, optional): Initialization method to use. Options are "kmeans++" or "random". Defaults to "kmeans++".
    """
    def __init__(self, k: int, max_iter: int = 100, distance_metric: DISTANCE_METRICS = "euclidean", init_method: INIT_METHOD = "kmeans++"):
        self.k: int = k
        self.max_iter: int = max_iter
        self.centroids: np.ndarray | None = None
        self.distance_metric: DISTANCE_METRICS = distance_metric
        self.inertia_: float | None = None
        self.init_method: INIT_METHOD = init_method

    def fit(self, x: np.ndarray | pd.DataFrame):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        n_samples = x.shape[0]
        self.centroids = self._initialize_centroids(x)

        for i in range(self.max_iter):
            # Schritt 1: Clusterzuweisung
            distances = self._compute_distance(x, self.centroids)
            labels = np.argmin(distances, axis=1)

            # Schritt 2: Zentroiden aktualisieren
            new_centroids = []
            for j in range(self.k):
                members = x[labels == j]
                if len(members) == 0:
                    # Leerer Cluster: zufällig neuen Punkt wählen
                    new_centroid = x[np.random.randint(0, n_samples)]
                else:
                    new_centroid = np.mean(members, axis=0)
                new_centroids.append(new_centroid)
            new_centroids = np.array(new_centroids)

            # Prüfen auf Konvergenz
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        # Inertia berechnen
        self.inertia_ = np.sum(np.min(self._compute_distance(x, self.centroids), axis=1))
        return self


    def _initialize_centroids(self, x: np.ndarray) -> np.ndarray:
        n_samples = x.shape[0]
        if self.init_method == "random":
            indices = np.random.choice(n_samples, self.k, replace=False)
            return x[indices]
        elif self.init_method == "kmeans++":
            centroids = []
            # Wähle ersten Zentroid zufällig
            centroids.append(x[np.random.randint(n_samples)])

            for _ in range(1, self.k):
                distances = np.min(self._compute_distance(x, np.array(centroids)), axis=1)
                probs = distances / distances.sum()
                next_centroid = x[np.random.choice(n_samples, p=probs)]
                centroids.append(next_centroid)
            return np.array(centroids)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")


    def _compute_distance(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        if self.distance_metric == "euclidean":
            return np.linalg.norm(x[:, np.newaxis] - centroids, axis=2)
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(x[:, np.newaxis] - centroids), axis=2)
        elif self.distance_metric == "dtw":
            return self._dtw(x, centroids)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")


    def _dtw(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        n_samples = x.shape[0]
        k = centroids.shape[0]
        distances = np.zeros((n_samples, k))

        for i in tqdm(range(n_samples), desc="DTW Distance"):
            for j in range(k):
                xi = x[i]
                cj = centroids[j]
                # Falls Daten mehrdimensional sind, mitteln über Achsen
                if xi.ndim == 2:
                    d = np.mean([dtw.distance(xi[:, d], cj[:, d]) for d in range(xi.shape[1])])
                else:
                    d = dtw.distance(xi, cj)
                distances[i, j] = d

        return distances
