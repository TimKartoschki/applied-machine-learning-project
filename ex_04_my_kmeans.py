from typing import Literal
import numpy as np
import logging
from tqdm import tqdm
from dtaidistance import dtw

DISTANCE_METRICS = Literal["euclidean", "manhattan", "dtw"]
INIT_METHOD = Literal["random", "kmeans++"]

logging.basicConfig(level=logging.INFO)

class MyKMeans:
    def __init__(self, k=3, max_iter=100, distance_metric: DISTANCE_METRICS = "euclidean", init: INIT_METHOD = "kmeans++", random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.distance_metric = distance_metric
        self.init = init
        self.centroids = None
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, x: np.ndarray):
        self.centroids = self._initialize_centroids(x)
        for _ in tqdm(range(self.max_iter), desc="KMeans Iterations"):
            labels = self.predict(x)

            new_centroids = []
            for i in range(self.k):
                cluster_points = x[labels == i]
                if len(cluster_points) == 0:
                    # Neu initialisieren, falls ein Cluster leer ist
                    new_centroids.append(x[np.random.choice(x.shape[0])])
                    continue

                if self.distance_metric == "dtw":
                    # DTW-Median als "zentralste" Serie
                    dists = np.array([[dtw.distance(s1, s2) for s2 in cluster_points] for s1 in cluster_points])
                    total_dists = dists.sum(axis=1)
                    new_centroids.append(cluster_points[np.argmin(total_dists)])
                else:
                    new_centroids.append(np.mean(cluster_points, axis=0))

            new_centroids = np.array(new_centroids)
            if np.allclose(self.centroids, new_centroids):
                break  # Konvergenz
            self.centroids = new_centroids

    def predict(self, x: np.ndarray):
        labels = []
        for xi in x:
            distances = [self._compute_distance(xi, centroid) for centroid in self.centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)

    def fit_predict(self, x: np.ndarray):
        self.fit(x)
        return self.predict(x)

    def _initialize_centroids(self, x: np.ndarray):
        n_samples = x.shape[0]
        if self.init == "random":
            indices = np.random.choice(n_samples, self.k, replace=False)
            return x[indices]

        # kmeans++ Initialisierung
        centroids = [x[np.random.choice(n_samples)]]
        for _ in range(1, self.k):
            distances = np.array([
                min([self._compute_distance(xi, c) for c in centroids]) for xi in x
            ])
            probs = distances ** 2
            probs = probs / probs.sum()
            next_centroid = x[np.random.choice(n_samples, p=probs)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def _compute_distance(self, a, b):
        if self.distance_metric == "euclidean":
            return np.linalg.norm(a - b)
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(a - b))
        elif self.distance_metric == "dtw":
            return dtw.distance(a, b)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
