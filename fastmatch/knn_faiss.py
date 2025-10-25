import numpy as np
import faiss
import json
import os
from datetime import datetime, timezone


class FastNearestNeighbors:
    def __init__(
        self,
        metric: str = "mahalanobis",
        index_type: str = "flatl2",
        n_cells: int = 100,
        n_probes: int = 10,
    ):
        """
        NearestNeighbors object using faiss for speed. Identical API to sklearn but faster.
        Utilizes the `faiss` library https://github.com/facebookresearch/faiss . Tested with
        version 1.7.0. If `faiss-gpu` is installed from pypi, GPU acceleration
        will be used if available.

        Args:
            metric (str) :  Distance metric for finding nearest neighbors
                (default: "mahalanobis")
            index_type (str) : Index type within faiss to use
                (supported: "flatl2" and "ivfflat")
            n_cells (int) : Number of voronoi cells (only used for "ivfflat",
                default: 100)
            n_probes (int) : Number of voronoi cells to search in
                (only used for "ivfflat", default: 10)
        Attributes (after running `fit`):
            index_ : the faiss index fit from the data. For details about
            faiss indices, see the faiss documentation at
            https://github.com/facebookresearch/faiss/wiki/Faiss-indexes .
        """
        self.metric = metric
        self.n_cells = n_cells
        self.n_probes = n_probes
        self.index_type = index_type

    def fit(
        self,
        X: np.ndarray,
    ):
        """Create faiss index and train with data.

        Args:
            X (np.array): Array of N samples of shape (NxM)

        Returns:
            self: Fitted object
        """
        cov = np.cov(X, rowvar=False)
        # np.cov returns scalar for single covariate, reshape to (1,1)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        self._setattr("VI", cov)
        X = self._transform_covariates(X)
        if self.index_type == "flatl2":
            self.index_ = faiss.IndexFlatL2(X.shape[1])
            self.index_.add(X)
        elif self.index_type == "ivfflat":
            quantizer = faiss.IndexFlatL2(X.shape[1])
            n_cells = max(1, min(self.n_cells, X.shape[0] // 200))
            n_probes = min(self.n_probes, n_cells)
            self.index_ = faiss.IndexIVFFlat(quantizer, X.shape[1], n_cells)
            self.index_.train(X)
            self.index_.nprobe = n_probes
            self.index_.add(X)
        else:
            raise NotImplementedError(
                "Index type {} not implemented. Please select"
                'one of ["flatl2", "ivfflat"]'.format(self.index_type)
            )
        return self

    def kneighbors(
        self,
        X: np.ndarray,
        n_neighbors: int = 1,
    ):
        """Find the k nearest neighbors of each sample in X

        Args:
            X (np.array):  Array of shape (N,M) of samples to search
                for neighbors of. M must be the same as the fit data.
            n_neighbors (int, optional): Number of neighbors to find.
                Defaults to 1.

        Returns:
            (distances, indices): Two np.array objects of shape (N,n_neighbors)
                containing the distances and indices of the closest neighbors.
        """
        X = self._transform_covariates(X)
        distances, indices = self.index_.search(X, n_neighbors)
        # faiss returns euclidean distance squared
        return np.sqrt(distances), indices

    def _transform_covariates(self, X):
        if self.metric == "mahalanobis":
            if not hasattr(self, "VI"):
                raise AttributeError("Set inverse covariance VI first.")
            X = np.dot(X, self.VI.T)
        return np.ascontiguousarray(X).astype("float32")

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if parameter == "metric_params":
                self.set_params(**value)
            else:
                self._setattr(parameter, value)
        return self

    def get_params(self, deep=True):
        # `deep` plays no role because there are no sublearners
        params_to_return = ["metric", "n_cells", "n_probes", "index_type"]
        return {i: self.__getattribute__(i) for i in params_to_return}

    def _setattr(self, parameter, value):
        # based on faiss docs https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
        if parameter == "VI":
            value = np.linalg.inv(value)
            chol = np.linalg.cholesky(value)
            cholvi = np.linalg.inv(chol)
            value = cholvi
        setattr(self, parameter, value)

    def save(self, filepath: str):
        """Save FAISS index and metadata to disk for persistence across sessions.

        This saves two files:
        - {filepath}.index : Binary FAISS index
        - {filepath}.meta : JSON metadata with hyperparameters and auxiliary data

        Args:
            filepath (str): Base path for saving files (without extension)

        Example:
            model.save("./indices/control_pop")
            # Creates: control_pop.index and control_pop.meta
        """
        if not hasattr(self, "index_"):
            raise ValueError("Model must be fitted before saving. Call fit() first.")

        index_path = f"{filepath}.index"
        meta_path = f"{filepath}.meta"

        faiss.write_index(self.index_, index_path)

        metadata = {
            "backend": "faiss",
            "index_type": self.index_type,
            "metric": self.metric,
            "n_samples": self.index_.ntotal,
            "n_features": self.index_.d,
            "hyperparameters": {
                "n_cells": self.n_cells,
                "n_probes": self.n_probes,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "0.2.0",
        }

        if hasattr(self, "VI"):
            metadata["covariance_matrix"] = self.VI.tolist()

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load FAISS index and metadata from disk.

        Args:
            filepath (str): Base path to load from (without extension)

        Returns:
            FastNearestNeighbors: Loaded model ready for kneighbors() queries

        Example:
            model = FastNearestNeighbors.load("./indices/control_pop")
            distances, indices = model.kneighbors(X_query, k=5)
        """
        index_path = f"{filepath}.index"
        meta_path = f"{filepath}.meta"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        instance = cls(
            metric=metadata["metric"],
            index_type=metadata["index_type"],
            n_cells=metadata["hyperparameters"]["n_cells"],
            n_probes=metadata["hyperparameters"]["n_probes"],
        )

        instance.index_ = faiss.read_index(index_path)

        if "covariance_matrix" in metadata:
            instance.VI = np.array(metadata["covariance_matrix"])

        return instance
