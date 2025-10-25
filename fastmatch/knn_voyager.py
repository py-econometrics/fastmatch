import numpy as np
import voyager
import json
import os
from datetime import datetime


class VoyagerNearestNeighbors:
    def __init__(self, space: str = "euclidean", index_type: str = "hnsw", **kwargs):
        """
        NearestNeighbors object using voyager for speed. Identical API to sklearn but faster.
        Utilizes the `voyager` library https://github.com/spotify/voyager .

        Args:
            space (str) :  Distance metric for finding nearest neighbors
                (default: "euclidean")
            index_type (str) : Index type within voyager to use
                (supported: "hnsw")
        """
        if space.lower() == "euclidean":
            self.space = voyager.Space.Euclidean
        else:
            raise NotImplementedError(f"Space {space} not implemented.")
        self.space_str = space.lower()
        self.index_type = index_type

    def fit(self, X: np.ndarray):
        """Create voyager index and train with data.

        Args:
            X (np.array): Array of N samples of shape (NxM)

        Returns:
            self: Fitted object
        """
        self.index_ = voyager.Index(self.space, num_dimensions=X.shape[1])
        self.index_.add_items(X)
        return self

    def kneighbors(self, X: np.ndarray, n_neighbors: int = 1):
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
        indices, distances = self.index_.query(X, k=n_neighbors)
        return distances, indices.astype(np.int64)

    def save(self, filepath: str):
        """Save Voyager index and metadata to disk for persistence across sessions.

        This saves two files:
        - {filepath}.index : Binary Voyager index
        - {filepath}.meta : JSON metadata with hyperparameters

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

        self.index_.save(index_path)

        metadata = {
            "backend": "voyager",
            "index_type": self.index_type,
            "space": self.space_str,
            "n_samples": len(self.index_),
            "n_features": self.index_.num_dimensions,
            "created_at": datetime.utcnow().isoformat(),
            "version": "0.2.0",
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load Voyager index and metadata from disk.

        Args:
            filepath (str): Base path to load from (without extension)

        Returns:
            VoyagerNearestNeighbors: Loaded model ready for kneighbors() queries

        Example:
            model = VoyagerNearestNeighbors.load("./indices/control_pop")
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
            space=metadata["space"],
            index_type=metadata["index_type"],
        )

        space_enum = voyager.Space.Euclidean if metadata["space"] == "euclidean" else None
        instance.index_ = voyager.Index.load(
            index_path,
            space=space_enum,
            num_dimensions=metadata["n_features"]
        )

        return instance
