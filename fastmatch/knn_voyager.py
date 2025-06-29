import numpy as np
import voyager


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
