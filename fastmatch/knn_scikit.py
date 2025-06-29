import numpy as np
from sklearn.neighbors import NearestNeighbors


class ScikitNearestNeighbors:
    """
    A wrapper for scikit-learn's NearestNeighbors to conform to the API
    used by the other backends in this package.
    """

    def __init__(self, **kwargs):
        """
        Initializes the scikit-learn NearestNeighbors model.
        Any keyword arguments are passed directly to the scikit-learn constructor.
        """
        self.model = NearestNeighbors(**kwargs)

    def fit(self, X: np.ndarray):
        """
        Fits the NearestNeighbors model to the provided data.

        Args:
            X (np.ndarray): Array of shape (N, M) to fit the model on.

        Returns:
            self: The fitted object.
        """
        self.model.fit(X)
        return self

    def kneighbors(self, X: np.ndarray, n_neighbors: int = 1):
        """
        Finds the k-nearest neighbors for the samples in X.

        Args:
            X (np.ndarray): Array of shape (N, M) to find neighbors for.
            n_neighbors (int): The number of neighbors to find.

        Returns:
            (distances, indices): Tuple of arrays containing the distances
                                  and indices of the nearest neighbors.
        """
        # n_neighbors in the method call overrides the one in the constructor
        distances, indices = self.model.kneighbors(X, n_neighbors=n_neighbors)
        return distances, indices
