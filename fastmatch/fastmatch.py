from .knn_faiss import FastNearestNeighbors
import numpy as np


class matching(object):
    """
    Matching estimator for causal effects using a fast nearest neighbors library
    """

    def __init__(self, estimand, k=1, bias_corr_mod=None):
        self.estimand = estimand
        self.k = k
        self.bias_corr_mod = bias_corr_mod

    def fit(self, y: np.ndarray, w: np.ndarray, X: np.ndarray, **kwargs) -> tuple:
        """
        Fit matching estimator for causal effects for specified estimand.

        Args:
            y (np.ndarray): outcome vector.
            w (np.ndarray): treatment vector (binary)
            X (np.ndarray): covariate matrix.

        Returns:
            tuple: point estimate and standard error
        """
        if self.estimand == "ATT":
            n, n1 = len(w), w.sum()
            mod = FastNearestNeighbors(**kwargs)
            # create index on untreated obs
            treat_nn_mod = mod.fit(X[w == 0, :])
            # find neighbours for treated obs
            neighbours = treat_nn_mod.kneighbors(X[w == 1, :], self.k)[1]
            # average outcomes for neighbours to construct y^0
            y0hat = y[w == 0][neighbours].mean()
            if self.bias_corr_mod:
                # outcome model
                muhat = self.bias_corr_mod.fit(X[w == 0, :], y[w == 0])
                # bias correction term is μ^0(x_i) - μ^0(x_j) for each
                bias_corr_term = (
                    muhat.predict(X[w == 1, :]).mean()
                    - muhat.predict(X[w == 0, :][neighbours.flatten(), :]).mean()
                )
                point_est = y[w == 1].mean() - y0hat.mean() - bias_corr_term
                # variance estimation - Otsu and Rai
                K = np.zeros(n)
                np.put(
                    K,
                    np.argwhere(w == 0),
                    np.bincount(neighbours.flatten(), minlength=n)[w == 0],
                )
                psi = w * (y - muhat.predict(X)) - (1 - w) * (K / self.k) * (
                    y - muhat.predict(X)
                )
                v_or = (1 / (n1**2)) * np.sum((psi - point_est * n1 / n) ** 2)
                return point_est, np.sqrt(v_or)
            return y[w == 1].mean() - y0hat.mean()
        elif self.estimand == "ATE":
            n = len(w)
            # nearest neigbours : Ki
            mod = FastNearestNeighbors(**kwargs)
            # find control neighbours for treat
            treat_nn_mod = mod.fit(X[w == 0, :])
            treat_neighbours = treat_nn_mod.kneighbors(X[w == 1, :], self.k)[1]
            # find treat neighbours for control
            ctrl_nn_mod = mod.fit(X[w == 1, :])
            ctrl_neighbours = ctrl_nn_mod.kneighbors(X[w == 0, :], self.k)[1]
            y1hat, y0hat = y.copy(), y.copy()
            np.put(
                y1hat,
                np.argwhere(w == 0),  # replace y1hat for control units
                y[w == 1][ctrl_neighbours].mean(axis=1),  # average over k neighbours
            )
            np.put(
                y0hat,
                np.argwhere(w == 1),
                y[w == 0][treat_neighbours].mean(axis=1),
            )
            tauhat_m = y1hat.mean() - y0hat.mean()
            if self.bias_corr_mod:
                # outcome model - μ̂1 (Xi) , μ̂0(Xi )
                muhat1 = self.bias_corr_mod.fit(X[w == 1], y[w == 1])
                mu1_i = muhat1.predict(X)
                muhat0 = self.bias_corr_mod.fit(X[w == 0], y[w == 0])
                mu0_i = muhat0.predict(X)
                K_i = np.zeros(n)
                # how many times treat units have been matched with control units
                np.put(
                    K_i,
                    np.argwhere(w == 1),
                    np.bincount(ctrl_neighbours.flatten(), minlength=n)[w == 1],
                )
                # how many times ctrl units have been matched with treat units
                np.put(
                    K_i,
                    np.argwhere(w == 0),
                    np.bincount(treat_neighbours.flatten(), minlength=n)[w == 0],
                )
                # residual
                R_i = np.where(w == 1, y - mu1_i, y - mu0_i)
                # ψ̂i = μ̂1 (Xi) − μ̂0(Xi ) + (2Wi − 1)(1 + Ki/M ){Yi − μ̂Wi (Xi)}
                psi_i = tauhat_m + (2 * w - 1) * (1 + K_i / self.k) * R_i
                return psi_i.mean(), np.sqrt(1 / n * psi_i.var())
            return tauhat_m
