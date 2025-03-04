from .knn_faiss import FastNearestNeighbors
import numpy as np
import copy


class Matching(object):
    """
    Matching estimator for causal effects using faiss, a fast nearest neighbors library

    Args:

        estimand (str): The estimand to compute. Either "ATT" for average treatment effect on the treated or "ATE" for average treatment effect.
        k (int): Number of nearest neighbours to use for matching. Default is 1.
        bias_corr_mod (object): A model object to fit the bias correction term. Default is None.

    """

    def __init__(
        self,
        estimand: str,
        k=1,
        bias_corr_mod=None,
    ):
        self.estimand = estimand
        self.k = k
        self.bias_corr_mod = bias_corr_mod

    def fit(
        self,
        y: np.ndarray,
        w: np.ndarray,
        X: np.ndarray,
        **kwargs,
    ) -> tuple:
        """

        Args:

            y (np.ndarray): outcome vector.
            w (np.ndarray): treatment vector (binary)
            X (np.ndarray): covariate matrix.


        Returns:

            (float, float): (point_estimate, standard_error)


        Fit matching estimator for causal effects for specified estimand (ATE or ATT) and return point estimate and SE. Without bias correction, this is
        $$
        \hat{\\tau}^{ATT} = 1/N_1 \sum_{W_i = 1} \left( Y_i - 1/K \sum_{k=1}^K \hat{\mu}^{0, k}(X_i) \\right)
        $$

        where $\hat{\mu}^{0}(X_i)$ is average outcome of the $K$ matched units fit using `faiss`. With bias correction, the estimator is amended to

        $$
        \hat{\\tau}^{ATT} = 1/N_1 \sum_{W_i = 1} \left( Y_i - 1/K \sum_{k=1}^K \hat{\mu}^{0, k}(X_i) - [\hat{\phi}^{0}(X_i) - \hat{\phi}(X_{j(i)}) ] \\right )
        $$

        where the second term includes a bias adjustment using a regression function $\phi(x) = E[Y \mid X = x, w = 0]$ applied to the matching discrepancy (covariate discrepancy between a treated cell and its match); this collapses to zero when the covariate values match exactly. This estimator is doubly robust as the matching procedure implicitly estimates a density ratio. ATE analogues of the above involve matching symmetrically for both treatment and control units.

        References:

        - Abadie, A., and G. W. Imbens. 2011: “Bias-Corrected Matching Estimators for Average Treatment Effects,” Journal of business & economic statistics: a publication of the American Statistical Association, 29, 1–11.
        - Ding, Peng. A first course in causal inference. CRC Press, 2024. Chapter 15.


        """

        self._check_inputs(y, w, X)

        if self.estimand == "ATT":
            n, n1 = len(w), w.sum()
            mod = FastNearestNeighbors(**kwargs)
            # create index on untreated obs
            treat_nn_mod = mod.fit(X[w == 0, :])
            # find neighbours for treated obs
            neighbours = treat_nn_mod.kneighbors(X[w == 1, :], self.k)[1]
            # average outcomes for neighbours to construct y^0
            y0hat = y[w == 0][neighbours].mean()
            if self.bias_corr_mod:  # truthy only when model object attached
                # outcome model
                muhat = self.bias_corr_mod.fit(X[w == 0, :], y[w == 0])
                # bias correction term is μ^0(x_i) - μ^0(x_j) for each
                bias_corr_term = (
                    muhat.predict(X[w == 1, :]).mean()
                    - muhat.predict(X[w == 0, :][neighbours.flatten(), :]).mean()
                )
                point_est = y[w == 1].mean() - y0hat - bias_corr_term
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
            # no SE without bias correction
            return y[w == 1].mean() - y0hat, np.nan
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
                mod1 = copy.deepcopy(self.bias_corr_mod)
                mod1.fit(X[w == 1], y[w == 1])
                mu1_i = mod1.predict(X)
                mod0 = copy.deepcopy(self.bias_corr_mod)
                mod0.fit(X[w == 0], y[w == 0])
                mu0_i = mod0.predict(X)
                ############################################################
                # SE calculation
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
            # for no bias-correction, SE is null
            return tauhat_m, np.nan

    def _check_inputs(self, y: np.ndarray, w: np.ndarray, X: np.ndarray) -> None:
        if len(y) != len(w) or len(w) != X.shape[0]:
            raise ValueError("Dimension mismatch among y, w, and X.")
        if not np.isin(w, [0, 1]).all():
            raise ValueError("Treatment vector w must contain only 0s and 1s.")
        if self.k > np.sum(w == 0) and self.estimand == "ATT":
            raise ValueError("Not enough control units to match with k neighbors.")
        if self.k > np.sum(w == 1) and self.estimand == "ATE":
            # Also relevant in 'ATE' for matching controls to treated, etc.
            raise ValueError("Not enough treated units to match with k neighbors.")
