from .knn_faiss import FastNearestNeighbors
from .knn_voyager import VoyagerNearestNeighbors
from .knn_scikit import ScikitNearestNeighbors
import numpy as np
import copy  # used only for bias correction models
import pickle
import os
from datetime import datetime, timezone


class Matching(object):
    """
    Matching estimator for causal effects using faiss, a fast nearest neighbors library

    Args:

        estimand (str): The estimand to compute. Either "ATT" for average treatment effect on the treated or "ATE" for average treatment effect.
        k (int): Number of nearest neighbours to use for matching. Default is 1.
        bias_corr_mod (object): A model object to fit the bias correction term. Default is None.
        backend (str): The backend to use for nearest neighbor search. One of "faiss", "voyager", or "scikit". Default is "faiss".

    """

    def __init__(
        self,
        estimand: str,
        k=1,
        bias_corr_mod=None,
        backend="faiss",
    ):
        self.estimand = estimand
        self.k = k
        self.bias_corr_mod = bias_corr_mod
        self.backend = backend

    def fit(
        self,
        y: np.ndarray,
        w: np.ndarray,
        X: np.ndarray,
        **kwargs,
    ) -> tuple:
        r"""Fit method for matching estimator.

        Args:

            y (np.ndarray): outcome vector.
            w (np.ndarray): treatment vector (binary)
            X (np.ndarray): covariate matrix.


        Returns:

            (float, float): (point_estimate, standard_error)


        Fit matching estimator for causal effects for specified estimand (ATE or ATT) and return point estimate and SE. Without bias correction, this is
        $
        \hat{\tau}^{ATT} = 1/N_1 \sum_{W_i = 1} \left( Y_i - 1/K \sum_{k=1}^K \hat{\mu}^{0, k}(X_i) \right)
        $

        where $\hat{\mu}^{0}(X_i)$ is average outcome of the $K$ matched units fit using `faiss` or `voyager`. With bias correction, the estimator is amended to

        $
        \hat{\tau}^{ATT} = 1/N_1 \sum_{W_i = 1} \left( Y_i - 1/K \sum_{k=1}^K \hat{\mu}^{0, k}(X_i) - [\hat{\phi}^{0}(X_i) - \hat{\phi}(X_{j(i)}) ] \right )
        $

        where the second term includes a bias adjustment using a regression function $\phi(x) = E[Y \mid X = x, w = 0]$ applied to the matching discrepancy (covariate discrepancy between a treated cell and its match); this collapses to zero when the covariate values match exactly. This estimator is doubly robust as the matching procedure implicitly estimates a density ratio. ATE analogues of the above involve matching symmetrically for both treatment and control units.

        References:

        - Abadie, A., and G. W. Imbens. 2011: “Bias-Corrected Matching Estimators for Average Treatment Effects,” Journal of business & economic statistics: a publication of the American Statistical Association, 29, 1–11.
        - Ding, Peng. A first course in causal inference. CRC Press, 2024. Chapter 15.

        """

        self._check_inputs(y, w, X)

        # Ensure X is 2D (handle single covariate case)
        if X.ndim == 1:
            X = X[:, np.newaxis]

        if self.backend == "faiss":
            knn_backend = FastNearestNeighbors
        elif self.backend == "voyager":
            knn_backend = VoyagerNearestNeighbors
        elif self.backend == "scikit":
            knn_backend = ScikitNearestNeighbors
        else:
            raise ValueError(
                "Backend not supported. Choose 'faiss', 'voyager', or 'scikit'."
            )

        if self.estimand == "ATT":
            n, n1 = len(w), w.sum()
            mod = knn_backend(**kwargs)
            # create index on untreated obs
            treat_nn_mod = mod.fit(X[w == 0, :])
            self._fitted_control_mod = treat_nn_mod
            # find neighbours for treated obs
            neighbours = treat_nn_mod.kneighbors(X[w == 1, :], self.k)[1]
            # average outcomes for neighbours to construct y^0
            y0hat = y[w == 0][neighbours].mean()
            if self.bias_corr_mod:  # truthy only when model object attached
                # outcome model
                muhat = self.bias_corr_mod.fit(X[w == 0, :], y[w == 0])
                self._fitted_bias_corr_control = muhat
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
            # find control neighbours for treat
            treat_nn_mod = knn_backend(**kwargs).fit(X[w == 0, :])
            self._fitted_control_mod = treat_nn_mod
            treat_neighbours = treat_nn_mod.kneighbors(X[w == 1, :], self.k)[1]
            # find treat neighbours for control
            ctrl_nn_mod = knn_backend(**kwargs).fit(X[w == 1, :])
            self._fitted_treat_mod = ctrl_nn_mod
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
                self._fitted_bias_corr_treat = mod1
                mu1_i = mod1.predict(X)
                mod0 = copy.deepcopy(self.bias_corr_mod)
                mod0.fit(X[w == 0], y[w == 0])
                self._fitted_bias_corr_control = mod0
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

    def save_indices(
        self,
        base_path: str,
        y: np.ndarray = None,
        w: np.ndarray = None,
        X: np.ndarray = None,
    ):
        """Save fitted indices and bias correction model to disk.

        Saves index files for control (and treatment if ATE) groups, plus bias correction model
        if fitted. Files are named with user prefix, timestamp, and backend type.

        Args:
            base_path (str): Directory and prefix for saved files (e.g., "./indices/my_study")
            y (np.ndarray, optional): Outcome vector (stored for reference)
            w (np.ndarray, optional): Treatment vector (stored for reference)
            X (np.ndarray, optional): Covariate matrix (stored for reference)

        Example:
            m = Matching("ATT", k=5, backend="faiss", bias_corr_mod=LinearRegression())
            m.fit(y, w, X)
            m.save_indices("./indices/lalonde_att")
            # Creates files like:
            #   lalonde_att_control_faiss_20250125_103045.index
            #   lalonde_att_control_faiss_20250125_103045.meta
            #   lalonde_att_biasmod_20250125_103045.pkl (if bias_corr_mod exists)
        """
        if not hasattr(self, "_fitted_control_mod"):
            raise ValueError("No fitted indices to save. Call fit() first.")

        os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        control_path = f"{base_path}_control_{self.backend}_{timestamp}"
        self._fitted_control_mod.save(control_path)

        if self.estimand == "ATE" and hasattr(self, "_fitted_treat_mod"):
            treat_path = f"{base_path}_treat_{self.backend}_{timestamp}"
            self._fitted_treat_mod.save(treat_path)

        if hasattr(self, "_fitted_bias_corr_control"):
            biasmod_path = f"{base_path}_biasmod_{timestamp}.pkl"
            with open(biasmod_path, "wb") as f:
                models = {
                    "control": self._fitted_bias_corr_control,
                }
                if hasattr(self, "_fitted_bias_corr_treat"):
                    models["treat"] = self._fitted_bias_corr_treat
                pickle.dump(models, f)

    def load_indices(
        self,
        control_index_path: str,
        treat_index_path: str = None,
        bias_model_path: str = None,
    ):
        """Load pre-fitted indices from disk.

        Args:
            control_index_path (str): Path to control group index (without .index extension)
            treat_index_path (str, optional): Path to treatment group index (for ATE only)
            bias_model_path (str, optional): Path to pickled bias correction model

        Example:
            m = Matching("ATT", k=5, backend="faiss")
            m.load_indices("./indices/lalonde_att_control_faiss_20250125_103045")
            # Now ready to estimate with new treated units without re-indexing controls
        """
        if self.backend == "faiss":
            self._fitted_control_mod = FastNearestNeighbors.load(control_index_path)
        elif self.backend == "voyager":
            self._fitted_control_mod = VoyagerNearestNeighbors.load(control_index_path)
        elif self.backend == "scikit":
            raise NotImplementedError("Scikit backend does not support persistence.")
        else:
            raise ValueError("Backend not supported.")

        if treat_index_path:
            if self.backend == "faiss":
                self._fitted_treat_mod = FastNearestNeighbors.load(treat_index_path)
            elif self.backend == "voyager":
                self._fitted_treat_mod = VoyagerNearestNeighbors.load(treat_index_path)

        if bias_model_path:
            with open(bias_model_path, "rb") as f:
                models = pickle.load(f)
                self._fitted_bias_corr_control = models.get("control")
                self._fitted_bias_corr_treat = models.get("treat")

    def estimate_with_preloaded(
        self, y: np.ndarray, w: np.ndarray, X: np.ndarray
    ) -> tuple:
        """Estimate treatment effects using pre-loaded indices.

        Use this after load_indices() to compute estimates without rebuilding indices.
        This is useful for production scenarios or when re-analyzing with new treatment assignments.

        Args:
            y (np.ndarray): Outcome vector
            w (np.ndarray): Treatment vector (binary)
            X (np.ndarray): Covariate matrix

        Returns:
            (float, float): (point_estimate, standard_error)

        Example:
            m = Matching("ATT", k=5, backend="faiss", bias_corr_mod=LinearRegression())
            m.load_indices("./indices/control_pop_faiss_20250125")
            estimate, se = m.estimate_with_preloaded(y_new, w_new, X_new)
        """
        if not hasattr(self, "_fitted_control_mod"):
            raise ValueError("No pre-loaded indices found. Call load_indices() first.")

        self._check_inputs(y, w, X)

        # Ensure X is 2D (handle single covariate case)
        if X.ndim == 1:
            X = X[:, np.newaxis]

        if self.estimand == "ATT":
            n, n1 = len(w), w.sum()
            neighbours = self._fitted_control_mod.kneighbors(X[w == 1, :], self.k)[1]
            y0hat = y[w == 0][neighbours].mean()

            if (
                hasattr(self, "_fitted_bias_corr_control")
                and self._fitted_bias_corr_control
            ):
                muhat = self._fitted_bias_corr_control
                bias_corr_term = (
                    muhat.predict(X[w == 1, :]).mean()
                    - muhat.predict(X[w == 0, :][neighbours.flatten(), :]).mean()
                )
                point_est = y[w == 1].mean() - y0hat - bias_corr_term
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

            return y[w == 1].mean() - y0hat, np.nan

        elif self.estimand == "ATE":
            if not hasattr(self, "_fitted_treat_mod"):
                raise ValueError(
                    "ATE requires both control and treatment indices. Load both."
                )

            n = len(w)
            treat_neighbours = self._fitted_control_mod.kneighbors(
                X[w == 1, :], self.k
            )[1]
            ctrl_neighbours = self._fitted_treat_mod.kneighbors(X[w == 0, :], self.k)[1]

            y1hat, y0hat = y.copy(), y.copy()
            np.put(y1hat, np.argwhere(w == 0), y[w == 1][ctrl_neighbours].mean(axis=1))
            np.put(y0hat, np.argwhere(w == 1), y[w == 0][treat_neighbours].mean(axis=1))
            tauhat_m = y1hat.mean() - y0hat.mean()

            if (
                hasattr(self, "_fitted_bias_corr_control")
                and self._fitted_bias_corr_control
            ):
                mod1 = self._fitted_bias_corr_treat
                mod0 = self._fitted_bias_corr_control
                mu1_i = mod1.predict(X)
                mu0_i = mod0.predict(X)

                K_i = np.zeros(n)
                np.put(
                    K_i,
                    np.argwhere(w == 1),
                    np.bincount(ctrl_neighbours.flatten(), minlength=n)[w == 1],
                )
                np.put(
                    K_i,
                    np.argwhere(w == 0),
                    np.bincount(treat_neighbours.flatten(), minlength=n)[w == 0],
                )
                R_i = np.where(w == 1, y - mu1_i, y - mu0_i)
                psi_i = tauhat_m + (2 * w - 1) * (1 + K_i / self.k) * R_i
                return psi_i.mean(), np.sqrt(1 / n * psi_i.var())

            return tauhat_m, np.nan
