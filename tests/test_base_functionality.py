"""
Comprehensive tests for base functionality across all backends and estimands.
Tests all combinations of {scikit, faiss, voyager} X {ATT, ATE} with and without bias correction.
Run with: pytest tests/test_base_functionality.py -v
"""

import numpy as np
import pytest
from fastmatch import Matching
from sklearn.linear_model import LinearRegression


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 5)
    w = np.random.binomial(1, 0.3, n)
    y = X[:, 0] + w * 1.5 + np.random.randn(n) * 0.5
    return y, w, X


@pytest.fixture
def balanced_data():
    """Generate balanced data for ATE testing."""
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 5)
    w = np.random.binomial(1, 0.5, n)
    y = X[:, 0] + w * 2.0 + np.random.randn(n) * 0.5
    return y, w, X


class TestATTEstimation:
    """Test ATT estimation across all backends."""

    def test_faiss_att_no_bias_correction(self, sample_data):
        """Test FAISS backend with ATT estimand, no bias correction."""
        y, w, X = sample_data
        m = Matching("ATT", k=5, backend="faiss")
        est, se = m.fit(y, w, X)

        assert isinstance(est, (float, np.floating))
        assert np.isnan(se)  # No SE without bias correction
        assert not np.isnan(est)

    def test_faiss_att_with_bias_correction(self, sample_data):
        """Test FAISS backend with ATT estimand and bias correction."""
        y, w, X = sample_data
        m = Matching("ATT", k=5, backend="faiss", bias_corr_mod=LinearRegression())
        est, se = m.fit(y, w, X)

        assert isinstance(est, (float, np.floating))
        assert isinstance(se, (float, np.floating))
        assert not np.isnan(est)
        assert not np.isnan(se)
        assert se > 0

    def test_voyager_att_no_bias_correction(self, sample_data):
        """Test Voyager backend with ATT estimand, no bias correction."""
        y, w, X = sample_data
        X_float32 = X.astype(np.float32)
        m = Matching("ATT", k=5, backend="voyager")
        est, se = m.fit(y, w, X_float32)

        assert isinstance(est, (float, np.floating))
        assert np.isnan(se)
        assert not np.isnan(est)

    def test_voyager_att_with_bias_correction(self, sample_data):
        """Test Voyager backend with ATT estimand and bias correction."""
        y, w, X = sample_data
        X_float32 = X.astype(np.float32)
        m = Matching("ATT", k=5, backend="voyager", bias_corr_mod=LinearRegression())
        est, se = m.fit(y, w, X_float32)

        assert isinstance(est, (float, np.floating))
        assert isinstance(se, (float, np.floating))
        assert not np.isnan(est)
        assert not np.isnan(se)
        assert se > 0

    def test_scikit_att_no_bias_correction(self, sample_data):
        """Test Scikit backend with ATT estimand, no bias correction."""
        y, w, X = sample_data
        m = Matching("ATT", k=5, backend="scikit")
        est, se = m.fit(y, w, X)

        assert isinstance(est, (float, np.floating))
        assert np.isnan(se)
        assert not np.isnan(est)

    def test_scikit_att_with_bias_correction(self, sample_data):
        """Test Scikit backend with ATT estimand and bias correction."""
        y, w, X = sample_data
        m = Matching("ATT", k=5, backend="scikit", bias_corr_mod=LinearRegression())
        est, se = m.fit(y, w, X)

        assert isinstance(est, (float, np.floating))
        assert isinstance(se, (float, np.floating))
        assert not np.isnan(est)
        assert not np.isnan(se)
        assert se > 0


class TestATEEstimation:
    """Test ATE estimation across all backends."""

    def test_faiss_ate_no_bias_correction(self, balanced_data):
        """Test FAISS backend with ATE estimand, no bias correction."""
        y, w, X = balanced_data
        m = Matching("ATE", k=5, backend="faiss")
        est, se = m.fit(y, w, X)

        assert isinstance(est, (float, np.floating))
        assert np.isnan(se)
        assert not np.isnan(est)

    def test_faiss_ate_with_bias_correction(self, balanced_data):
        """Test FAISS backend with ATE estimand and bias correction."""
        y, w, X = balanced_data
        m = Matching("ATE", k=5, backend="faiss", bias_corr_mod=LinearRegression())
        est, se = m.fit(y, w, X)

        assert isinstance(est, (float, np.floating))
        assert isinstance(se, (float, np.floating))
        assert not np.isnan(est)
        assert not np.isnan(se)
        assert se > 0

    def test_voyager_ate_no_bias_correction(self, balanced_data):
        """Test Voyager backend with ATE estimand, no bias correction."""
        y, w, X = balanced_data
        X_float32 = X.astype(np.float32)
        m = Matching("ATE", k=5, backend="voyager")
        est, se = m.fit(y, w, X_float32)

        assert isinstance(est, (float, np.floating))
        assert np.isnan(se)
        assert not np.isnan(est)

    def test_voyager_ate_with_bias_correction(self, balanced_data):
        """Test Voyager backend with ATE estimand and bias correction."""
        y, w, X = balanced_data
        X_float32 = X.astype(np.float32)
        m = Matching("ATE", k=5, backend="voyager", bias_corr_mod=LinearRegression())
        est, se = m.fit(y, w, X_float32)

        assert isinstance(est, (float, np.floating))
        assert isinstance(se, (float, np.floating))
        assert not np.isnan(est)
        assert not np.isnan(se)
        assert se > 0

    def test_scikit_ate_no_bias_correction(self, balanced_data):
        """Test Scikit backend with ATE estimand, no bias correction."""
        y, w, X = balanced_data
        m = Matching("ATE", k=5, backend="scikit")
        est, se = m.fit(y, w, X)

        assert isinstance(est, (float, np.floating))
        assert np.isnan(se)
        assert not np.isnan(est)

    def test_scikit_ate_with_bias_correction(self, balanced_data):
        """Test Scikit backend with ATE estimand and bias correction."""
        y, w, X = balanced_data
        m = Matching("ATE", k=5, backend="scikit", bias_corr_mod=LinearRegression())
        est, se = m.fit(y, w, X)

        assert isinstance(est, (float, np.floating))
        assert isinstance(se, (float, np.floating))
        assert not np.isnan(est)
        assert not np.isnan(se)
        assert se > 0


class TestBackendConsistency:
    """Test that different backends produce similar results."""

    def test_att_backends_similar(self, sample_data):
        """Test that ATT estimates are similar across backends."""
        y, w, X = sample_data
        X_float32 = X.astype(np.float32)

        m_faiss = Matching("ATT", k=5, backend="faiss", bias_corr_mod=LinearRegression())
        est_faiss, _ = m_faiss.fit(y, w, X)

        m_voyager = Matching("ATT", k=5, backend="voyager", bias_corr_mod=LinearRegression())
        est_voyager, _ = m_voyager.fit(y, w, X_float32)

        m_scikit = Matching("ATT", k=5, backend="scikit", bias_corr_mod=LinearRegression())
        est_scikit, _ = m_scikit.fit(y, w, X)

        # Estimates should be reasonably close (within 50% relative difference)
        assert abs(est_faiss - est_voyager) / abs(est_faiss) < 0.5
        assert abs(est_faiss - est_scikit) / abs(est_faiss) < 0.5

    def test_ate_backends_similar(self, balanced_data):
        """Test that ATE estimates are similar across all backends."""
        y, w, X = balanced_data
        X_float32 = X.astype(np.float32)

        m_faiss = Matching("ATE", k=5, backend="faiss", bias_corr_mod=LinearRegression())
        est_faiss, _ = m_faiss.fit(y, w, X)

        m_voyager = Matching("ATE", k=5, backend="voyager", bias_corr_mod=LinearRegression())
        est_voyager, _ = m_voyager.fit(y, w, X_float32)

        m_scikit = Matching("ATE", k=5, backend="scikit", bias_corr_mod=LinearRegression())
        est_scikit, _ = m_scikit.fit(y, w, X)

        # Estimates should be reasonably close (within 50% relative difference)
        assert abs(est_faiss - est_voyager) / abs(est_faiss) < 0.5
        assert abs(est_faiss - est_scikit) / abs(est_faiss) < 0.5


class TestInputValidation:
    """Test input validation and error handling."""

    def test_dimension_mismatch_y_w(self, sample_data):
        """Test error raised for dimension mismatch between y and w."""
        y, w, X = sample_data
        m = Matching("ATT", k=5, backend="faiss")
        with pytest.raises(ValueError, match="Dimension mismatch"):
            m.fit(y[:-10], w, X)

    def test_dimension_mismatch_w_X(self, sample_data):
        """Test error raised for dimension mismatch between w and X."""
        y, w, X = sample_data
        m = Matching("ATT", k=5, backend="faiss")
        with pytest.raises(ValueError, match="Dimension mismatch"):
            m.fit(y, w[:-10], X)

    def test_invalid_treatment_values(self, sample_data):
        """Test error raised for non-binary treatment values."""
        y, w, X = sample_data
        w_invalid = w.copy()
        w_invalid[0] = 2
        m = Matching("ATT", k=5, backend="faiss")
        with pytest.raises(ValueError, match="only 0s and 1s"):
            m.fit(y, w_invalid, X)

    def test_insufficient_control_units(self, sample_data):
        """Test error raised when k exceeds control units."""
        y, w, X = sample_data
        n_control = (w == 0).sum()
        m = Matching("ATT", k=n_control + 1, backend="faiss")
        with pytest.raises(ValueError, match="Not enough control units"):
            m.fit(y, w, X)

    def test_invalid_backend(self, sample_data):
        """Test error raised for invalid backend."""
        y, w, X = sample_data
        m = Matching("ATT", k=5, backend="invalid_backend")
        with pytest.raises(ValueError, match="Backend not supported"):
            m.fit(y, w, X)


class TestDifferentKValues:
    """Test estimation with different k values."""

    def test_k1_vs_k5_faiss(self, sample_data):
        """Test that different k values produce different estimates."""
        y, w, X = sample_data

        m1 = Matching("ATT", k=1, backend="faiss", bias_corr_mod=LinearRegression())
        est1, _ = m1.fit(y, w, X)

        m5 = Matching("ATT", k=5, backend="faiss", bias_corr_mod=LinearRegression())
        est5, _ = m5.fit(y, w, X)

        # Different k should produce different estimates
        assert not np.isclose(est1, est5, atol=0.01)

    def test_increasing_k_reduces_variance(self, sample_data):
        """Test that increasing k generally reduces standard errors."""
        y, w, X = sample_data

        m1 = Matching("ATT", k=1, backend="faiss", bias_corr_mod=LinearRegression())
        _, se1 = m1.fit(y, w, X)

        m10 = Matching("ATT", k=10, backend="faiss", bias_corr_mod=LinearRegression())
        _, se10 = m10.fit(y, w, X)

        # Higher k should generally have lower SE (though not guaranteed)
        assert se10 < se1 * 2  # At least not dramatically worse


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_sample(self):
        """Test with very small sample size."""
        np.random.seed(42)
        n = 50
        X = np.random.randn(n, 3)
        w = np.random.binomial(1, 0.3, n)
        y = X[:, 0] + w * 1.0 + np.random.randn(n) * 0.3

        m = Matching("ATT", k=3, backend="faiss", bias_corr_mod=LinearRegression())
        est, se = m.fit(y, w, X)

        assert not np.isnan(est)
        assert not np.isnan(se)

    def test_single_covariate(self):
        """Test with single covariate (1D array automatically reshaped to 2D)."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n)  # 1D array
        w = np.random.binomial(1, 0.3, n)
        y = X + w * 1.5 + np.random.randn(n) * 0.5

        # FAISS backend should now handle 1D case by auto-reshaping
        m = Matching("ATT", k=5, backend="faiss", bias_corr_mod=LinearRegression())
        est, se = m.fit(y, w, X)

        assert not np.isnan(est)
        assert not np.isnan(se)

    def test_single_covariate_2d(self):
        """Test with single covariate as (n, 1) array."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 1)  # 2D array with single column
        w = np.random.binomial(1, 0.3, n)
        y = X[:, 0] + w * 1.5 + np.random.randn(n) * 0.5

        m = Matching("ATT", k=5, backend="faiss", bias_corr_mod=LinearRegression())
        est, se = m.fit(y, w, X)

        assert not np.isnan(est)
        assert not np.isnan(se)

    def test_many_covariates(self):
        """Test with many covariates."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 20)
        w = np.random.binomial(1, 0.3, n)
        y = X[:, 0] + w * 1.5 + np.random.randn(n) * 0.5

        m = Matching("ATT", k=5, backend="faiss", bias_corr_mod=LinearRegression())
        est, se = m.fit(y, w, X)

        assert not np.isnan(est)
        assert not np.isnan(se)
