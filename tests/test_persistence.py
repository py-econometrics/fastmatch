"""
Basic tests for index persistence functionality.
Run with: pytest tests/test_persistence.py
"""

import numpy as np
import tempfile
import os
from fastmatch import Matching
from sklearn.linear_model import LinearRegression


def test_faiss_backend_save_load():
    """Test FAISS backend save/load roundtrip."""

    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 5)
    w = np.random.binomial(1, 0.3, n)
    y = X[:, 0] + w * 1.5 + np.random.randn(n) * 0.5

    with tempfile.TemporaryDirectory() as tmpdir:
        m = Matching("ATT", k=5, backend="faiss", bias_corr_mod=LinearRegression())
        est1, se1 = m.fit(y, w, X)

        base_path = os.path.join(tmpdir, "test_index")
        m.save_indices(base_path)

        control_files = [f for f in os.listdir(tmpdir) if "control" in f]

        m2 = Matching("ATT", k=5, backend="faiss", bias_corr_mod=LinearRegression())
        control_index = os.path.join(
            tmpdir, [f for f in control_files if f.endswith(".index")][0][:-6]
        )
        bias_model = os.path.join(
            tmpdir, [f for f in os.listdir(tmpdir) if f.endswith(".pkl")][0]
        )
        m2.load_indices(control_index, bias_model_path=bias_model)

        est2, se2 = m2.estimate_with_preloaded(y, w, X)

        assert np.isclose(est1, est2, atol=1e-6), f"Estimates differ: {est1} vs {est2}"
        assert np.isclose(se1, se2, atol=1e-6), f"SEs differ: {se1} vs {se2}"

def test_voyager_backend_save_load():
    """Test Voyager backend save/load roundtrip."""
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 5).astype(np.float32)
    w = np.random.binomial(1, 0.3, n)
    y = X[:, 0] + w * 1.5 + np.random.randn(n) * 0.5

    with tempfile.TemporaryDirectory() as tmpdir:
        m = Matching("ATT", k=5, backend="voyager")
        est1, se1 = m.fit(y, w, X)

        base_path = os.path.join(tmpdir, "test_voyager")
        m.save_indices(base_path)

        control_files = [f for f in os.listdir(tmpdir) if "control" in f]
        control_index = os.path.join(
            tmpdir, [f for f in control_files if f.endswith(".index")][0][:-6]
        )

        m2 = Matching("ATT", k=5, backend="voyager")
        m2.load_indices(control_index)

        est2, se2 = m2.estimate_with_preloaded(y, w, X)

        assert np.isclose(est1, est2, atol=1e-6), f"Estimates differ: {est1} vs {est2}"

def test_ate_save_load():
    """Test ATE estimand with both control and treatment indices."""
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 5)
    w = np.random.binomial(1, 0.5, n)
    y = X[:, 0] + w * 1.5 + np.random.randn(n) * 0.5

    with tempfile.TemporaryDirectory() as tmpdir:
        m = Matching("ATE", k=5, backend="faiss", bias_corr_mod=LinearRegression())
        est1, se1 = m.fit(y, w, X)

        base_path = os.path.join(tmpdir, "test_ate")
        m.save_indices(base_path)

        control_files = [
            f for f in os.listdir(tmpdir) if "control" in f and f.endswith(".index")
        ]
        treat_files = [
            f for f in os.listdir(tmpdir) if "treat" in f and f.endswith(".index")
        ]
        bias_files = [f for f in os.listdir(tmpdir) if f.endswith(".pkl")]

        control_index = os.path.join(tmpdir, control_files[0][:-6])
        treat_index = os.path.join(tmpdir, treat_files[0][:-6])
        bias_model = os.path.join(tmpdir, bias_files[0])

        m2 = Matching("ATE", k=5, backend="faiss")
        m2.load_indices(
            control_index, treat_index_path=treat_index, bias_model_path=bias_model
        )

        est2, se2 = m2.estimate_with_preloaded(y, w, X)

        assert np.isclose(est1, est2, atol=1e-6), f"Estimates differ: {est1} vs {est2}"
        assert np.isclose(se1, se2, atol=1e-6), f"SEs differ: {se1} vs {se2}"


def test_voyager_ate_save_load():
    """Test Voyager ATE estimand persistence (verifies deepcopy fix)."""
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 5).astype(np.float32)
    w = np.random.binomial(1, 0.5, n)
    y = X[:, 0] + w * 1.5 + np.random.randn(n) * 0.5

    with tempfile.TemporaryDirectory() as tmpdir:
        m = Matching("ATE", k=5, backend="voyager", bias_corr_mod=LinearRegression())
        est1, se1 = m.fit(y, w, X)

        base_path = os.path.join(tmpdir, "test_voyager_ate")
        m.save_indices(base_path)

        control_files = [
            f for f in os.listdir(tmpdir) if "control" in f and f.endswith(".index")
        ]
        treat_files = [
            f for f in os.listdir(tmpdir) if "treat" in f and f.endswith(".index")
        ]
        bias_files = [f for f in os.listdir(tmpdir) if f.endswith(".pkl")]

        control_index = os.path.join(tmpdir, control_files[0][:-6])
        treat_index = os.path.join(tmpdir, treat_files[0][:-6])
        bias_model = os.path.join(tmpdir, bias_files[0])

        m2 = Matching("ATE", k=5, backend="voyager")
        m2.load_indices(
            control_index, treat_index_path=treat_index, bias_model_path=bias_model
        )

        est2, se2 = m2.estimate_with_preloaded(y, w, X)

        assert np.isclose(est1, est2, atol=1e-6), f"Estimates differ: {est1} vs {est2}"
        assert np.isclose(se1, se2, atol=1e-6), f"SEs differ: {se1} vs {se2}"
