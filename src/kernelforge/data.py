"""Inspectable preprocessing, frozen dataset identity and split indices."""
import hashlib
import importlib.metadata
from pathlib import Path
import json
import numpy as np
from .kernels import matrix


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(a.dtype.str.encode()); h.update(a.tobytes())
    return h.hexdigest()


class Standardizer:
    def fit(self, X):
        X = matrix(X)
        with np.errstate(over="raise", invalid="raise"):
            try:
                self.mean_ = X.mean(axis=0)
                self.scale_ = X.std(axis=0, ddof=0)
            except FloatingPointError as exc:
                raise ValueError("standardization overflow") from exc
        self.scale_[self.scale_ == 0] = 1
        if not np.isfinite(self.mean_).all() or not np.isfinite(self.scale_).all():
            raise ValueError("nonfinite standardizer")
        return self

    def transform(self, X):
        if not hasattr(self, "mean_"):
            raise RuntimeError("standardizer not fitted")
        X = matrix(X, allow_empty=True)
        if X.shape[1] != len(self.mean_):
            raise ValueError("standardizer feature mismatch")
        with np.errstate(over="raise", invalid="raise"):
            try:
                return (X - self.mean_) / self.scale_
            except FloatingPointError as exc:
                raise ValueError("standardization overflow") from exc


def load_wdbc():
    from sklearn.datasets import load_breast_cancer
    ds = load_breast_cancer()
    X = np.asarray(ds.data, dtype=np.float64)
    y = np.where(ds.target == 0, 1, -1).astype(np.int64)
    info = {"name": "WDBC_sklearn_packaged", "shape": list(X.shape),
            "source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
            "loader": "sklearn.datasets.load_breast_cancer",
            "sklearn_version": importlib.metadata.version("scikit-learn"),
            "array_sha256": array_hash(X, y), "positive_label": "malignant (+1)",
            "negative_label": "benign (-1)", "feature_names": list(ds.feature_names),
            "scope": "packaged public numerical features, not raw image feature reproduction"}
    return X, y, info


def split_indices(y, seed):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y), dtype=np.int64)
    trainval, test = train_test_split(idx, test_size=.2, random_state=seed, stratify=y)
    train, val = train_test_split(trainval, test_size=.25, random_state=seed, stratify=y[trainval])
    return {"train": train, "validation": val, "test": test}


def moons(n=240, noise=.12, seed=2026):
    """Two half-circles; explicit generator rather than a hidden data fixture."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    t0, t1 = rng.uniform(0,np.pi,n0), rng.uniform(0,np.pi,n-n0)
    X = np.vstack([np.c_[np.cos(t0),np.sin(t0)], np.c_[1-np.cos(t1), .5-np.sin(t1)]])
    X += rng.normal(0, noise, X.shape)
    y = np.r_[-np.ones(n0,dtype=int),np.ones(n-n0,dtype=int)]
    order = rng.permutation(n)
    return X[order], y[order]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_bundle(path, model, scaler):
    """One inference artifact includes the training-fitted scaler; no pickle."""
    model.save(path, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)


def load_bundle(path):
    from .svm import KernelSVM
    model = KernelSVM.load(path)
    with np.load(path, allow_pickle=False) as z:
        mean, scale = np.array(z["scaler_mean"], dtype=float), np.array(z["scaler_scale"], dtype=float)
    if mean.shape != (model.n_features_in_,) or scale.shape != mean.shape:
        raise ValueError("invalid scaler shape")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("invalid scaler values")
    scaler = Standardizer()
    scaler.mean_, scaler.scale_ = mean, scale
    return model, scaler
