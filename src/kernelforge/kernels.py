"""Kernel rows and an explicit row-budget LRU; no third-party kernel routine."""
from collections import OrderedDict
import numbers
import numpy as np


def matrix(X, *, allow_empty=False):
    """Return finite float64 (n,d), rejecting ambiguous vectors and zero features."""
    a = np.asarray(X, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] == 0 or (not allow_empty and a.shape[0] == 0):
        raise ValueError("X must have shape (n,d), n>0 and d>0")
    if not np.isfinite(a).all():
        raise ValueError("X must contain only finite values")
    return a


def positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite positive real")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real")
    return value


def kernel_vector(X, x, kernel, gamma):
    """n-by-d against one d-vector: O(nd) time; RBF avoids norm subtraction."""
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            if kernel == "linear":
                out = X @ x
            elif kernel == "rbf":
                diff = X - x
                out = np.exp(-gamma * np.einsum("ij,ij->i", diff, diff))
            else:
                raise ValueError("kernel must be linear or rbf")
        except FloatingPointError as exc:
            raise ValueError("kernel arithmetic overflow; scale finite inputs") from exc
    if not np.isfinite(out).all():
        raise ValueError("nonfinite kernel result; scale inputs")
    return out


class KernelRows:
    """Own immutable data and cache immutable full kernel rows.

    cache_bytes counts retained row buffers only, not active references, temporary
    n*d differences, Python objects, input storage, or process peak memory.
    """
    def __init__(self, X, kernel="rbf", gamma=0.1, cache="lru", cache_mb=16.0):
        self.X = np.array(matrix(X), dtype=np.float64, copy=True, order="C")
        self.X.setflags(write=False)
        if kernel not in ("linear", "rbf") or cache not in ("none", "lru", "full"):
            raise ValueError("unsupported kernel/cache")
        self.kernel, self.cache = kernel, cache
        self.gamma = positive(gamma, "gamma")
        self.cache_mb = positive(cache_mb, "cache_mb")
        self.n = len(self.X)
        self.capacity = min(self.n, int(self.cache_mb * 2**20) // (8 * self.n))
        self._rows = OrderedDict()
        self._full = None
        self.requests = self.hits = self.computed_rows = self.evictions = 0
        self.peak_rows = 0
        if kernel == "rbf":
            self.diag = np.ones(self.n)
        else:
            with np.errstate(over="raise", invalid="raise"):
                try:
                    self.diag = np.einsum("ij,ij->i", self.X, self.X)
                except FloatingPointError as exc:
                    raise ValueError("linear diagonal overflow; scale inputs") from exc
        if not np.isfinite(self.diag).all():
            raise ValueError("nonfinite kernel diagonal")
        if cache == "full":
            self._full = np.empty((self.n, self.n), dtype=np.float64)
            for i in range(self.n):
                self._full[i] = self._calculate(i)
            self._full.setflags(write=False)
            self.peak_rows = self.n

    def _calculate(self, i):
        self.computed_rows += 1
        out = kernel_vector(self.X, self.X[i], self.kernel, self.gamma)
        out.setflags(write=False)
        return out

    def row(self, i):
        if isinstance(i, (bool, np.bool_)) or not isinstance(i, numbers.Integral):
            raise IndexError("kernel row index must be an integer")
        if i < 0 or i >= self.n:
            raise IndexError("kernel row out of range")
        self.requests += 1
        if self._full is not None:
            self.hits += 1
            return self._full[i]
        if i in self._rows:
            self.hits += 1
            self._rows.move_to_end(i)
            return self._rows[i]
        out = self._calculate(i)
        if self.cache == "lru" and self.capacity:
            self._rows[i] = out
            if len(self._rows) > self.capacity:
                self._rows.popitem(last=False)
                self.evictions += 1
            self.peak_rows = max(self.peak_rows, len(self._rows))
        return out

    def matvec(self, weights):
        out = np.zeros(self.n)
        for i in np.flatnonzero(weights):
            out += weights[i] * self.row(int(i))
        return out

    def stats(self):
        return {"requests": self.requests, "hits": self.hits,
                "computed_rows": self.computed_rows,
                "computed_kernel_values": self.computed_rows * self.n,
                "evictions": self.evictions,
                "capacity_rows": self.n if self.cache == "full" else self.capacity if self.cache == "lru" else 0,
                "peak_cached_rows": self.peak_rows,
                "cache_buffer_bytes": self.peak_rows * self.n * 8,
                "hit_rate": self.hits / self.requests if self.requests else 0.0}
