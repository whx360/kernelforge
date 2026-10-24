"""Feasible pair-coordinate SMO implemented from the binary C-SVC dual."""
import numbers
import json
from pathlib import Path
import numpy as np
from .kernels import KernelRows, kernel_vector, matrix, positive


def sets(alpha, y, C):
    up = ((y == 1) & (alpha < C)) | ((y == -1) & (alpha > 0))
    low = ((y == 1) & (alpha > 0)) | ((y == -1) & (alpha < C))
    return up, low


def certificate(alpha, y, C, F):
    """Diagnostics from a supplied F=y-K@(alpha*y); final F is recomputed."""
    up, low = sets(alpha, y, C)
    lo, hi = float(np.max(F[up])), float(np.min(F[low]))
    b = 0.5 * lo + 0.5 * hi
    f0 = y - F
    norm2 = float(np.dot(alpha * y, f0))
    margin = y * (f0 + b)
    primal = 0.5 * norm2 + C * float(np.maximum(0, 1 - margin).sum())
    dual = float(alpha.sum()) - 0.5 * norm2
    violation = np.where(alpha == 0, np.maximum(0, 1 - margin),
                         np.where(alpha == C, np.maximum(0, margin - 1), np.abs(margin - 1)))
    out = {"pair_gap": max(0.0, lo - hi), "bias": b,
           "primal": primal, "dual": dual, "duality_gap": primal - dual,
           "relative_duality_gap": (primal - dual) / max(1.0, abs(primal), abs(dual)),
           "equality_residual": abs(float(np.dot(alpha, y))),
           "box_violation": max(0.0, float(-alpha.min()), float(alpha.max() - C)),
           "margin_kkt_residual": float(violation.max()),
           "support_count": int(np.count_nonzero(alpha > 0)),
           "free_support_count": int(np.count_nonzero((alpha > 0) & (alpha < C)))}
    if not all(np.isfinite(v) for v in out.values()):
        raise ValueError("nonfinite optimization certificate")
    return out


class KernelSVM:
    """Binary C-SVC. Training is deterministic; no QP/estimator is called here.

    tol is the feasible-pair KKT gap. A budget exit retains converged_=False.
    second_order is a constrained pair-gain heuristic inspired by WSS ideas,
    not a claim to reproduce LIBSVM's complete algorithm.
    """
    def __init__(self, C=1.0, kernel="rbf", gamma=0.1, tol=1e-5,
                 max_iter=100000, working_set="second_order", cache="lru", cache_mb=16.0,
                 refresh_every=200):
        self.C, self.gamma, self.tol = positive(C, "C"), positive(gamma, "gamma"), positive(tol, "tol")
        self.cache_mb = positive(cache_mb, "cache_mb")
        for name, value in [("max_iter", max_iter), ("refresh_every", refresh_every)]:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if kernel not in ("linear", "rbf") or cache not in ("none", "lru", "full"):
            raise ValueError("unsupported kernel/cache")
        if working_set not in ("first_order", "second_order"):
            raise ValueError("unsupported working set")
        self.kernel, self.cache, self.working_set = kernel, cache, working_set
        self.max_iter, self.refresh_every = int(max_iter), int(refresh_every)

    def get_params(self):
        return {name: getattr(self, name) for name in ["C", "kernel", "gamma", "tol", "max_iter",
                "working_set", "cache", "cache_mb", "refresh_every"]}

    def fit(self, X, y):
        # Clear a previous fit before validating a replacement training request.
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        X = matrix(X)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1 or len(y) != len(X) or not np.isfinite(y).all():
            raise ValueError("y must be a finite vector of length n")
        if set(np.unique(y)) != {-1.0, 1.0}:
            raise ValueError("training requires both labels -1 and +1")
        rows = KernelRows(X, self.kernel, self.gamma, self.cache, self.cache_mb)
        y = y.copy()
        n, d = X.shape
        alpha, F = np.zeros(n), y.copy()
        self.history_ = []
        refreshes = zero_curvature_steps = 0
        status = "max_iter"
        for iteration in range(self.max_iter + 1):
            if iteration and iteration % self.refresh_every == 0:
                F = y - rows.matvec(alpha * y)
                refreshes += 1
            up, low = sets(alpha, y, self.C)
            i = int(np.argmax(np.where(up, F, -np.inf)))
            j_first = int(np.argmin(np.where(low, F, np.inf)))
            pair_gap = F[i] - F[j_first]
            if pair_gap <= self.tol:
                F = y - rows.matvec(alpha * y)
                refreshes += 1
                up, low = sets(alpha, y, self.C)
                i = int(np.argmax(np.where(up, F, -np.inf)))
                j_first = int(np.argmin(np.where(low, F, np.inf)))
                pair_gap = F[i] - F[j_first]
                if pair_gap <= self.tol:
                    status = "converged"
                    break
            if iteration % 25 == 0:
                self.history_.append({"iteration": iteration, **certificate(alpha, y, self.C, F)})
            if iteration == self.max_iter:
                break
            row_i = rows.row(i)
            cap_i = self.C - alpha[i] if y[i] == 1 else alpha[i]
            caps_j = np.where(y == 1, alpha, self.C - alpha)
            gaps = F[i] - F
            eta = rows.diag[i] + rows.diag - 2 * row_i
            scale = np.maximum(1.0, np.abs(rows.diag[i]) + np.abs(rows.diag))
            if np.any(eta < -1e-10 * scale):
                raise ValueError("negative kernel curvature beyond roundoff")
            eta = np.maximum(eta, 0)
            if self.working_set == "first_order":
                j = j_first
            else:
                bounds = np.minimum(cap_i, caps_j)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    steps = np.where(eta > 0, gaps / eta, bounds)
                steps = np.minimum(np.maximum(steps, 0), bounds)
                gains = gaps * steps - 0.5 * eta * steps**2
                gains[~low | (gaps <= 0)] = -np.inf
                j = int(np.argmax(gains))
            cap_j = caps_j[j]
            bound = min(cap_i, cap_j)
            if eta[j] > 0:
                with np.errstate(over="ignore"):
                    t = min(float((F[i] - F[j]) / eta[j]), bound)
            else:
                t = bound
                zero_curvature_steps += 1
            if not np.isfinite(t) or t <= 0:
                status = "no_progress"
                break
            alpha[i] += y[i] * t
            alpha[j] -= y[j] * t
            # Exact chosen endpoints prevent spurious near-bound interior points.
            if t == cap_i:
                alpha[i] = self.C if y[i] == 1 else 0.0
            if t == cap_j:
                alpha[j] = 0.0 if y[j] == 1 else self.C
            if np.any(alpha < 0) or np.any(alpha > self.C):
                raise ArithmeticError("pair update escaped the feasible box")
            F -= t * (row_i - rows.row(j))
            if not np.isfinite(F).all():
                raise ArithmeticError("nonfinite gradient update")
        F = y - rows.matvec(alpha * y)
        refreshes += 1
        final = certificate(alpha, y, self.C, F)
        # Final direct recomputation owns convergence, never incremental drift.
        converged = final["pair_gap"] <= self.tol
        if converged:
            status = "converged"
        elif status == "converged":
            status = "numerical_drift"
        self.history_.append({"iteration": iteration, **final})
        support = np.flatnonzero(alpha > 0)
        self.X_, self.y_, self.alpha_ = rows.X, y, alpha
        self.support_vectors_ = rows.X[support].copy()
        self.dual_coef_ = (alpha[support] * y[support]).copy()
        self.support_ = support
        self.n_features_in_, self.bias_ = d, final["bias"]
        self.n_iter_, self.converged_, self.status_ = iteration, converged, status
        self.diagnostics_ = {**final, "iterations": iteration, "converged": converged,
                             "status": status, "refreshes": refreshes,
                             "zero_curvature_steps": zero_curvature_steps}
        self.kernel_stats_ = rows.stats()
        return self

    def decision_function(self, X):
        if not hasattr(self, "support_vectors_"):
            raise RuntimeError("model is not fitted")
        X = matrix(X, allow_empty=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("feature count differs from fitted model")
        scores = np.empty(len(X))
        for i, x in enumerate(X):
            scores[i] = np.dot(self.dual_coef_, kernel_vector(self.support_vectors_, x,
                                                            self.kernel, self.gamma)) + self.bias_
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite prediction")
        return scores

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def save(self, path, *, scaler_mean=None, scaler_scale=None):
        if not hasattr(self, "support_vectors_"):
            raise RuntimeError("model is not fitted")
        metadata = {"schema": 1, "params": self.get_params(), "n_features": self.n_features_in_,
                    "diagnostics": self.diagnostics_, "scope": "inference_only_not_training_resume"}
        arrays = {"metadata": np.array(json.dumps(metadata, allow_nan=False)),
                  "support_vectors": self.support_vectors_, "dual_coef": self.dual_coef_,
                  "bias": np.array(self.bias_), "support_indices": self.support_}
        if (scaler_mean is None) != (scaler_scale is None):
            raise ValueError("both scaler arrays are required")
        if scaler_mean is not None:
            mean, scale = np.asarray(scaler_mean), np.asarray(scaler_scale)
            if mean.shape != (self.n_features_in_,) or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("invalid scaler arrays")
            arrays.update(scaler_mean=mean, scaler_scale=scale)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # File handle avoids NumPy silently appending an extension.
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"].item()))
            if meta.get("schema") != 1:
                raise ValueError("unsupported model schema")
            model = cls(**meta["params"])
            sv = matrix(z["support_vectors"], allow_empty=True)
            coef = np.asarray(z["dual_coef"], dtype=np.float64)
            bias = np.asarray(z["bias"])
            support = np.asarray(z["support_indices"])
            d = meta["n_features"]
            if not isinstance(d, int) or d <= 0 or sv.shape[1] != d:
                raise ValueError("invalid saved feature count")
            if coef.shape != (len(sv),) or not np.isfinite(coef).all() or bias.shape != () or not np.isfinite(bias):
                raise ValueError("invalid saved prediction arrays")
            if support.shape != (len(sv),) or not np.issubdtype(support.dtype, np.integer) or np.any(support < 0):
                raise ValueError("invalid saved support indices")
            model.support_vectors_, model.dual_coef_, model.support_ = sv.copy(), coef.copy(), support.copy()
            model.bias_, model.n_features_in_ = float(bias), d
            model.diagnostics_ = meta["diagnostics"]
            model.converged_ = bool(model.diagnostics_["converged"])
            model.status_, model.n_iter_ = model.diagnostics_["status"], model.diagnostics_["iterations"]
            model.history_ = []
        return model
