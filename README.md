# KernelForge

[![checks](https://github.com/whx360/kernelforge/actions/workflows/ci.yml/badge.svg)](https://github.com/whx360/kernelforge/actions/workflows/ci.yml)

**Binary kernel SVM with from-scratch SMO, numerical certificates, and reproducible benchmarks.**

NumPy binary C-SVC with linear/RBF kernels, feasible pair updates, explicit KKT stopping, periodic gradient reconstruction, two working-set policies and none/LRU/full kernel-row storage. SciPy QP and LIBSVM are independent references, not the training implementation.

## Quick start

Python 3.11+; reference environment: Windows/Python 3.12.3.

```bash
git clone https://github.com/whx360/kernelforge.git
cd kernelforge
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python run.py demo
python run.py verify --reference
```

The 26-test suite covers analytic cases, eight independent small QP subcases, LIBSVM comparisons, duplicates, numerical boundaries, cache equivalence, ownership, persistence and the actual prediction CLI. Reference verification recalculates metrics from saved scores and checks core/record identity; it does not retrain models or rerun timings.

## Numerical contract

For y in {-1,+1}, maximize:

```text
W(alpha) = sum(alpha) - 0.5 * (alpha*y)^T K (alpha*y)
subject to 0 <= alpha <= C and y^T alpha = 0
```

- Pair updates preserve equality; zero curvature uses the feasible endpoint.
- Final KKT pair gap must be <= tol; budget exhaustion remains explicit.
- Diagnostics include primal/dual, duality gap, equality/box and margin residuals.
- Input is copied. Invalid refit clears old state. Empty predictions and valid zero-support models are supported.
- Scores are not probabilities. Exact zero predicts +1.

## Recorded results

Frozen snapshot: 2026-10-24, Ryzen 7 7435H, Windows CPU, one BLAS thread.

| Fixed synthetic workload | first-order/no cache | second-order/LRU | Median ratio |
| --- | ---: | ---: | ---: |
| n=800,d=16,C=1,gamma=1/16,KKT<=1e-5 | 490.107 ms | 207.411 ms | 2.36x |

Three rotated repeats; cache construction is included. Each workload fixes input/objective/tolerance. The three families jointly change n/d/gamma, not isolated sample-count scaling. Small LRU was slower than no cache in some configurations.

![Equal-precision benchmark](docs/figures/benchmark.svg)

Main WDBC held-out RBF accuracy: 0.9737; ROC-AUC: 0.9954; predictions agree with same-parameter LIBSVM. Logistic accuracy was higher (0.9825). All three predefined splits and negative findings are retained. One small dataset does not establish clinical utility or general superiority.

## Reproduce

```bash
python run.py test
python run.py experiment --quick  # separate smoke output
python run.py experiment          # full fixed protocol
python run.py verify              # local run, or reference if none exists
python run.py report
```

Each experiment creates a new directory. Generated results/caches/models are ignored. Frozen CSV/JSON lives in benchmarks/reference. WDBC uses sklearn's packaged features; no dataset download is needed after dependencies are installed.

A full-run verification checks saved models and records; --reference checks the committed snapshot without models. Neither is a new timing run. Cache-buffer bytes are not process peak memory.

## API

```python
from kernelforge import KernelSVM
model = KernelSVM(kernel="rbf", C=1, gamma=0.1, tol=1e-5)
model.fit(X_train_scaled, y_train)  # both labels -1 and +1
print(model.converged_, model.diagnostics_)
prediction = model.predict(X_test_scaled)
```

Install with `python -m pip install --no-deps -e .` for API use. Reuse the training scaler at inference. Bundle functions save it with the model; `run.py predict --model <bundle.npz> --input <raw_features.npy>` applies it.

## Repository map

- src/kernelforge: numerical core, preprocessing, experiments, reports and verification
- tests: independent numerical, boundary and integration checks
- docs: design, mathematics, experiments and sources
- benchmarks/reference: small measured records and provenance
- .github/workflows: Windows/Linux checks

[Design](docs/design.md) | [Mathematics](docs/mathematics.md) | [Experiments](docs/experiments.md) | [Sources](docs/sources.md)

Scope: dense binary C-SVC, no custom kernels, multiclass, SVR, shrinking, probability calibration, GPU or parallel training. Pair-gain selection is inspired by second-order research, not a reproduction of LIBSVM's full algorithm. Classical methods are cited, not claimed as novel.
