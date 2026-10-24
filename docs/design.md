# Design

```text
finite (X,y) -> train-only Standardizer -> KernelRows -> feasible-pair SMO
                                              |             |
                                      none / LRU / full      +-> certificates
                                                            +-> model / bundle
```

The numerical core is NumPy-only. SciPy SLSQP and sklearn SVC are independent references. Preprocessing is implemented locally; sklearn supplies the public dataset, split routine, metrics and baselines.

| Component | Responsibility | Contract |
| --- | --- | --- |
| kernels.py | kernel rows, bounded LRU | immutable owned data/rows, finite outputs |
| svm.py | pair updates, working set, stopping | feasible box/equality, explicit budget |
| data.py | standardization, splits, bundles | train-only fitting, no pickle |
| experiments.py | fixed grids, controlled timings | validation-only selection, convergence eligibility |
| verification.py | generated/reference checks | inference versus saved-score verification |
| report.py | plots and local report | draw from raw records |

Training copies input. A new fit clears old state before validation. Prediction rejects wrong dimensions, accepts empty batches and assigns +1 at score zero. NPZ bundles support inference, not training resume.

First-order selection chooses the largest feasible KKT violation. Second-order selection fixes the most violating i and chooses j by clipped dual gain. Both share update/refresh/stopping rules. More candidate work per step can offset iteration reduction.

A missing kernel row costs O(nd); selection and incremental update cost O(n). With T updates and no cache, a coarse cost is O(Tnd), plus reconstruction/monitoring. Full storage builds O(n²d) kernel work and retains O(n²) values. An r-row LRU retains O(rn), not a bound on process memory. Prediction costs O(qsd).

RBF uses direct coordinate differences to reduce cancellation under large translations. Extreme finite arithmetic can still overflow and is rejected. No universal floating-point stability guarantee is claimed.

Persistence validates shape and finite values with pickle disabled. A zero-row support matrix is legal when a loose tolerance accepted zero coefficients. A bundle includes the training-fitted scaler.
