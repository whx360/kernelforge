# Sources and boundaries

- Platt (1998), [Sequential Minimal Optimization](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/): pair decomposition and analytic updates.
- Fan, Chen and Lin (2005), [Second-order working-set selection](https://jmlr.org/papers/v6/fan05a.html): curvature-aware selection. This implementation's clipped pair gain is not the complete published WSS/LIBSVM algorithm.
- Chang and Lin (2011), [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/): independent SVM reference via sklearn.
- [UCI WDBC](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), loaded through [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html): array identity, label mapping and version are recorded. Raw image features are not recreated; original archive byte identity is not claimed.

Implemented here: kernels, cache policy, pair updates, working sets, gradient maintenance, certificates, preprocessing, persistence and experiment organization. NumPy provides arrays; OrderedDict provides the cache container. SciPy/sklearn optimizers are references only.

Code was written for this repository with AI assistance. Classical algorithms are not claimed as original research. Other papers' speedups are not project measurements.

MIT covers original repository code. Data/dependencies retain their own terms. Raw datasets and fitted models are not redistributed.
