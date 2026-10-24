"""Hand cases, independent QP/reference, numeric and input boundary checks."""
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize
from sklearn.svm import SVC
from kernelforge import KernelSVM
from kernelforge.kernels import KernelRows


class KernelTests(unittest.TestCase):
    def test_values(self):
        X = np.array([[0., 0.], [1., 2.], [-2., 1.]])
        for kind in ["linear", "rbf"]:
            rows = KernelRows(X, kernel=kind, gamma=.4)
            got = np.stack([rows.row(i) for i in range(len(X))])
            expected = X @ X.T if kind == "linear" else np.exp(-.4 * ((X[:, None] - X[None, :])**2).sum(2))
            assert_allclose(got, expected, atol=1e-14)
            assert_allclose(got, got.T)

    def test_large_translation_rbf(self):
        X = np.array([[0., 1.], [2., 4.], [1., 7.]])
        a, b = KernelRows(X, gamma=.1), KernelRows(X + 1e12, gamma=.1)
        for i in range(3):
            assert_allclose(a.row(i), b.row(i), atol=1e-15)

    def test_cache_equivalence_and_budget(self):
        X = np.arange(24.).reshape(6, 4)
        for cache in ["none", "lru", "full"]:
            rows = KernelRows(X, cache=cache, cache_mb=96 / 2**20)
            for i in [0, 1, 0, 2, 3, 0]:
                assert_allclose(rows.row(i), np.exp(-.1 * ((X-X[i])**2).sum(1)))
            if cache == "lru":
                self.assertLessEqual(rows.stats()["cache_buffer_bytes"], 96)
                self.assertGreater(rows.hits, 0)
                self.assertGreater(rows.evictions, 0)
            with self.assertRaises(ValueError):
                rows.row(0)[0] = 2

    def test_tiny_cache_is_no_store(self):
        rows = KernelRows([[0.], [1.]], cache="lru", cache_mb=1e-9)
        rows.row(0); rows.row(0)
        self.assertEqual(rows.hits, 0)
        self.assertEqual(rows.stats()["cache_buffer_bytes"], 0)

    def test_invalid(self):
        for X in [[], [1,2], np.empty((2,0)), [[np.nan]], [[np.inf]]]:
            with self.subTest(X=str(X)), self.assertRaises(ValueError):
                KernelRows(X)
        rows = KernelRows([[0.], [1.]])
        for i in [-1, 2, 1.2, True]:
            with self.assertRaises(IndexError): rows.row(i)
        with self.assertRaises(ValueError): KernelRows([[1.]], gamma=0)
        with self.assertRaises(ValueError): KernelRows([[1e308]], kernel="linear")


class SMOTests(unittest.TestCase):
    def check_certificate(self, model, tol=1e-7):
        self.assertTrue(model.converged_, model.diagnostics_)
        self.assertLessEqual(model.diagnostics_["pair_gap"], model.tol)
        self.assertLess(model.diagnostics_["equality_residual"], tol)
        self.assertEqual(model.diagnostics_["box_violation"], 0)
        self.assertGreaterEqual(model.diagnostics_["duality_gap"], -tol)

    def test_two_points_hand_solution(self):
        for working in ["first_order", "second_order"]:
            model = KernelSVM(kernel="linear", C=1, tol=1e-10, working_set=working).fit([[-1.], [1.]], [-1,1])
            assert_allclose(model.alpha_, [.5,.5], atol=1e-12)
            assert_allclose(model.bias_, 0, atol=1e-12)
            assert_allclose(model.diagnostics_["primal"], .5, atol=1e-12)
            self.check_certificate(model)

    def test_small_C_hits_bounds(self):
        model = KernelSVM(kernel="linear", C=.1, tol=1e-10).fit([[-1.], [1.]], [-1,1])
        assert_allclose(model.alpha_, [.1,.1])
        assert_allclose(model.diagnostics_["duality_gap"], 0, atol=1e-12)
        self.check_certificate(model)

    def test_opposite_duplicate_zero_curvature(self):
        for kernel in ["linear", "rbf"]:
            model = KernelSVM(kernel=kernel, C=2).fit([[3.,1.],[3.,1.]], [-1,1])
            assert_allclose(model.alpha_, [2,2])
            self.assertEqual(model.diagnostics_["zero_curvature_steps"], 1)
            assert_array_equal(model.predict([[3.,1.]]), [1])
            self.check_certificate(model)

    def test_xor_kernel(self):
        X = [[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]]
        y = [1,-1,-1,1]
        model = KernelSVM(C=10, gamma=1, tol=1e-9).fit(X,y)
        assert_array_equal(model.predict(X), y)
        self.check_certificate(model)

    def test_history_dual_increases_and_final_gradient_is_direct(self):
        rng=np.random.default_rng(61)
        X=rng.normal(size=(90,4));y=np.where(X[:,0]+X[:,1]*X[:,2]>0,1,-1)
        model=KernelSVM(gamma=.4,tol=1e-7,refresh_every=7).fit(X,y)
        self.check_certificate(model)
        dual=np.array([r["dual"] for r in model.history_])
        self.assertTrue(np.all(np.diff(dual)>=-1e-9))
        K=np.exp(-.4*((X[:,None]-X[None,:])**2).sum(2))
        expected=model.alpha_.sum()-.5*(model.alpha_*y)@K@(model.alpha_*y)
        assert_allclose(expected,model.diagnostics_["dual"],atol=1e-11)

    def test_unequal_duplicate_labels_no_free_support(self):
        model=KernelSVM(C=2).fit([[1.,2.]]*3,[1,1,-1])
        self.check_certificate(model)
        self.assertLess(model.diagnostics_["duality_gap"],1e-10)
        assert_array_equal(model.predict([[1.,2.]]),[1])

    def test_independent_qp(self):
        for seed in range(4):
            for kernel in ["linear", "rbf"]:
                rng = np.random.default_rng(seed)
                X, y = rng.normal(size=(10,3)), np.array([-1,1]*5)
                rows = KernelRows(X, kernel=kernel, gamma=.4, cache="full")
                K = np.stack([rows.row(i) for i in range(10)])
                Q = K * np.outer(y,y)
                result = minimize(lambda a: .5*a@Q@a-a.sum(), np.zeros(10),
                                  jac=lambda a: Q@a-1, bounds=[(0,.7)]*10,
                                  constraints={"type":"eq", "fun":lambda a: a@y, "jac":lambda a:y.astype(float)},
                                  method="SLSQP", options={"ftol":1e-12,"maxiter":2000})
                self.assertTrue(result.success, result.message)
                model = KernelSVM(C=.7, kernel=kernel, gamma=.4, tol=1e-8).fit(X,y)
                self.check_certificate(model)
                assert_allclose(model.diagnostics_["dual"], -result.fun, rtol=1e-7, atol=1e-7)
                self.assertLess(model.diagnostics_["duality_gap"], 2e-7)

    def test_reference_and_cache_modes(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(55,5)); y = np.where(X[:,0]*X[:,1]+.2*X[:,2]>0,1,-1)
        query = rng.normal(size=(30,5))
        for kernel in ["linear","rbf"]:
            ref = SVC(C=1.3, kernel=kernel, gamma=.3, tol=1e-9, shrinking=False).fit(X,y)
            for working in ["first_order","second_order"]:
                decisions = []
                for cache in ["none","lru","full"]:
                    model = KernelSVM(C=1.3, kernel=kernel,gamma=.3,tol=1e-7,
                                      working_set=working,cache=cache,cache_mb=.001).fit(X,y)
                    self.check_certificate(model)
                    decisions.append(model.decision_function(query))
                    assert_allclose(decisions[-1], ref.decision_function(query), atol=3e-5)
                assert_allclose(decisions[0], decisions[1], atol=1e-12)
                assert_allclose(decisions[0], decisions[2], atol=1e-12)

    def test_invalid_inputs(self):
        for kwargs in [{"C":0},{"C":np.inf},{"gamma":-1},{"tol":True},{"max_iter":0},
                       {"max_iter":1.2},{"refresh_every":0},{"cache":"bad"},{"working_set":"bad"}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): KernelSVM(**kwargs)
        for y in [[1,1],[0,1],[1],[[1],[-1]],[1,np.nan]]:
            with self.assertRaises(ValueError): KernelSVM().fit([[0.],[1.]],y)

    def test_budget_is_explicit(self):
        rng = np.random.default_rng(11)
        model = KernelSVM(max_iter=1,tol=1e-12).fit(rng.normal(size=(20,3)),[-1,1]*10)
        self.assertFalse(model.converged_)
        self.assertEqual(model.status_, "max_iter")
        self.assertEqual(len(model.predict([[0.,0.,0.]])), 1)

    def test_lifetime_empty_predict_refit(self):
        X = np.array([[-1.],[1.]])
        model = KernelSVM(kernel="linear").fit(X,[-1,1])
        before = model.decision_function(X.copy())
        X[:] = 99
        assert_allclose(before,model.decision_function([[-1.],[1.]]))
        self.assertEqual(model.predict(np.empty((0,1))).shape, (0,))
        with self.assertRaises(ValueError): model.predict([[1.,2.]])
        with self.assertRaises(ValueError): model.predict([[np.nan]])
        with self.assertRaises(RuntimeError): KernelSVM().predict([[1.]])
        with self.assertRaises(ValueError): model.fit([[1.],[2.]],[1,1])
        with self.assertRaises(RuntimeError): model.predict([[1.]])


if __name__ == "__main__": unittest.main()
