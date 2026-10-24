import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from kernelforge import KernelSVM
from kernelforge.data import Standardizer, load_wdbc, split_indices, save_bundle, load_bundle, moons


class IntegrationTests(unittest.TestCase):
    def test_scaler_matches_independent_reference(self):
        X = np.array([[1.,4.,0.],[2.,4.,5.],[7.,4.,-1.]])
        own, ref = Standardizer().fit(X), StandardScaler().fit(X)
        assert_allclose(own.transform(X), ref.transform(X))
        assert_allclose(own.transform([[100.,4.,2.]]),ref.transform([[100.,4.,2.]]))
        self.assertEqual(own.scale_[1],1)

    def test_scaler_is_frozen_for_test(self):
        scaler = Standardizer().fit([[0.],[2.]])
        assert_allclose(scaler.transform([[100.]]),[[99.]])
        assert_allclose(scaler.mean_,[1.])
        with self.assertRaises(RuntimeError): Standardizer().transform([[0.]])
        with self.assertRaises(ValueError): scaler.transform([[0.,1.]])

    def test_dataset_identity_and_disjoint_splits(self):
        X,y,info=load_wdbc()
        self.assertEqual(X.shape,(569,30))
        self.assertEqual(len(info["array_sha256"]),64)
        for seed in [2026,7,19]:
            parts=split_indices(y,seed)
            self.assertEqual([len(parts[k]) for k in ["train","validation","test"]],[341,114,114])
            all_ids=np.concatenate(list(parts.values()))
            assert_array_equal(np.sort(all_ids),np.arange(569))
            for ids in parts.values(): self.assertEqual(set(y[ids]),{-1,1})
            assert_array_equal(parts["train"],split_indices(y,seed)["train"])

    def test_model_bundle_roundtrip(self):
        X,y=moons(90,seed=3)
        scaler=Standardizer().fit(X)
        model=KernelSVM(C=3,gamma=.7).fit(scaler.transform(X),y)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"模型.npz"
            save_bundle(path,model,scaler)
            loaded,prep=load_bundle(path)
            assert_array_equal(model.predict(scaler.transform(X)),loaded.predict(prep.transform(X)))
            assert_allclose(model.decision_function(scaler.transform(X)),loaded.decision_function(prep.transform(X)),atol=0)
            self.assertTrue(loaded.converged_)

    def test_zero_support_roundtrip(self):
        # A loose but valid tolerance can stop at the all-zero dual solution.
        model=KernelSVM(kernel="linear",tol=2).fit([[-1.],[1.]],[-1,1])
        self.assertEqual(model.support_vectors_.shape,(0,1))
        self.assertTrue(model.converged_)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"zero.npz";model.save(path)
            loaded=KernelSVM.load(path)
            assert_array_equal(loaded.predict([[-1.],[1.]]),model.predict([[-1.],[1.]]))

    def test_model_rejects_schema_and_values(self):
        model=KernelSVM(kernel="linear").fit([[-1.],[1.]],[-1,1])
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"m"
            model.save(path)
            with np.load(path,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
            arrays["dual_coef"]=np.array([np.inf,1.])
            with path.open("wb") as f: np.savez(f,**arrays)
            with self.assertRaises(ValueError): KernelSVM.load(path)
            model.save(path)
            with np.load(path,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
            meta=json.loads(str(arrays["metadata"].item()));meta["schema"]=99
            arrays["metadata"]=np.array(json.dumps(meta))
            with path.open("wb") as f: np.savez(f,**arrays)
            with self.assertRaises(ValueError): KernelSVM.load(path)
        with self.assertRaises(RuntimeError): KernelSVM().save("unused.npz")

    def test_real_cli_predict(self):
        import subprocess
        import sys
        root=Path(__file__).resolve().parents[1]
        X=np.array([[-1.],[1.]])
        scaler=Standardizer().fit(X);model=KernelSVM(kernel="linear").fit(scaler.transform(X),[-1,1])
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"bundle.npz";raw=Path(td)/"raw.npy"
            save_bundle(path,model,scaler);np.save(raw,X)
            completed=subprocess.run([sys.executable,str(root/"run.py"),"predict","--model",str(path),"--input",str(raw)],capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            assert_array_equal(json.loads(completed.stdout)["prediction"],[-1,1])

    def test_logistic_budget_is_recorded(self):
        from kernelforge.experiments import fit_logistic
        X,y=moons(100,seed=83)
        _,state=fit_logistic(1,X,y,max_iter=1)
        self.assertFalse(state["converged"])
        self.assertEqual(state["iterations"],1)
        self.assertTrue(state["convergence_warnings"])

    def test_committed_reference_records(self):
        from kernelforge.verification import verify_reference
        root=Path(__file__).resolve().parents[1]
        result=verify_reference(root,output=False)
        self.assertEqual(result["status"],"passed")
        self.assertEqual(result["own_metric_rows_recomputed"],6)
        self.assertEqual(result["benchmark_raw_rows"],45)

    def test_real_data_reference(self):
        X,y,_=load_wdbc();parts=split_indices(y,2026)
        scaler=Standardizer().fit(X[parts["train"]])
        tr,te=scaler.transform(X[parts["train"]]),scaler.transform(X[parts["test"]])
        for kernel in ["linear","rbf"]:
            own=KernelSVM(C=1,kernel=kernel,gamma=.1,tol=1e-6).fit(tr,y[parts["train"]])
            ref=SVC(C=1,kernel=kernel,gamma=.1,tol=1e-8,shrinking=False).fit(tr,y[parts["train"]])
            self.assertTrue(own.converged_)
            assert_allclose(own.decision_function(te),ref.decision_function(te),atol=2e-5)
            assert_array_equal(own.predict(te),ref.predict(te))


if __name__=="__main__": unittest.main()
