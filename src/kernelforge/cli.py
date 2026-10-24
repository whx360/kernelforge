"""Project commands for tests, experiments, inference and record verification."""
import argparse
import json
from pathlib import Path


def main(root: Path) -> int:
    parser = argparse.ArgumentParser(description="KernelForge: auditable binary kernel SVM")
    parser.add_argument("command", choices=["smoke", "test", "demo", "predict", "experiment", "report", "verify"])
    parser.add_argument("--core", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--input", help="NPY raw finite 2D features")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reference", action="store_true", help="verify committed records without saved models")
    args = parser.parse_args()
    if args.command == "smoke":
        path = root / "artifacts" / "spine_smoke.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"purpose": "entrypoint_smoke_only", "status": "ready",
                                    "model_trained": False}, indent=2) + "\n", encoding="utf-8")
        print(path)
    elif args.command == "test":
        import unittest
        suite = unittest.defaultTestLoader.discover(str(root / "tests"),
                                                    pattern="test_core.py" if args.core else "test_*.py")
        from threadpoolctl import threadpool_limits
        import io
        from .data import write_json
        stream=io.StringIO()
        with threadpool_limits(limits=1):
            result = unittest.TextTestRunner(verbosity=2,stream=stream).run(suite)
        print(stream.getvalue())
        (root/"artifacts").mkdir(exist_ok=True)
        (root/"artifacts"/"tests_latest.log").write_text(stream.getvalue(),encoding="utf-8")
        write_json(root/"artifacts"/"tests_latest.json",{"tests_run":result.testsRun,"failures":len(result.failures),
                   "errors":len(result.errors),"scope":"core" if args.core else "full","passed":result.wasSuccessful()})
        return 0 if result.wasSuccessful() else 1
    elif args.command == "demo":
        from .data import moons, Standardizer, write_json, save_bundle
        from .svm import KernelSVM
        from threadpoolctl import threadpool_limits
        X,y = moons()
        # A synthetic demonstration, explicitly not a held-out real-data result.
        scaler = Standardizer().fit(X)
        with threadpool_limits(limits=1):
            model = KernelSVM(C=3,gamma=.7).fit(scaler.transform(X),y)
        write_json(root/"artifacts"/"demo.json", {"scope":"synthetic_training_demo_only",
                   "diagnostics":model.diagnostics_, "kernel_stats":model.kernel_stats_})
        save_bundle(root/"artifacts"/"demo_model.npz",model,scaler)
        print(json.dumps(model.diagnostics_,indent=2))
        return 0 if model.converged_ else 1
    elif args.command == "predict":
        if not args.model or not args.input:
            parser.error("predict requires --model and --input")
        import numpy as np
        from .data import load_bundle
        model,scaler = load_bundle(args.model)
        X = np.load(args.input,allow_pickle=False)
        print(json.dumps({"decision":model.decision_function(scaler.transform(X)).tolist(),
                          "prediction":model.predict(scaler.transform(X)).tolist()},allow_nan=False))
    elif args.command == "experiment":
        from .experiments import run_experiments
        print(run_experiments(root,args.quick))
    elif args.command == "report":
        from .report import latest,generate
        out=latest(root);generate(out);print(out/"report.html")
    elif args.command == "verify":
        import subprocess
        import sys
        for command in ["test","demo"]:
            completed=subprocess.run([sys.executable,str(root/"run.py"),command],cwd=root)
            if completed.returncode:return completed.returncode
        from .verification import verify, verify_reference
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            record=verify_reference(root) if args.reference or not (root/"results"/"latest.json").exists() else verify(root)
        print(json.dumps(record,indent=2))
    return 0
