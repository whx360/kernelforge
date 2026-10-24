"""Artifact integrity plus model-to-record reproduction, without retuning."""
import hashlib
import json
import numpy as np
from .data import load_wdbc, load_bundle, write_json
from .experiments import source_manifest, metrics
from .report import latest, read_csv


def verify(root):
    out=latest(root);manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    if manifest["scope"]!="formal" or manifest["status"]!="completed":
        raise RuntimeError("formal completed run required")
    current=source_manifest(root)
    # Verification module is not involved in fitting. It may be introduced after
    # a historical run; all source paths actually in that run must still match.
    for name,digest in manifest["source_files"].items():
        if current.get(name)!=digest:raise RuntimeError(f"execution source changed: {name}")
    for name,digest in manifest["artifact_files"].items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"artifact changed: {name}")
    X,y,source=load_wdbc()
    expected=json.loads((out/"data_source.json").read_text(encoding="utf-8"))
    if source["array_sha256"]!=expected["array_sha256"]:raise RuntimeError("dataset identity changed")
    parts=json.loads((out/"splits.json").read_text(encoding="utf-8"))
    results=json.loads((out/"test_results.json").read_text(encoding="utf-8"))
    predictions=read_csv(out/"test_predictions.csv");checked=0
    for seed,splits in parts.items():
        ids=np.concatenate([np.asarray(v) for v in splits.values()])
        if not np.array_equal(np.sort(ids),np.arange(len(y))):raise RuntimeError("split overlap/coverage error")
        for kernel in ["linear","rbf"]:
            model,scaler=load_bundle(out/"models"/f"{kernel}_seed{seed}.npz")
            test=np.asarray(splits["test"])
            scores=model.decision_function(scaler.transform(X[test]))
            saved=[r for r in predictions if r["seed"]==seed and r["model"]==kernel]
            if [int(r["source_row"]) for r in saved]!=test.tolist():raise RuntimeError("prediction row order mismatch")
            if not np.allclose(scores,[float(r["decision"]) for r in saved],rtol=0,atol=1e-12):
                raise RuntimeError("saved inference does not reproduce score records")
            row=next(r for r in results if str(r["seed"])==seed and r["model"]=="own_"+kernel)
            measured=metrics(y[test],scores)
            for key in ["accuracy","balanced_accuracy","macro_f1","roc_auc","hinge"]:
                if abs(measured[key]-row[key])>1e-12:raise RuntimeError(f"metric mismatch {key}")
            if not row["converged"] or row["pair_gap"]>model.tol or row["equality_residual"]>1e-7 or row["box_violation"]!=0:
                raise RuntimeError("selected model numerical certificate failed")
            checked+=1
    raw=read_csv(out/"benchmark_raw.csv");summary=read_csv(out/"benchmark_summary.csv")
    for r in raw:
        if r["converged"]!="True" or float(r["pair_gap"])>float(r["tol"]):raise RuntimeError("unequal precision benchmark")
    for r in summary:
        group=[s for s in raw if s["n"]==r["n"] and s["configuration"]==r["configuration"]]
        if len(group)!=3:raise RuntimeError("formal benchmark requires three repeats")
        if abs(float(r["median_seconds"])-float(np.median([float(s["fit_seconds"]) for s in group])))>1e-12:
            raise RuntimeError("benchmark summary median mismatch")
    record={"status":"passed","scope":"source/artifact identity, dataset/splits, saved inference, metrics and benchmark consistency",
            "run_directory":str(out.relative_to(root)).replace("\\","/"),"source_files_checked":len(manifest["source_files"]),
            "artifact_files_checked":len(manifest["artifact_files"]),"models_reproduced":checked,
            "benchmark_raw_rows":len(raw),"test_result_rows":len(results),
            "limitation":"not a fresh timing rerun, formal proof or independent clinical validation"}
    write_json(root/"artifacts"/"verification_latest.json",record)
    return record


def verify_reference(root, *, output=True):
    """Check portable frozen records; no model or timing reproduction implied."""
    out=root/"benchmarks"/"reference"
    manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema")!=1:raise RuntimeError("unsupported reference schema")
    for name,digest in manifest["numeric_core_source_files"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"numeric core differs from frozen experiment: {name}")
    for name,digest in manifest["files"].items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"reference record changed: {name}")
    X,y,source=load_wdbc()
    expected=json.loads((out/"data_source.json").read_text(encoding="utf-8"))
    if source["array_sha256"]!=expected["array_sha256"]:raise RuntimeError("reference dataset mismatch")
    parts=json.loads((out/"splits.json").read_text(encoding="utf-8"))
    results=json.loads((out/"test_results.json").read_text(encoding="utf-8"))
    predictions=read_csv(out/"test_predictions.csv")
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    checked=0
    for seed,splits in parts.items():
        ids=np.concatenate([np.asarray(v) for v in splits.values()])
        if not np.array_equal(np.sort(ids),np.arange(len(y))):raise RuntimeError("reference split overlap/coverage error")
        for kernel in ["linear","rbf"]:
            saved=[r for r in predictions if r["seed"]==seed and r["model"]==kernel]
            test=np.asarray(splits["test"])
            if [int(r["source_row"]) for r in saved]!=test.tolist():raise RuntimeError("reference prediction order mismatch")
            if [int(r["truth"]) for r in saved]!=y[test].tolist():raise RuntimeError("reference prediction labels mismatch")
            scores=np.asarray([float(r["decision"]) for r in saved])
            measured=metrics(y[test],scores)
            row=next(r for r in results if str(r["seed"])==seed and r["model"]=="own_"+kernel)
            for key in ["accuracy","balanced_accuracy","macro_f1","roc_auc","hinge"]:
                if abs(measured[key]-row[key])>1e-12:raise RuntimeError(f"reference metric mismatch: {key}")
            if not row["converged"] or row["pair_gap"]>protocol["tol"]:raise RuntimeError("reference model did not meet tolerance")
            checked+=1
    raw=read_csv(out/"benchmark_raw.csv");summaries=read_csv(out/"benchmark_summary.csv")
    if len(raw)!=manifest["benchmark_raw_rows"] or len(results)!=manifest["test_result_rows"]:
        raise RuntimeError("reference result row counts changed")
    for r in raw:
        if r["converged"]!="True" or float(r["pair_gap"])>float(r["tol"]):raise RuntimeError("uncertified reference timing")
    for r in summaries:
        group=[s for s in raw if s["n"]==r["n"] and s["configuration"]==r["configuration"]]
        if len(group)!=3:raise RuntimeError("three timing repeats required")
        median=float(np.median([float(s["fit_seconds"]) for s in group]))
        if abs(median-float(r["median_seconds"]))>1e-12:raise RuntimeError("reference timing median mismatch")
    record={"status":"passed","scope":"frozen record integrity and saved-score metric recalculation; no saved-model inference or new timing run",
            "numeric_core_files_checked":len(manifest["numeric_core_source_files"]),
            "reference_files_checked":len(manifest["files"]),"own_metric_rows_recomputed":checked,
            "benchmark_raw_rows":len(raw),"test_result_rows":len(results)}
    if output:write_json(root/"artifacts"/"reference_verification.json",record)
    return record
