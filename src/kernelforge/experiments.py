"""Frozen protocol: references, validation-only selection and controlled timings."""
import cProfile
import csv
import datetime
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import pstats
import sys
import time
import uuid
import warnings
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from .svm import KernelSVM
from .data import Standardizer, load_wdbc, split_indices, write_json, save_bundle, array_hash, moons

PROTOCOL = {"version":"formal_v1", "split_seeds":[2026,7,19], "C_grid":[.1,1.,10.],
            "gamma_grid":[.01,.1,1.], "tol":1e-5, "max_iter":100000,
            "selection":"validation balanced_accuracy, tie by declared candidate order",
            "positive_label":"malignant +1", "blas_threads":1, "timing_repeats":3,
            "timing_scope":"fit wall time including kernel cache construction; prediction separate",
            "inference":"scores are not probabilities", "stats_scope":"split variability, no significance claim",
            "eligibility":"SVM pair_gap convergence; logistic no ConvergenceWarning and n_iter<max_iter",
            "benchmark_workloads":"(n,d)=(200,8),(480,12),(800,16), gamma=1/d; not isolated sample-count scaling",
            "benchmark_generator":"seed2026 Gaussian X; sign(x0*x1+0.3*x2+0.15*Gaussian noise)",
            "warmup":"one fit per configuration per workload, excluded",
            "score_comparison_atol":2e-4}


def fit_logistic(C, X, y, max_iter=5000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    model=LogisticRegression(C=C,max_iter=max_iter,tol=1e-10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning)
        model.fit(X,y)
    warning_text=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
    iterations=int(np.max(model.n_iter_))
    state={"iterations":iterations,"converged":not warning_text and iterations<max_iter,
           "convergence_scope":"no ConvergenceWarning and n_iter<max_iter; no independent KKT certificate",
           "convergence_warnings":warning_text}
    return model,state


def table(path, records):
    if not records: return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in records for k in row))
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)


def metrics(y, score):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
    prediction=np.where(score>=0,1,-1)
    return {"accuracy":float(accuracy_score(y,prediction)),
            "balanced_accuracy":float(balanced_accuracy_score(y,prediction)),
            "macro_f1":float(f1_score(y,prediction,average="macro")),
            "roc_auc":float(roc_auc_score(y,score)),
            "hinge":float(np.maximum(0,1-y*score).mean()),
            "confusion_labels": [-1,1], "confusion":confusion_matrix(y,prediction,labels=[-1,1]).tolist()}


def environment():
    name=platform.processor()
    if sys.platform=="win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            name=winreg.QueryValueEx(key,"ProcessorNameString")[0].strip()
    return {"date_asia_shanghai":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).date().isoformat(), "python":sys.version,"executable":sys.executable,
            "platform":platform.platform(),"cpu":name,
            "versions":{p:importlib.metadata.version(p) for p in ["numpy","scipy","scikit-learn","matplotlib","threadpoolctl"]},
            "threadpools":threadpool_info(),"gpu_used":False}


def synthetic(n,d,seed=2026):
    rng=np.random.default_rng(seed)
    X=rng.normal(size=(n,d))
    y=np.where(X[:,0]*X[:,1]+.3*X[:,2]+.15*rng.normal(size=n)>0,1,-1)
    return X,y


def profile_baseline(out):
    X,y=synthetic(480,12)
    model=KernelSVM(kernel="rbf",gamma=.1,working_set="first_order",cache="none",tol=PROTOCOL["tol"])
    pr=cProfile.Profile();pr.enable();model.fit(X,y);pr.disable()
    pr.dump_stats(str(out/"baseline_profile.prof"))
    stream=io.StringIO();stats=pstats.Stats(pr,stream=stream).sort_stats("cumulative");stats.print_stats(25)
    (out/"baseline_profile.txt").write_text(stream.getvalue(),encoding="utf-8")
    functions=[]
    for (filename,line,name),(cc,nc,tt,ct,callers) in stats.stats.items():
        functions.append({"function":name,"file":Path(filename).name,"line":line,
                          "calls":nc,"self_seconds":tt,"cumulative_seconds":ct})
    functions.sort(key=lambda x:x["self_seconds"],reverse=True)
    record={"scope":"cProfile baseline only; profiled timings excluded from benchmark",
            "input_shape":list(X.shape),"array_sha256":array_hash(X,y),
            "config":model.get_params(),"diagnostics":model.diagnostics_,
            "kernel_stats":model.kernel_stats_,"top_self_time":functions[:20]}
    write_json(out/"profile_summary.json",record)
    return record


def benchmark(out,quick=False):
    rows=[];histories=[]
    configs=[("first_none",dict(working_set="first_order",cache="none")),
             ("second_none",dict(working_set="second_order",cache="none")),
             ("second_lru_small",dict(working_set="second_order",cache="lru",cache_mb=.05)),
             ("second_lru",dict(working_set="second_order",cache="lru",cache_mb=16)),
             ("second_full",dict(working_set="second_order",cache="full"))]
    shapes=[(200,8),(480,12),(800,16)] if not quick else [(200,8)]
    repeats=1 if quick else PROTOCOL["timing_repeats"]
    for n,d in shapes:
        X,y=synthetic(n,d);Q,_=synthetic(64,d,seed=53)
        # One explicit warmup per configuration, excluded from records.
        for _,kwargs in configs:
            KernelSVM(gamma=1/d,tol=PROTOCOL["tol"],**kwargs).fit(X,y)
        reference=None
        for repeat in range(repeats):
            order=configs[repeat:]+configs[:repeat]
            for label,kwargs in order:
                start=time.perf_counter()
                model=KernelSVM(gamma=1/d,tol=PROTOCOL["tol"],**kwargs).fit(X,y)
                fit_seconds=time.perf_counter()-start
                start=time.perf_counter();scores=model.decision_function(Q);pred_seconds=time.perf_counter()-start
                if reference is None:reference=scores.copy()
                max_diff=float(np.max(np.abs(scores-reference)))
                if not model.converged_ or max_diff>2e-4:
                    raise RuntimeError(f"benchmark certificate/reference failed: {n},{label},{model.diagnostics_},{max_diff}")
                record={"n":n,"d":d,"configuration":label,"repeat":repeat,
                        "C":1,"gamma":1/d,"tol":PROTOCOL["tol"],"fit_seconds":fit_seconds,
                        "predict_64_seconds":pred_seconds,"max_score_diff":max_diff,
                        "array_sha256":array_hash(X,y),**model.diagnostics_,**model.kernel_stats_}
                rows.append(record)
                if repeat==0:
                    for h in model.history_: histories.append({"n":n,"configuration":label,**h})
        table(out/"benchmark_raw.csv",rows)
        print(f"benchmark n={n}, d={d}: {len(configs)*repeats} certified fits",flush=True)
    summaries=[]
    for n,d in shapes:
        baseline=np.median([r["fit_seconds"] for r in rows if r["n"]==n and r["configuration"]=="first_none"])
        for label,_ in configs:
            group=[r for r in rows if r["n"]==n and r["configuration"]==label]
            times=[r["fit_seconds"] for r in group]
            summaries.append({"n":n,"d":d,"configuration":label,"repeats":len(group),
                              "median_seconds":float(np.median(times)),"min_seconds":min(times),"max_seconds":max(times),
                              "baseline_over_this_median":float(baseline/np.median(times)),
                              "iterations":group[0]["iterations"],"computed_rows":group[0]["computed_rows"],
                              "cache_buffer_bytes":group[0]["cache_buffer_bytes"],
                              "max_pair_gap":max(r["pair_gap"] for r in group),
                              "max_score_diff":max(r["max_score_diff"] for r in group)})
    table(out/"benchmark_summary.csv",summaries);table(out/"convergence.csv",histories)
    return summaries


def real_experiments(out,quick=False):
    from sklearn.svm import SVC
    X,y,source=load_wdbc();write_json(out/"data_source.json",source)
    candidates=[];results=[];predictions=[];split_records={};selected_records=[];ablations=[]
    seeds=PROTOCOL["split_seeds"][:1] if quick else PROTOCOL["split_seeds"]
    for seed in seeds:
        parts=split_indices(y,seed)
        split_records[str(seed)]={k:v.tolist() for k,v in parts.items()}
        train,val,test=(parts[k] for k in ["train","validation","test"])
        prep=Standardizer().fit(X[train]);tr,va=prep.transform(X[train]),prep.transform(X[val])
        best={}
        for kernel in ["linear","rbf"]:
            best_score=-np.inf
            for C in PROTOCOL["C_grid"]:
                for gamma in ([.1] if kernel=="linear" else PROTOCOL["gamma_grid"]):
                    model=KernelSVM(C=C,kernel=kernel,gamma=gamma,tol=PROTOCOL["tol"]).fit(tr,y[train])
                    m=metrics(y[val],model.decision_function(va))
                    candidates.append({"seed":seed,"model":kernel,"C":C,"gamma":gamma,
                                       "validation_balanced_accuracy":m["balanced_accuracy"],
                                       "validation_roc_auc":m["roc_auc"],**model.diagnostics_})
                    if model.converged_ and m["balanced_accuracy"]>best_score:
                        best_score=m["balanced_accuracy"];best[kernel]={"C":C,"gamma":gamma}
            if kernel not in best: raise RuntimeError("no converged validation candidate")
        best_log=None;best_score=-np.inf
        for C in PROTOCOL["C_grid"]:
            log,log_state=fit_logistic(C,tr,y[train])
            m=metrics(y[val],log.decision_function(va))
            candidates.append({"seed":seed,"model":"logistic","C":C,
                               "validation_balanced_accuracy":m["balanced_accuracy"],"validation_roc_auc":m["roc_auc"],**log_state})
            if log_state["converged"] and m["balanced_accuracy"]>best_score:best_score=m["balanced_accuracy"];best_log=C
        if best_log is None: raise RuntimeError("no converged logistic validation candidate")
        trainval=np.r_[train,val];scaler=Standardizer().fit(X[trainval])
        full,te=scaler.transform(X[trainval]),scaler.transform(X[test])
        selected_records.append({"seed":seed,"linear":best["linear"],"rbf":best["rbf"],"logistic_C":best_log,
                                 "split_counts":{k:len(v) for k,v in parts.items()}})
        score=np.full(len(test),1. if np.count_nonzero(y[trainval]==1)>=np.count_nonzero(y[trainval]==-1) else -1.)
        results.append({"seed":seed,"model":"majority",**metrics(y[test],score),"scope":"heldout_test"})
        start=time.perf_counter();log,log_state=fit_logistic(best_log,full,y[trainval]);seconds=time.perf_counter()-start
        if not log_state["converged"]:raise RuntimeError("final logistic baseline did not report convergence")
        results.append({"seed":seed,"model":"logistic",**metrics(y[test],log.decision_function(te)),
                        "fit_seconds_single_run":seconds,"scope":"heldout_test",**log_state})
        for kernel in ["linear","rbf"]:
            start=time.perf_counter()
            model=KernelSVM(kernel=kernel,tol=PROTOCOL["tol"],**best[kernel]).fit(full,y[trainval])
            seconds=time.perf_counter()-start
            if not model.converged_: raise RuntimeError("selected real-data fit did not converge")
            scores=model.decision_function(te)
            ref=SVC(kernel=kernel,tol=1e-8,shrinking=False,**best[kernel]).fit(full,y[trainval])
            ref_scores=ref.decision_function(te)
            max_diff=float(np.max(np.abs(scores-ref_scores)))
            agree=float(np.mean((scores>=0)==(ref_scores>=0)))
            if max_diff>2e-3: raise RuntimeError(f"real reference score mismatch {max_diff}")
            results.append({"seed":seed,"model":"own_"+kernel,**metrics(y[test],scores),
                            "fit_seconds_single_run":seconds,"scope":"heldout_test",
                            "reference_max_score_diff":max_diff,"reference_prediction_agreement":agree,
                            **model.diagnostics_})
            results.append({"seed":seed,"model":"libsvm_"+kernel,**metrics(y[test],ref_scores),
                            "scope":"same_selected_parameters_independent_reference"})
            save_bundle(out/"models"/f"{kernel}_seed{seed}.npz",model,scaler)
            for idx,truth,s in zip(test,y[test],scores):
                predictions.append({"seed":seed,"model":kernel,"source_row":int(idx),"truth":int(truth),
                                    "decision":float(s),"prediction":1 if s>=0 else -1,
                                    "wrong":bool((1 if s>=0 else -1)!=truth)})
            if seed==2026 and kernel=="rbf":
                for scale in [True,False]:
                    raw=KernelSVM(kernel="rbf",tol=PROTOCOL["tol"],**best[kernel]).fit(full if scale else X[trainval],y[trainval])
                    ablations.append({"seed":seed,"standardized":scale,"C":best[kernel]["C"],"gamma":best[kernel]["gamma"],
                                      **metrics(y[test],raw.decision_function(te if scale else X[test])),
                                      "training_accuracy":float(np.mean(raw.predict(full if scale else X[trainval])==y[trainval])),
                                      **raw.diagnostics_,"scope":"fixed standardized-selected hyperparameters; not retuned raw baseline"})
        table(out/"validation_candidates.csv",candidates)
        write_json(out/"test_results.json",results)
        table(out/"test_predictions.csv",predictions)
        print(f"WDBC split {seed}: validation-selected models, heldout test and references complete",flush=True)
    write_json(out/"splits.json",split_records);write_json(out/"selected_parameters.json",selected_records)
    write_json(out/"scaling_ablation.json",ablations)
    return results


def source_manifest(root):
    files=[root/"run.py"]+list((root/"src").rglob("*.py"))+list((root/"tests").rglob("*.py"))
    return {str(p.relative_to(root)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def run_experiments(root,quick=False):
    run_id="run_"+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:6]
    out=root/("smoke" if quick else "results")/run_id;out.mkdir(parents=True,exist_ok=False)
    source_before=source_manifest(root)
    write_json(out/"protocol.json",{**PROTOCOL,"scope":"smoke" if quick else "formal", "quick":quick,
                                    "effective_split_seeds":PROTOCOL["split_seeds"][:1] if quick else PROTOCOL["split_seeds"],
                                    "effective_timing_repeats":1 if quick else PROTOCOL["timing_repeats"]})
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        write_json(out/"environment.json",environment())
        print("Baseline profiling (separate from timing benchmark)",flush=True)
        profile=profile_baseline(out)
        write_json(out/"optimization_decision.json",{
            "evidence":"baseline_profile.txt and profile_summary.json produced before benchmark",
            "measured_computed_rows":profile["kernel_stats"]["computed_rows"],
            "next":"Compare working-set iteration reduction, then cache repeated row computations; preserve losing configurations",
            "cpp_decision":"No C++ rewrite without evidence it benefits this scoped workload"})
        bench=benchmark(out,quick)
        real=real_experiments(out,quick)
    from .report import generate
    generate(out)
    if source_before!=source_manifest(root):
        raise RuntimeError("source files changed during experiment; run cannot be finalized")
    manifest={"scope":"smoke" if quick else "formal", "status":"completed",
              "wall_seconds_full_pipeline_not_speedup":time.perf_counter()-start,
              "source_files":source_before,"benchmark_groups":len(bench),"real_result_rows":len(real),
              "artifact_files":{str(p.relative_to(out)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(out.rglob("*")) if p.is_file()}}
    write_json(out/"manifest.json",manifest)
    write_json(root/("smoke" if quick else "results")/"latest.json",{"run_directory":str(out.relative_to(root)).replace("\\","/"),"scope":manifest["scope"]})
    return out
