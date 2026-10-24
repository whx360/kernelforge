"""Local standalone HTML and scientific plots drawn from saved raw records."""
import base64
import csv
import html
import json
from pathlib import Path
import numpy as np


def read_csv(path):
    with Path(path).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))


def generate(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=Path(out);figdir=out/"figures";figdir.mkdir(exist_ok=True)
    bench=read_csv(out/"benchmark_summary.csv")
    convergence=read_csv(out/"convergence.csv")
    candidates=read_csv(out/"validation_candidates.csv")
    results=json.loads((out/"test_results.json").read_text(encoding="utf-8"))
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    profile=json.loads((out/"profile_summary.json").read_text(encoding="utf-8"))
    labels=list(dict.fromkeys(r["configuration"] for r in bench))
    sizes=sorted(set(int(r["n"]) for r in bench))
    colors=["#64748b","#3b82f6","#eab308","#14b8a6","#8b5cf6"]
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(10,4.8),layout="constrained")
    x=np.arange(len(sizes));width=.15
    for i,label in enumerate(labels):
        group=[next(r for r in bench if int(r["n"])==n and r["configuration"]==label) for n in sizes]
        values=np.array([float(r["median_seconds"])*1000 for r in group])
        lower=values-np.array([float(r["min_seconds"])*1000 for r in group])
        upper=np.array([float(r["max_seconds"])*1000 for r in group])-values
        ax.bar(x+(i-2)*width,values,width,color=colors[i],label=label,yerr=[lower,upper],capsize=3)
    ax.set_xticks(x,[f"n={n}, d={next(r['d'] for r in bench if int(r['n'])==n)}" for n in sizes])
    ax.set_ylabel("Fit wall time (ms), median and min/max")
    ax.set_title("Equal KKT tolerance; cache construction included; 1 BLAS thread")
    ax.legend(ncol=3,fontsize=8);fig.savefig(figdir/"benchmark.png",dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.3),layout="constrained")
    focus=480 if 480 in sizes else sizes[0]
    for label,color in [("first_none",colors[0]),("second_none",colors[1])]:
        group=[r for r in convergence if int(r["n"])==focus and r["configuration"]==label]
        ax.semilogy([int(r["iteration"]) for r in group],[max(1e-15,float(r["pair_gap"])) for r in group],label=label,color=color)
    ax.axhline(protocol["tol"],ls="--",color="#ef4444",label="stopping tolerance")
    ax.set_xlabel("Feasible pair updates");ax.set_ylabel("KKT pair gap")
    ax.set_title(f"Convergence on the same synthetic instance, n={focus}");ax.legend()
    fig.savefig(figdir/"convergence.png",dpi=160);plt.close(fig)

    rbf=[r for r in candidates if r["model"]=="rbf" and r["seed"]=="2026"]
    Cs=protocol["C_grid"];gammas=protocol["gamma_grid"]
    grid=np.array([[float(next(r["validation_balanced_accuracy"] for r in rbf if float(r["C"])==C and float(r["gamma"])==g)) for g in gammas] for C in Cs])
    fig,ax=plt.subplots(figsize=(6.4,4.7),layout="constrained")
    im=ax.imshow(grid,vmin=.5,vmax=1,cmap="YlGnBu")
    ax.set_xticks(range(len(gammas)),gammas);ax.set_yticks(range(len(Cs)),Cs)
    ax.set_xlabel("RBF gamma");ax.set_ylabel("C")
    ax.set_title("Validation balanced accuracy only, split seed 2026")
    for i in range(len(Cs)):
        for j in range(len(gammas)):ax.text(j,i,f"{grid[i,j]:.3f}",ha="center",va="center",color="white" if grid[i,j]>.86 else "black")
    fig.colorbar(im,ax=ax);fig.savefig(figdir/"validation.png",dpi=160);plt.close(fig)

    ablation=json.loads((out/"scaling_ablation.json").read_text(encoding="utf-8"))
    def fmt(value):
        return f"{value:.4f}" if isinstance(value,float) else str(value)
    main=[r for r in results if r["seed"]==2026]
    table_rows="".join("<tr>"+"".join(f"<td>{html.escape(fmt(r.get(k,'')))}</td>" for k in ["model","accuracy","balanced_accuracy","macro_f1","roc_auc"])+"</tr>" for r in main)
    benchmark_rows="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in [r["n"],r["configuration"],f"{float(r['median_seconds'])*1000:.2f}",f"{float(r['baseline_over_this_median']):.2f}",r["iterations"],r["computed_rows"]])+"</tr>" for r in bench)
    def img(name):return "data:image/png;base64,"+base64.b64encode((figdir/name).read_bytes()).decode()
    limits="仅一个569×30公共数值特征基准；三次分层划分反映划分波动，不是临床验证或显著性检验。速度仅适用于这里的合成规模、停止容差和本机环境；缓存字节不是进程峰值内存。LIBSVM是同参数独立参考，未声称本实现超过成熟库。"
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KernelForge · 实验报告</title>
<style>body{{margin:0;background:#f3f6fa;color:#152238;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1120px;margin:auto;padding:44px 24px}}h1{{font-size:36px;margin:4px 0}}h2{{margin-top:36px}}.tag{{color:#007f75;font-weight:700}}.card{{background:white;padding:24px;border:1px solid #dde5ef;border-radius:14px;margin:20px 0}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #e2e8f0;text-align:left}}.scroll{{overflow:auto}}small{{color:#526174}}a{{color:#1764b5}}code{{background:#eaf0f6;padding:2px 5px}}.warning{{border-left:4px solid #e0a52b;padding-left:16px}}</style>
<main><div class="tag">KernelForge / 自实现核 SVM / {html.escape(protocol['scope'])}</div><h1>正确性、优化与泛化的证据</h1>
<p>先证明求解器遵守数学契约，再测量工作集与核缓存。所有表格由本目录原始记录生成。</p>
<div class="card"><b>真实数据 · 固定 split seed 2026</b><p>验证集选C与gamma，合并训练验证后重新拟合标准化，再评估114个测试样本。恶性为+1；分数不是概率。</p><div class="scroll"><table><tr><th>模型</th><th>Accuracy</th><th>Balanced Acc.</th><th>Macro F1</th><th>ROC-AUC</th></tr>{table_rows}</table></div></div>
<h2>同精度性能比较</h2><p>fit墙钟包含缓存构建，正式运行三次轮换顺序；误差线为最小/最大值。比值为first-order/no-cache中位耗时除以本配置中位耗时，大于1表示本配置更快。</p>
<div class="card"><img alt="Benchmark timing" src="{img('benchmark.png')}"><div class="scroll"><table><tr><th>n</th><th>配置</th><th>中位ms</th><th>基线/本配置</th><th>更新数</th><th>计算核行</th></tr>{benchmark_rows}</table></div></div>
<div class="card"><img alt="KKT convergence" src="{img('convergence.png')}"></div>
<h2>模型选择与消融</h2><div class="card"><img alt="Validation-only hyperparameter matrix" src="{img('validation.png')}"><p>标准化对照使用相同、由标准化验证集选出的C/gamma；没有为raw输入重新调优，不代表raw输入的最优性能。</p><pre>{html.escape(json.dumps([{k:r[k] for k in ['standardized','training_accuracy','accuracy','roc_auc','converged']} for r in ablation],ensure_ascii=False,indent=2))}</pre></div>
<h2>边界与复核</h2><p class="warning">{limits}</p><p>基线profiling计算核行 {profile['kernel_stats']['computed_rows']} 次。详见<a href="baseline_profile.txt">剖析记录</a>、<a href="benchmark_raw.csv">原始计时</a>、<a href="test_predictions.csv">逐样本预测</a>、<a href="test_results.json">全部划分结果</a>、<a href="protocol.json">协议</a>。</p><small>方法源于Platt SMO与二阶工作集研究；数据源为sklearn随包WDBC。实现与测量属于本工程，既有算法不声称原创。</small></main></html>'''
    (out/"report.html").write_text(content,encoding="utf-8")
    lines=["# KernelForge 实验记录", "",f"范围：{protocol['scope']}；协议formal_v1；数值均从原始记录生成。", "", "## 主划分测试集", "", "| 模型 | Accuracy | balanced_accuracy | macroF1 | AUC |", "| --- | ---: | ---: | ---: | ---: |"]
    for r in main:lines.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['balanced_accuracy']:.4f} | {r['macro_f1']:.4f} | {r['roc_auc']:.4f} |")
    lines += ["", "## 同精度合成benchmark", "", "| n | 配置 | median ms | min/max ms | baseline/this | 更新数 | 核行计算 |", "| --- | --- | ---: | --- | ---: | ---: | ---: |"]
    for r in bench:lines.append(f"| {r['n']} | {r['configuration']} | {float(r['median_seconds'])*1000:.3f} | {float(r['min_seconds'])*1000:.3f}/{float(r['max_seconds'])*1000:.3f} | {float(r['baseline_over_this_median']):.3f} | {r['iterations']} | {r['computed_rows']} |")
    lines += ["", "## 三划分波动", "", "数值为不同划分的均值±总体标准差，不是同一测试集上的多次训练，也没有统计显著性声明。", ""]
    for label in ["own_linear","own_rbf","logistic","majority"]:
        group=[r for r in results if r["model"]==label]
        vals=np.array([r["balanced_accuracy"] for r in group]);auc=np.array([r["roc_auc"] for r in group])
        lines.append(f"- {label}: balanced_accuracy={vals.mean():.4f}±{vals.std():.4f}; AUC={auc.mean():.4f}±{auc.std():.4f}; splits={len(group)}")
    lines += ["", "## 解释边界", "", limits, "", "数据：data_source.json；划分：splits.json；选参：validation_candidates.csv、selected_parameters.json；错误分析：test_predictions.csv；停止状态：test_results.json；剖析：profile_summary.json。"]
    (out/"experiment_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def latest(root):
    pointer=json.loads((root/"results"/"latest.json").read_text(encoding="utf-8"))
    return root/pointer["run_directory"]
