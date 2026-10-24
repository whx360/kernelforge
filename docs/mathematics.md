# 从最大间隔到可执行SMO

## 1. 问题与几何

训练数据(x_i,y_i)，y_i∈{-1,+1}，映射phi，截距b不正则化。软间隔原始问题：

P = min 1/2||w||² + C Σ xi_i，约束 y_i(w·phi(x_i)+b)≥1-xi_i，xi_i≥0。

函数间隔是y_i f(x_i)，几何间隔需要除||w||。C控制违反间隔的代价，不是学习率。hinge为max(0,1-yf)。

## 2. 对偶与KKT

对偶最大化 W(alpha)=Σalpha_i - 1/2(alpha⊙y)^T K(alpha⊙y)，约束0≤alpha_i≤C，y^Talpha=0。

K为PSD时这是凹二次优化；软间隔有严格可行点，可使用强对偶。最优原始和对偶值相等；有限容差下只报告间隙，不声称精确实数相等。

KKT：alpha=0要求yf≥1；0<alpha<C要求yf=1；alpha=C要求yf≤1。支持向量为alpha>0；alpha=C并不必然误分类。

## 3. 为什么更新两个变量

单独更新一个alpha通常破坏y^Talpha=0。令alpha_i←alpha_i+y_i t、alpha_j←alpha_j-y_j t，则等式始终保持。

定义f0=K(alpha⊙y)，F_i=y_i-f0_i。沿上述方向：

Delta W(t)=(F_i-F_j)t - (K_ii+K_jj-2K_ij)t²/2。

eta=K_ii+K_jj-2K_ij是特征空间两点平方距离。eta>0时无约束最优t=(F_i-F_j)/eta；随后截到[0,tmax]。eta=0且一阶增益为正时取可行端点。不能把重复点直接跳过，否则相反标签的重复点可能永不收敛。

tmax=min(cap_i,cap_j)：cap_i为沿+y_i方向到盒边界的距离，cap_j为沿-y_j方向的距离。

## 4. 工作集与停止

I_up：y=+1且alpha<C，或y=-1且alpha>0。

I_low：y=+1且alpha>0，或y=-1且alpha<C。

最大可行方向违反量pair_gap=max(F[I_up])-min(F[I_low])。当pair_gap≤tol时，已找不到超过容差的改进方向。b取两端中点；收敛时它近似满足margin条件。pair_gap不是单点margin残差，两者分别记录。

first_order：选最大发生违反的i和j。

second_order：固定同一个i，为每个j计算截断后的Delta W，选最大者。这借鉴二阶工作集思想，但不声称就是LIBSVM的完整WSS实现/理论。

## 5. 增量梯度、缓存与复杂度

一次更新使f0增加t(K[:,i]-K[:,j])，故F减少同样向量。利用核对称，只需两行核和O(n)向量更新。周期刷新消除累计浮点漂移。

核行linear/RBF时间O(nd)，输出O(n)；RBF使用直接差分和平方和，避免||x||²+||z||²-2x·z在大平移下的消减。full保存O(n²)核值；LRU保存最多r行O(rn)；none不跨调用缓存。工作集选择O(n)，总训练依赖实际更新次数T，不能宣称固定O(n)或O(n²)。none粗略O(Tnd)，另计刷新和监测；full构建O(n²d)，更新O(Tn)。

预测用支持向量，q个查询时间O(q*s*d)，分查询计算避免q×s×d三维临时数组。

## 6. 独立验证为何不能比较alpha逐位一致

重复点/线性相关时对偶最优解可能不唯一。比较可行性、对偶目标、KKT、decision与预测；不要求独立求解器alpha数组相等。小例用SciPy SLSQP求对偶作独立QP参考；较大例用sklearn SVC（LIBSVM）比较同目标。

## 7. 函数契约（实施前固定）

| 函数 | 输入/输出 | 复杂度/意义 |
| --- | --- | --- |
| KernelRows.row(i) | 样本索引→长度n核向量 | 未命中O(nd)；只读行，不允许外部修改缓存 |
| KernelSVM.fit(X,y) | 有限n×d、两类±1→self | 复制输入；训练状态不由外部数组生命周期决定 |
| diagnostics | alpha/F/K/b→诊断字典 | 原始/对偶/间隙/盒与等式残差，证明目标与约束的一致性 |
| decision_function(X) | q×d→q分数 | O(qsd)，空q合法；不输出概率 |
| predict(X) | q×d→±1 | score≥0取+1；预测与标签约定一致 |
| save/load | 已拟合模型↔NPZ | 无pickle；保留核/超参数/维数和有限系数 |

## 思想来源

- Platt 1998：[SMO原始论文](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/)。
- Fan, Chen, Lin 2005：[二阶工作集选择](https://jmlr.org/papers/v6/fan05a.html)。
- Chang与Lin：[LIBSVM官方说明](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)。

上述算法属于既有方法；本工程的价值是自实现、验证与受控测量。

## 8. 展开对偶推导与伪代码

拉格朗日函数：

L(w,b,ξ,alpha,mu)=1/2||w||²+CΣξ_i+Σalpha_i[1-ξ_i-y_i(w·phi_i+b)]-Σmu_i ξ_i。

alpha、mu≥0。对w、b、ξ求驻点分别给出w=Σalpha_i y_i phi_i、Σalpha_i y_i=0、C-alpha_i-mu_i=0。代回并消去w/b/ξ得到前面的W(alpha)。强对偶允许用最优对偶解恢复原始解；有限容差时仍要测量间隙。

增益推导：令delta的i位置为y_i t、j位置为-y_j t，则delta⊙y在i为t、j为-t。将alpha+delta代入二次型后，一次项是(F_i-F_j)t，二次项是-eta t²/2。将无约束抛物线顶点截在可行区间就是解析子问题解。

```text
validate finite X, y in {-1,+1}, both classes, positive C/gamma/tol
alpha = 0; F = y
repeat until update budget:
    if refresh interval: F = y - K @ (alpha*y)
    construct feasible I_up and I_low
    i = argmax F on I_up; j_first = argmin F on I_low
    if pair gap <= tol:
        recompute F directly, confirm gap, then stop as converged
    get row_i and curvature eta_j
    j = j_first OR argmax clipped predicted dual gain for fixed i
    t = min(gap / eta, pair bound), or endpoint for zero curvature
    alpha_i += y_i*t; alpha_j -= y_j*t
    F -= t*(row_i-row_j)
recompute F; produce all certificates and explicit budget status
recover b from KKT interval; keep all alpha>0 supports
```

这里无需学习率；C是约束/正则化参数，tol是数值求解精度。预测分数f(x)=Σalpha_i y_i K(x_i,x)+b，不是概率。
