# 结构化稀疏：group LASSO 与核范数低秩 调研

- **日期**：2026-11-17
- **状态**：调研（T5，P1）
- **归属**：总路线图 `docs/survey/2026-11-17-算法调研扩展总路线图.md` 的 T5

## 1. 问题形式与记号

**group LASSO**：变量 `x` 分成 G 个互不相交的组 `{x_{g}}`，用 ℓ₂ 范数对每组做联合稀疏：

当前求解器与证书要求各组为非空整数索引，且**每个坐标恰好出现一次**。重叠组、遗漏坐标、重复或非整数索引会显式报错，不静默求解另一个问题。

\[
\min_{x}\ \tfrac12\|Ax-y\|_2^2+\lambda\sum_{g=1}^{G}\|x_{(g)}\|_2.
\]

（常规 LASSO 是每组只含一个坐标的特例。）

**核范数低秩**（矩阵变量）：观测算子 `𝒜: ℝ^{m×n}→ℝ^p` 线性，观测 `y=𝒜(X)+η`，恢复低秩 X：

\[
\min_X\ \tfrac12\|\mathcal A(X)-y\|_2^2+\lambda\|X\|_*,\qquad \|X\|_*=\sum_i\sigma_i(X).
\]

## 2. 最优性证书

### 2.1 group LASSO KKT

梯度 `g = Aᵀ(Ax−y)`，按组分块 `g_{(g)}`。次梯度条件 `0 ∈ g + λ ∂‖x_{(g)}‖₂`：

\[
\begin{cases}
g_{(g)} = -\lambda\,\dfrac{x_{(g)}}{\|x_{(g)}\|_2}, & \|x_{(g)}\|_2\neq0,\\
\|g_{(g)}\|_2 \le \lambda, & \|x_{(g)}\|_2=0.
\end{cases}
\]

归一化违反度：

\[
\varepsilon_{\rm KKT}=\frac{\max_g \operatorname{viol}_g}{\max(\lambda,\ \max_g\|g_{(g)}\|_2)},\quad
\operatorname{viol}_g=\begin{cases}
\Big\|g_{(g)}+\lambda\frac{x_{(g)}}{\|x_{(g)}\|_2}\Big\|_2, & x_{(g)}\neq0,\\
\max(0,\|g_{(g)}\|_2-\lambda), & x_{(g)}=0.
\end{cases}
\]

### 2.2 group LASSO 对偶

与 LASSO 同构，约束换成逐组：`max_ν { −½‖ν‖² − νᵀy : ‖(Aᵀν)_{(g)}‖₂ ≤ λ, ∀g }`。给定迭代 `x`、残差 `r=Ax−y`，取 `ν=−θ r`，`θ = min(1, λ / max_g ‖(Aᵀr)_{(g)}‖₂)`：

\[
d(\theta)=-\tfrac12\theta^2\|r\|_2^2-\theta y^T r,\quad
\mathrm{gap}=\frac{p-d}{\max(1,|p|,|d|)}.
\]

### 2.3 核范数 KKT 与谱对偶

核范数次梯度：`∂‖X‖_* = {U_r V_rᵀ + W : U_rᵀW=0, WV_r=0, ‖W‖₂≤1}`，其中 `U_r,V_r` 只取非零奇异值对应的子空间。KKT 为 `𝒜*(𝒜(X)−y) + λ(U_r V_rᵀ+W)=0`。对偶约束为 `‖𝒜*(ν)‖₂ ≤ λ`。实现上核范数近端是 SVD 奇异值软阈值 `U diag(max(σ−λ,0)) Vᵀ`；证书的 stationarity 字段实际采用归一化 Frobenius 近端不动点残差 `‖X−S_λ(X−grad)‖_F/max(1,‖X‖_F)`，并配谱可行对偶 gap，而不是直接忽略 `W` 的谱 KKT。

## 3. 算法谱

| 问题 | 主推算法 | 近端/步 | 证书 |
|---|---|---|---|
| group LASSO | 近端梯度 FISTA | `x_{(g)}·max(0,1−λ/‖x_{(g)}‖₂)` | §2.1 KKT + §2.2 gap |
| 核范数低秩 | 近端梯度（SVD 软阈值） | `S_λ(X)=U diag(max(σ−λ,0)) Vᵀ` | 谱 KKT + 谱对偶 gap |

两类的**近端都有闭式**，故统一用 FISTA 骨架（复用现有 momentum/backtracking），仅换 prox 与证书。

## 4. 自建案例（真值已知）

1. **group LASSO 分组稀疏**：真值 `x_true` 的支撑集中在一个组内（组内共激活），验证 group 联合选择效应；对比普通 LASSO（会把组内坐标零散拉回）。
2. **group 解析锚点**：单组 `A=I`，组内两坐标，`‖x‖₂` 正则的闭式解 `x = y·max(0,1−λ/‖y‖₂)`。
3. **核范数低秩恢复**：真值 `X_true` 秩 r，观测 `y=𝒜(X_true)+η`，验证奇异值软阈值恢复秩 r；矩阵补全/随机投影取 𝒜 为元素采样或高斯投影。

## 5. 参考文献

1. Yuan, M., & Lin, Y. (2006). Model selection and estimation in regression with grouped variables. *JRSS-B*, 68(1), 49–67.
2. Candès, E. J., & Recht, B. (2009). Exact matrix completion via convex optimization. *Found. Comput. Math.*, 9(6), 717–772.
3. Beck, A., & Teboulle, M. (2009). FISTA. *SIAM J. Imaging Sci.*, 2(1), 183–202.

## 6. 落地接口（规划）

- 证书模块：`src/optimization_math/certificates_group.py` → `compute_certificate_group(A, y, x, lam, groups)`。
- 求解器：`src/optimization_math/solvers/fista_group.py` → `solve_fista_group(...)`。
- 核范数（扩展）：`prox_nuclear` + 谱证书（本类先落地 group LASSO，核范数作后续或同一模块的矩阵版）。
- 测试与基准：`tests/test_group.py`；`benchmarks/bench_group.py`。

## 7. 基准结果（自建案例，真值已知）

### group LASSO（组联合选择 vs 普通 LASSO）

真值：`n=30` 分 10 组（每组 3 坐标），仅组 0 活跃；观测 `y = A x_true + 0.01·noise`，`λ=0.1`。

| 方法 | 组一致性 | 活跃组误差 | 迭代 |
|---|---|---|---|
| group LASSO | **0.50** | **0.01** | 651 |
| 普通 LASSO | 0.00 | 1.95 | 2235 |

**结论**：group LASSO 因 ℓ₂ 组范数实现**组联合选择**（组内坐标共激活/共稀疏），活跃组误差 0.01 远优于普通 LASSO 的 1.95（普通 LASSO 把组内坐标零散拉回，破坏了组结构）。

### 核范数低秩

真值 `X_true` 秩 3（`m=n=12`，奇异值 17.3/6.8/4.4），观测加 `σ=0.05` 噪声（尾部噪声奇异值 ~0.29）。

- **最优降噪 λ=0.1**：去噪误差 0.458 < 噪声 0.584（有降噪）。
- **λ≥0.5 恢复精确秩 3**（奇异值软阈值把噪声尾部压到 0）。

**结论**：核范数去噪存在「降噪 vs 秩准确」的 λ 权衡——λ 大则秩准但信号收缩，λ 小则降噪最优但尾部奇异值残留。二者需分开量化，指标（`rank_recovered` + `best_err`）分别判定。
