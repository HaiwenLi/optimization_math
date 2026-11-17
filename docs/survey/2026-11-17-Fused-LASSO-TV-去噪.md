# Fused LASSO / Total Variation 去噪与去卷积 调研

- **日期**：2026-11-17
- **状态**：调研（T3，P0）
- **归属**：总路线图 `docs/survey/2026-11-17-算法调研扩展总路线图.md` 的 T3

## 1. 问题形式与记号

一维融合 LASSO（fused lasso）/ TV 去噪：

\[
\min_{x\in\mathbb{R}^n}\quad \tfrac12\|x-y\|_2^2 + \lambda_{\rm sp}\|x\|_1 + \lambda_{\rm tv}\|Dx\|_1,
\]

其中 `D ∈ R^{(n-1)×n}` 为一阶差分 `(Dx)_i = x_{i+1} − x_i`（本文聚焦 1D；2D 各向异性 TV 是各行/列差分的直和，接口不变）。`λ_sp ≥ 0` 控稀疏、`λ_tv ≥ 0` 控分段常数。纯 TV 去噪取 `λ_sp = 0`；fused lasso 取 `λ_sp, λ_tv > 0`。

去卷积变体：`min ½‖Ax−y‖² + λ_tv‖Dx‖₁`（A 为卷积/模糊算子），x 步变成 `(AᵀA + ρDᵀD)x = Aᵀy + ρDᵀ(z−u)`。本文先落地纯 TV 去噪与 fused lasso，去卷积可复用同一框架（换 A）。

## 2. 最优性证书

### 2.1 KKT 条件

目标 `f(x) = ½‖x−y‖² + λ_sp‖x‖₁ + λ_tv‖Dx‖₁` 的次梯度（无 box）：

\[
0 \in (x-y) + \lambda_{\rm sp}\partial\|x\|_1 + D^T(\lambda_{\rm tv}\,\partial\|Dx\|_1).
\]

引入中间变量 `z = Dx`，则最优性由以下分裂 KKT 刻画：

\[
\begin{cases}
x - y + \lambda_{\rm sp} v_x + D^T \nu = 0, & v_x \in \partial\|x\|_1,\\
\nu = \lambda_{\rm tv}\, v_z, & v_z \in \partial\|z\|_1,\ z = Dx.
\end{cases}
\]

**归一化 KKT 违反度**（工程口径，对 x 的次梯度做归一）：

\[
g = (x-y) + \lambda_{\rm sp} v_x + \lambda_{\rm tv} D^T \operatorname{sign}(Dx),
\qquad
\varepsilon_{\rm KKT} = \frac{\max_{i} \operatorname{dist}(g_i,\ \ldots)}
{\max(1,\|g\|_\infty)}.
\]

具体地：对 `x_i ≠ 0` 要求 `g_i = 0`（sparsity 的 active 分量被 `sign(x_i)` 吸收），对 `x_i = 0` 要求 `|(x-y)_i + λ_tv(Dᵀ sign(Dx))_i| ≤ λ_sp`。实现按此分支直接计算最大化违反度并归一化。

### 2.2 对偶与原始-对偶间隙（Chambolle）

TV 去噪 `min_x ½‖x−y‖² + λ_tv‖Dx‖₁` 的 Fenchel 对偶为

\[
\max_{p:\ \|p\|_\infty\le 1}\ \tfrac12\|y\|_2^2 - \tfrac12\|y - \lambda_{\rm tv}D^Tp\|_2^2.
\]

（对偶变量 `p ∈ R^{n-1}`，约束是 `‖p‖∞≤1`，**不是** `‖Dᵀp‖∞`——这与 LASSO 对偶不同，因 TV 项是 `‖Dx‖₁` 而非 `‖x‖₁`。）

给定迭代点 `x` 与其对偶 `p`（`‖p‖∞≤1`），原始-对偶（Fenchel）间隙：

\[
\mathrm{gap} = \underbrace{\tfrac12\|x-y\|^2+\lambda_{\rm tv}\|Dx\|_1}_{p}
-\underbrace{\left(\tfrac12\|y\|^2-\tfrac12\|y-\lambda_{\rm tv}D^Tp\|^2\right)}_{d}
\]
再按 `max(1,|p|,|d|)` 归一化。Chambolle 投影算法天然同时产出 `(x, p)`，且其迭代保证 `gap→0`——这是 1D TV 最干净、最紧的证书，无需在平台区（`Dx=0`）单独构造次梯度。

> **平台区次梯度的坑**：若用「`g = (x−y)+λ_tv Dᵀsign(Dx)` 的范数」当 KKT 残差，会把 `sign(0)=0` 误当次梯度，导致平台区（分段常数间的大段 `Dx=0`）处 KKT 永远不归零。Chambolle 的对偶 `p` 在平台区自动取到次梯度的正确内点，故 gap 才是 TV 的正确证书。

## 3. 算法谱

| 算法 | 核心迭代 | 优点 | 缺点 | 与本仓库关系 |
|---|---|---|---|---|
| 分裂 ADMM（split Bregman） | x 步 `(I+ρDᵀD)x = y+ρDᵀ(z−u)`（三对角，Thomas 解）、z 步软阈值 | 每步 O(n)、z 闭式、收敛稳健 | 调 ρ | **推荐主推**；`DᵀD` 三对角，x 步可用 Thomas/CG |
| 近端梯度（ISTA/FISTA） | TV 近端（1D 用 taut-string / 动态规划） | 无内层求解 | TV prox 非平凡（2D 尤甚） | 对照 |
| 主-对偶（Chambolle–Pock） | 交错梯度 + 对偶投影 | 无逆算子、步长易选 | 实现中等 | 对照/后续 |

**选型**：1D 主推 **Chambolle 投影算法**（对偶投影，`p ← (p+τD(y−λ_tv Dᵀp))/(1+τ|·|)`，τ≤1/(4λ_tv)），因为它是 1D TV 的标准算法、每步 O(n)、且**天然产出紧的对偶间隙**（§2.2）。ADMM/split Bregman 作为对照（x 步三对角 Thomas），去卷积变体 x 步换 `(AᵀA+ρDᵀD)` 用 CG（复用 `_cg_ata`）。

## 4. 自建案例（真值已知）

1. **阶梯/分段常数信号**：真值 `x_true` 为分段常数（如 `[0,0,0,1,1,1,0,0,2,2]` 延展），观测 `y = x_true + noise`，验证 TV 去噪恢复分段结构。指标：与真值相对 ℓ²、TV 值、分段边界召回。
2. **过渡点检测**：真值的跳变位置已知，测 `λ_tv` 对边界恢复的 F1（把"差分非零处"当支撑）。
3. **对比**：`λ_tv=0`（纯噪声）→ 不平滑；`λ_tv` 适中 → 恢复分段常数；`λ_tv` 过大 → 过度平滑到常数。三者用「与真值 ℓ² 误差」自动判定最优 λ_tv 区间。

## 5. 参考文献

1. Tibshirani, R., Saunders, M., Rosset, S., Zhu, J., & Knight, K. (2005). Sparsity and smoothness via the fused lasso. *JRSS-B*, 67(1), 91–108.
2. Rudin, L., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. *Physica D*, 60, 259–268.
3. Goldstein, T., & Osher, S. (2009). The split Bregman method for L1-regularized problems. *SIAM J. Imaging Sci.*, 2(2), 323–343.
4. Chambolle, A., & Pock, T. (2011). A first-order primal-dual algorithm for convex problems with applications to imaging. *J. Math. Imaging Vis.*, 40, 120–145.

## 6. 落地接口（规划）

- 差分算子：`src/optimization_math/operators_difference.py` → `first_difference_operator(n)`（`shape=(n-1,n)`，`matvec/rmatvec`）。
- 求解器+证书：`src/optimization_math/solvers/chambolle_tv.py` → `solve_tv_chambolle(y, lam_tv)`（Chambolle 投影，内建原始-对偶 gap 证书）。
- 测试与基准：`tests/test_tv.py`；`benchmarks/bench_tv.py`。

> 说明：本类型聚焦纯 TV 去噪（`λ_sp=0`）。fused lasso（`λ_sp>0`）需要在 `z=Dx` 之外再加一个 `z_x=x` 的稀疏分裂，作为后续扩展；去卷积变体同理留待后续（换 `A` 即可复用框架）。
