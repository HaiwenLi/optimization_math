# GAP-safe 筛选 与 主-对偶（Chambolle–Pock）—— 落地实现调研

- **日期**：2026-11-17
- **状态**：已落地（代码 + 证书 + 基准）
- **定位**：在「进阶算法族综述」选型结论中，按投入产出比前两位的两个算法族落地：**safe screening**（降维加速，叠加于现有 FISTA）与 **PDHG 主-对偶**（补上主-对偶范式）。本文记录其原理、实现口径、与本地框架的衔接，以及与现有 FISTA/ADMM 的实测对比结论。

---

## 1. GAP-safe screening

### 1.1 原理

给 `min_x ½‖Ax+b−y‖² + λ‖x‖₁ (+ μ/2‖x‖²)` 构造对偶下界，据此**安全地**剔除最优解中必为 0 的坐标：

\[
|a_i^\top \nu| + \|a_i\| \cdot r_{\mathrm{ball}} < \lambda \;\Rightarrow\; x_i^\star = 0,
\]

其中 `ν = θ·r` 是**对偶可行点**（由残差缩放得来，`θ = min(1, λ/‖q‖∞)`），`r_ball` 是以对偶间隙界定的安全球半径：

\[
r_{\mathrm{ball}} = \sqrt{2\,\mathrm{gap}},
\qquad \mathrm{gap} = P(x) - D(\nu).
\]

安全性来源：强对偶下，由于对偶 `D(ν) = −½‖ν‖²+⟨b−y,ν⟩` 是 1-强凹（`∇²D = −I`），`gap` 是 `½‖ν−ν*‖²` 的上界，故最优对偶 `ν*` 落在以 `ν` 为心、`r_ball` 为半径的球内；若球与该坐标的对偶约束 `|a_iᵀν| ≤ λ` 无交集，则该坐标必为 0。注意 **`ν = θr` 本身即对偶可行点**，强凹界对任意对偶可行 `ν` 成立，故半径**不含 θ 因子**。**被剔除的坐标保证不会误删**——screening 只加速，不改变最优解。

### 1.2 实现口径（`src/optimization_math/screening.py`）

- `column_norms_squared(op)`：逐列 `‖a_i‖² = e_iᵀ AᵀA e_i`，算子无关（对 dense 矩阵与 `LinearOperator` 都成立）。
- `compute_safe_screen(A, y, x, lam, *, mu, b, gap, theta)`：返回**安全剔除的布尔掩码**。`gap`（对偶间隙）复用 `compute_certificate_signed` 的 `primal − dual`；`θ`/`ν` 在函数内重算（`Certificate` dataclass 不含 `theta` 字段），但与证书用的是**同一套对偶构造**，语义零漂移。

> 关键复用点：`dual`（进而 gap）从证书直接拿到，避免另起对偶炉灶；`θ` 与列范数是 screening 特有的量，故在此补算。

> **范围限制（本次落地明确）**：GAP-safe 半径 `r_ball=√(2·gap)` 依赖纯 LASSO 的对偶结构（`D` 为 1-强凹，`∇²D=−I`），**仅对 μ=0 成立**。弹性网的 μ‖x‖² 会扭曲对偶，`r_ball` 与 `‖ν−ν*‖²` 的 bound 关系不再成立，需要专门的 elastic-net screening 规则（Ndiaye 2017 §6 有给）。故 `compute_safe_screen` 对 `mu>0` 显式抛 `NotImplementedError`，**不静默夸大安全性保证**——这是安全筛选的底线。

### 1.3 实测结论（`benchmarks/bench_screening.py`）

合成稀疏问题（`n=200, k=15, m=80`，5 个噪声种子）：

| seed | status | 最终 rel.gap | 安全剔除比例 | 最优支撑保持 |
|---|---|---|---|---|
| 0–4 | converged | ~5–10×10⁻⁹ | **0.92–0.95** | 全部 True |

- **安全性**：所有 seed 上，被筛选坐标都不含任何**最优解非零**分量（hard 保证）。
- **有效性**：收敛时 gap 极小时，可安全剔除 92–95% 的坐标，降维显著。

> **口径澄清**（本次落地最重要的教训）：screening 的安全契约是「被剔除坐标在**最优解**中为 0」，而非「在生成真值 `x_true` 中为 0」。LASSO 会正确地压掉被噪声淹没的弱真值坐标（`x_true[i]≠0` 但 `x_opt[i]=0`），这不是 screening 的误删。基准与测试的断言参照必须用 `x_opt` 的支撑，而非 `x_true` 的支撑。

### 1.4 关键文献

- Fercoq, O., Gramfort, A., & Salmon, J. (2015). Mind the duality gap: safer rules for the lasso. *ICML*, 333–342.
- Ndiaye, E., Fercoq, O., Gramfort, A., & Salmon, J. (2017). Gap safe screening rules for sparsity enforcing penalties. *JMLR*, 18(128), 1–33.

---

## 2. PDHG（Chambolle–Pock 主-对偶）

### 2.1 原理

将 LASSO 写成鞍点 `min_x max_z ⟨Ax,z⟩ + G(x) − F*(z)`（`G = λ‖x‖₁ + ½μ‖x‖²`，`F(z)=½‖z+b−y‖²`，`F*(z)=½‖z‖²+⟨z,y−b⟩`），迭代（θ=1）：

\[
\begin{aligned}
z^{n+1} &= \operatorname{prox}_{\sigma F^\*}(z^n + \sigma A \bar x^n),\\
x^{n+1} &= \operatorname{prox}_{\tau G}(x^n - \tau A^\top z^{n+1}),\\
\bar x^{n+1} &= x^{n+1} + \theta(x^{n+1} - x^n),
\end{aligned}
\]

步长条件 `τσ‖A‖² ≤ 1`（θ=1）。核心价值：**主、对偶的近端分开做，无需解任何线性系统（逆算子）**——这是 FISTA（x 步要 `(I+τ∂G)` 隐式步）与 ADMM（子求解）都不具备的性质，对 TV、去卷积等「A 无显式逆」的场景是决定性优势。

### 2.2 实现口径（`src/optimization_math/solvers/pdhg.py`）

- 近端闭式：`prox_{σF*}(z) = (z − σ(y−b)) / (1+σ)`；`prox_{τG}(x) = sign(x)·max(|x|−τλ,0) / (1+τμ)`。
- 默认步长 `τ = σ = 1/√‖A‖²`（满足 θ=1 条件 `τσ‖A‖²≤1`；**μ 在 G 中，不进耦合 K=A，故不含 μ**），`Lipschitz` 复用 `_estimate_lipschitz`（含 ×1.05 safety factor）。
- 收敛**仍走证书门**：复用 `compute_certificate_signed`，只有 KKT + 对偶间隙同时达标才报 `converged`——与 FISTA 同一把尺子，保证横向可对比。

### 2.3 实测结论（`benchmarks/bench_pdhg.py`）

T1 相同已知真值问题，PDHG vs FISTA：

| solver | status | iters | A/Aᵀ calls | ε_KKT | rel.gap | rel.L2 | F1 |
|---|---|---|---|---|---|---|---|
| fista-signed | converged | 133 | 133 | 9.6e-6 | 9.3e-6 | 0.250 | 0.720 |
| pdhg | converged | 197 | 197 | 8.8e-6 | 5.2e-6 | 0.250 | 0.720 |

- **收敛到同一最优解**（objective、rel.L2、F1 完全一致），证书统一验收成立。
- 本问题上 PDHG 步数更多（197 vs 133，两者每步都是 1 对 A/Aᵀ，FISTA 另有回溯的额外 matvec），**成本不占优**——这符合预期：普通病态不强的 LASSO 是 FISTA 的主场。
- **PDHG 的价值不在这种问题**，而在：① 无逆算子（TV/去卷积）；② saddle-point 约束结构。这印证了综述 §7 的选型表——**不是"全能更优"，而是"各有主场"**。

### 2.4 关键文献

- Chambolle, A., & Pock, T. (2011). A first-order primal-dual algorithm for convex problems with applications to imaging. *JMIV*, 40, 120–145.
- Chambolle, A., & Pock, T. (2016). On the ergodic convergence rates of a first-order primal-dual algorithm. *Math. Programming*, 159, 253–287.

---

## 3. 与本地框架的衔接小结

| 模块 | 文件 | 复用的现有资产 | 新增 |
|---|---|---|---|
| screening | `src/optimization_math/screening.py` | `compute_certificate_signed`（对偶点/θ/gap）、`as_linear_operator`、`_positive_float` | `column_norms_squared`、`compute_safe_screen` |
| pdhg | `src/optimization_math/solvers/pdhg.py` | `_estimate_lipschitz`、`_positive_int/float`、`compute_certificate_signed` | `solve_pdhg`、`PdhgResult`、两个近端闭式 |

两者都**未改动**任何既有求解器/证书，纯增量扩展，符合「算法无关的证书」架构原则。

---

## 4. 后续

- **screening 的下一步**：把 `compute_safe_screen` 真正接到求解器内环（在 FISTA 迭代中动态收缩活动集），而不只是"事后度量"。这是降维加速的实际收益来源，也是综述 §2 建议中"叠加加速器"的完整形态。第二步：实现 **elastic-net 的 safe screening**（μ>0 的修正规则，见 Ndiaye 2017 §6），解除当前 μ=0 的限制。
- **PDHG 的下一步**：在 2D TV / 去卷积问题（T3 的后续扩展）上实际验证其"无逆算子"优势，这是它相对 FISTA 的决胜场景。
- 其余算法族（坐标下降、OWL-QN/Prox-QN、SSNAL）仍按投入产出比排在后面，待前两者收益确认后落地。
