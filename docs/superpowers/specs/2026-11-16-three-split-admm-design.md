# 三分裂 ADMM / Split Bregman 核心设计规格

- 日期：2026-11-16
- 状态：已按规格自查修正，获用户复核通过；策略为“核心先行”（已知背景 `b`，背景块联合估计另立规格）。核心三分裂 ADMM 已于 2026-11-16 实现并本地合并。
- 范围：非负 LASSO / 弹性网的三分裂 ADMM（split Bregman 同族）核心求解器，复用现有黑盒算子与收敛证书

## 1. 目标与边界

实现一个只依赖 NumPy 的 `solve_admm`，求解与 FISTA 相同的原问题

\[
\min_{x\ge 0}\quad \frac12\|Ax+b-y\|_2^2+\lambda\|x\|_1+\frac{\mu}{2}\|x\|_2^2,
\qquad \lambda>0,\ \mu\ge 0.
\]

本规格只覆盖 **已知背景 `b`** 的核心三分裂 ADMM：变量 `x` 承载数据项与弹性网二次项，变量 `z` 承载非负约束与 L1，约束为 `x=z`。P0 交付包括：

- `x` 步线性系统 `(AᵀA + (μ+ρ)I)x = Aᵀ(y-b) + ρ(z-u)` 的黑盒 CG 内层求解，不组装 `AᵀA`；
- `z` 步闭式非负软阈值 `z = max(0, x+u-λ/ρ)`；
- scaled dual 更新 `u ← u + x - z`；
- 归一化原始/对偶残差与现有 KKT/相对对偶间隙双证书共同停机；
- 可选自适应 `ρ`，更新时保持未缩放对偶变量连续；
- 迭代/残差/CG 计数历史与诚实 `max_iter` 状态；
- 与 FISTA 相同的确定性小算例验收和 pytest 测试。

明确不包括：未知背景 `b` 或背景块 `B c` 的联合估计、FFT 成像算子、285,696 体素基准、EF-Spin 集成、外部求解器 ground truth、混合正则/约束的泛化证书、工作列和支撑内 NNLS 去偏。split Bregman 不另做重复实现，按同一分裂家族由本求解器覆盖。

## 2. 方案选择

采用 **三分裂 ADMM + 黑盒 CG**，而不是再写一个 FISTA 变体或直接组装正规方程矩阵：

- 与现有算子层一致，`x` 步只需要 `matvec/rmatvec`；
- `z` 步闭式完成软阈值与非负投影，适合作为 FISTA 的独立交叉验证；
- `(μ+ρ)I` 移位使法方程强凸，零算子或病态 `A` 下 CG 系统仍然定义良好；
- 证书层不变，最终验收口径与 FISTA 完全一致；
- 背景块联合估计会改变变量结构和证书可行性构造，按用户确认推迟到单独规格。

所有求解状态使用一维 `float64` 数组。多维成像变量仍由具体算子在 `matvec/rmatvec` 内 reshape，不把成像维数泄漏到通用求解器接口。

## 3. 模块与接口

拟新增/调整：

```text
src/optimization_math/
├── certificates.py
├── operators.py
└── solvers/
    ├── __init__.py
    ├── _common.py       # 由 fista.py 提炼的共享校验助手（私有）
    ├── admm.py          # 新增
    └── fista.py         # 仅改为复用 _common.py，行为不变
tests/
├── test_admm.py         # 新增
└── test_fista.py        # 原则上不因重构改变断言
```

### 3.1 共享校验助手

将 `fista.py` 中现有的 `_positive_int`、`_positive_float`、`_validate_common` 原样提炼到私有模块 `solvers/_common.py`，供 FISTA 与 ADMM 复用。该重构不得改变 FISTA 的错误类型、错误信息口径和数值行为；现有 44 个测试应保持通过。

`_validate_common(op, y, lam, mu, b, x0)` 继续负责：`y/b/x0` 有限且形状匹配、`x0` 非负且复制、`lam>0`、`mu>=0`，返回 `(y, b_vec, x, lam, mu)`。

### 3.2 ADMM 求解器

```python
def solve_admm(
    A: LinearOperator | NDArray,
    y: NDArray,
    lam: float,
    *,
    mu: float = 0.0,
    b: NDArray | None = None,
    x0: NDArray | None = None,
    rho: float = 1.0,
    adaptive_rho: bool = False,
    rho_min: float = 1e-3,
    rho_max: float = 1e3,
    rho_balance: float = 10.0,
    rho_scale: float = 2.0,
    max_iter: int = 200,
    cg_max_iter: int | None = None,
    cg_tol: float = 1e-8,
    residual_tol: float = 1e-4,
    kkt_tol: float = 1e-5,
    gap_tol: float = 1e-4,
    certificate_stride: int = 10,
) -> AdmmResult: ...
```

`cg_max_iter=None` 时使用 `A` 的输入维数 `n` 作为 CG 上限；CG 仍按相对残差提前停止。该默认是精确算术意义上的安全上限，不代表大规模问题应该跑到该上限；预条件与 FFT 算子属于后续集成。

```python
@dataclass(frozen=True)
class AdmmIteration:
    iteration: int
    primal_residual: float
    dual_residual: float
    rho: float
    cg_iterations: int


@dataclass(frozen=True)
class AdmmResult:
    x: NDArray                 # 最终非负 z 分裂变量
    status: Literal["converged", "max_iter"]
    iterations: int
    rho: float                 # 最终使用的 rho
    cg_iterations: int         # 全部 x 步 CG 迭代总数
    certificate: Certificate   # 最终 z 的证书
    history: tuple[AdmmIteration, ...]      # 每次迭代一条
    certificates: tuple[Certificate, ...]   # 实际计算证书的迭代点
```

`result.x` 返回非负的 `z`，而不是可能尚未与 `z` 一致的 `x`。`Certificate.stage` 对单阶段 ADMM 固定为 0，`Certificate.iteration` 是 ADMM 迭代编号。

## 4. 算法

### 4.1 分裂与增广拉格朗日

引入约束 `x=z`，使用 scaled dual `u`。第 `k` 次迭代为：

\[
\begin{aligned}
x^{k+1} &= \arg\min_x \frac12\|Ax+b-y\|_2^2+\frac{\mu}{2}\|x\|_2^2
+\frac{\rho_k}{2}\|x-z^k+u^k\|_2^2,\\
z^{k+1} &= \arg\min_{z\ge 0}\ \lambda\|z\|_1
+\frac{\rho_k}{2}\|x^{k+1}-z+u^k\|_2^2,\\
u^{k+1} &= u^k+x^{k+1}-z^{k+1}.
\end{aligned}
\]

`x` 步等价于求解

\[
(A^TA+(\mu+\rho_k)I)x=A^T(y-b)+\rho_k(z^k-u^k).
\]

`Aᵀ(y-b)` 在求解开始时只计算一次；`z` 步为

\[
z^{k+1}=\max\left(0,x^{k+1}+u^k-\frac{\lambda}{\rho_k}\right).
\]

初始化为 `x^0=z^0=x0`（未提供时为零向量）、`u^0=0`。`x0` 必须非负，不做静默投影。第 0 次迭代定义 `z^{-1}=z^0`，因此初始原始/对偶残差均为 0；若初始 `z^0` 的 KKT 与相对间隙已达标，可返回 `iterations=0` 的 `converged`。

### 4.2 黑盒 CG 内层

`x` 步法方程算子定义为

\[
N(v)=A^T(Av)+(\mu+\rho_k)v.
\]

CG 只通过 `matvec/rmatvec` 评估 `N(v)`，不显式形成 `AᵀA`。每个 ADMM 迭代用上一次 `x` 作为 CG 初值。CG 停止条件为

\[
\|r\|_2 \le \texttt{cg\_tol}\cdot\max(1,\|\mathrm{rhs}\|_2),
\]

或达到 `cg_max_iter`。CG 返回迭代数和最终相对残差；若 RHS 为零，直接返回零解且 CG 迭代数为 0。CG 输出必须有限，否则抛出 `FloatingPointError`。

### 4.3 归一化 ADMM 残差

在第 `k` 次完整更新后计算（迭代编号从 1 开始）：

\[
\epsilon_{\rm primal}
=\frac{\|x^k-z^k\|_2}{\max(1,\|x^k\|_2,\|z^k\|_2)},
\]

\[
\epsilon_{\rm dual}
=\frac{\rho_k\|z^k-z^{k-1}\|_2}{\max(1,\rho_k\|u^k\|_2)}.
\]

每次迭代都记录这两个残差、当前 `ρ` 和本次 CG 迭代数。仅当

\[
\epsilon_{\rm primal}\le\texttt{residual\_tol},\qquad
\epsilon_{\rm dual}\le\texttt{residual\_tol}
\]

同时成立时，才允许进入最终证书判定；残差缩小本身永远不能单独声明收敛。

### 4.4 可选自适应 ρ

`adaptive_rho=True` 时在每轮残差计算后做有界残差平衡：

- 若 `primal_residual > rho_balance * dual_residual`，则 `ρ ← min(rho_max, ρ * rho_scale)`；
- 若 `dual_residual > rho_balance * primal_residual`，则 `ρ ← max(rho_min, ρ / rho_scale)`；
- 其他情况不变。

更新 `ρ` 时必须保持未缩放对偶变量 `ρu` 连续：若 `ρ_old → ρ_new`，则同时执行

\[
u\leftarrow \frac{\rho_{\rm old}}{\rho_{\rm new}}u.
\]

该纯缩放应实现为可独立测试的内部助手。`ρ` 始终保持在 `[rho_min, rho_max]` 内；`rho_min<=0`、`rho_max<rho_min`、`rho_balance<1`、`rho_scale<=1` 均为非法参数。

### 4.5 停机与证书

证书层复用现有 `compute_certificate(A, y, z, lam, mu=mu, b=b)`，输出变量为非负 `z`。证书成本如实记录：每次证书计算增加一对 `A/Aᵀ`；ADMM 的 `x` 步成本来自 CG 内部的 `A/Aᵀ` 次数。

停机规则：

1. 第 0 次：计算初始 `z^0` 证书；若双证书达标，返回 `converged`、`iterations=0`。
2. 第 `k>=1` 次：先计算归一化原始/对偶残差；只有两个残差都达标时才计算当前 `z^k` 证书，且只有 `kkt_residual<=kkt_tol` 与 `relative_gap<=gap_tol` 同时成立才返回 `converged`。
3. 每 `certificate_stride` 次迭代以及最终迭代都计算证书并进入 `certificates` 历史；这些周期性证书用于记录与最终返回。只有当当前迭代的归一化残差也已达标时，对应证书才允许触发 `converged`；残差触发但最终证书未达标，或证书达标但残差未达标，都不得收敛。
4. 达到 `max_iter` 时返回最后一次 `z`、真实最终证书和完整残差历史，`status="max_iter"`，不得因目标下降、残差下降或 CG 收敛而伪报收敛。

只允许把浮点舍入量级的微小负间隙截为零；该规则已由证书层强制，ADMM 不另做放宽。

## 5. 错误处理与数值约定

- `lam<=0`、`mu<0`、`rho<=0`、非正 `max_iter/cg_max_iter/certificate_stride`、非正容差、非法自适应 `ρ` 参数均抛出 `ValueError`。
- `y/b/x0` 必须有限、形状一致；`x0` 必须非负，不做静默投影。
- CG、ADMM 迭代或证书计算出现 NaN/Inf 时抛出 `FloatingPointError`。
- 所有内部向量使用 `float64`；输出解不复用调用者的 `x0` 内存。
- `solve_admm` 不修改 `A/y/b/x0`；`b` 是已知常数，不被估计。
- 本规格不引入 SciPy；运行时仍只依赖 NumPy。

## 6. 测试设计（TDD）

每个生产行为先写一个因功能缺失而失败的测试，确认失败原因后再实现。测试至少覆盖：

1. **共享校验重构**：现有 44 个测试保持通过；ADMM 与 FISTA 对同类非法输入给出一致的 `ValueError`。
2. **CG 精确性**：对角黑盒算子的 `(AᵀA+(μ+ρ)I)x=rhs` 解与逐对角线解析解一致；只调用 `matvec/rmatvec`，不读取内部矩阵。
3. **标量解析锚点**：非负 LASSO `x*=max(0,y-b-λ)`；弹性网 `x*=max(0,(y-b-λ)/(1+μ))`；最终输出与双证书同时达标。
4. **已知背景 b**：`b≠0` 的标量/小问题命中解析解，证书使用同一个 `b`。
5. **零 λmax**：`λ>=lambda_max` 时初始零解直接收敛，`iterations=0`。
6. **z 步闭式**：`max(0,x+u-λ/ρ)` 的零区间、阈值和弹性网无关性（μ 不进入 z 步）有纯单元测试。
7. **残差停机**：构造残差已小但证书未达标的用例，确认不报告收敛；构造证书达标但 `x-z` 未一致的用例，同样不报告收敛。
8. **max_iter 诚实性**：极小 `max_iter` 返回 `max_iter`，带真实最终证书，不抛伪异常。
9. **自适应 ρ**：纯助手测试验证 `ρ_old*u_old == ρ_new*u_new`（未触发截断时）；积分测试验证历史中的 `ρ` 始终在 `[rho_min,rho_max]` 且有限。
10. **黑盒算子**：只暴露 `shape/matvec/rmatvec` 的自定义算子可完成求解，不依赖稠密访问。
11. **参数错误**：负/零 λ、负 μ、非正 ρ、非法容差、负初值、形状不一致、非有限输入均被拒绝。
12. **FISTA 交叉检查**：同一确定性小问题上，ADMM 与 FISTA 都通过双证书，最终目标值相对差异达到测试容差；不用 FISTA 作为逐位标准答案。

独立正确性锚点优先使用标量/小问题解析 KKT 与证书；本轮不引入 LARS、glmnet、SciPy 优化器、真实 FFT 算子或 EF-Spin 数据作为“标准答案”。

## 7. 验收标准

- pytest 全量通过，且每项新功能都有实现前观察到的预期失败记录；重构后 FISTA 既有测试不回退。
- 确定性小算例最终同时满足归一化 ADMM 残差、`ε_KKT ≤ 1e-5`、`relative_gap ≤ 1e-4`。
- `x` 步不依赖显式 `AᵀA` 或 `A` 的稠密存储；稠密矩阵只是算子的一种适配。
- 结果历史能说明每次迭代的原始/对偶残差、ρ 与 CG 迭代数；最终状态不会把 `max_iter` 伪装成收敛。
- README/调研中的 ADMM 描述与代码一致；不宣称已验证背景块联合估计、真实 FFT 算子、28.6 万体素性能或 EF-Spin 重建效果。
- 仓库文本不包含机器相关绝对路径、个人信息或密钥。
