# 非负 FISTA P0 设计规格

- 日期：2026-11-15
- 状态：用户已确认并批准；弹性网对偶符号已修正为增广形式（q = Aᵀr + μx，可行性 θ·q ≥ −λ·1），实现与测试已验证；实施计划已批准并执行完毕（2026-11-15 批准，2026-11-16 实现完成）
- 范围：非负 LASSO / 弹性网的通用 Python 求解基线

## 1. 目标与边界

实现一个只依赖 NumPy 的非负 FISTA 求解器，求解

\[
\min_{x\ge 0}\quad \frac12\|Ax+b-y\|_2^2+\lambda\|x\|_1+\frac{\mu}{2}\|x\|_2^2,
\qquad \lambda>0,\ \mu\ge 0.
\]

P0 交付包括：

- 黑盒线性算子 `A/Aᵀ` 与稠密矩阵适配；
- 非负弹性网近端算子；
- 幂迭代估计初始 Lipschitz 常数，并带回溯保护；
- 梯度式自适应重启；
- 显式 λ 延拓路径与暖启动；
- 投影 KKT 与相对对偶间隙双证书；
- 可重复的 pytest 测试和最小使用文档。

本轮不包括：FFT 成像算子、未知背景 `b` 的联合估计、ADMM、工作列、支撑内 NNLS 去偏、外部求解器 ground truth、真实 EF-Spin 数据接入和全规模性能结论。

## 2. 方案选择

采用 **Python/NumPy 通用库**，而不是单文件试验脚本或 MATLAB 原型：

- 仓库已有 Python 工程约定；
- 求解器只依赖 `matvec/rmatvec`，后续 FFT 算子无需修改求解器；
- 证书独立成模块，ADMM 和工作列可以复用；
- 运行时只引入 NumPy，测试引入 pytest，不依赖 SciPy。

所有求解状态使用一维 `float64` 数组。多维成像变量由具体算子在 `matvec/rmatvec` 内 reshape，不把成像维数泄漏到通用求解器接口。

## 3. 模块与接口

拟新增：

```text
pyproject.toml
src/optimization_math/
├── __init__.py
├── operators.py
├── certificates.py
└── solvers/
    ├── __init__.py
    └── fista.py
tests/
├── test_operators.py
├── test_certificates.py
└── test_fista.py
```

### 3.1 算子层

```python
class LinearOperator(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...

    def matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def rmatvec(self, y: NDArray[np.float64]) -> NDArray[np.float64]: ...


def as_linear_operator(A: LinearOperator | NDArray) -> LinearOperator: ...
```

约定 `shape == (m, n)`，`matvec: (n,) -> (m,)`，`rmatvec: (m,) -> (n,)`。稠密适配器不复制输入矩阵。算子输入形状或返回形状错误时抛出 `ValueError`。

### 3.2 求解器层

```python
def solve_fista(
    A: LinearOperator | NDArray,
    y: NDArray,
    lam: float,
    *,
    mu: float = 0.0,
    b: NDArray | None = None,
    x0: NDArray | None = None,
    lipschitz: float | None = None,
    max_iter: int = 1000,
    kkt_tol: float = 1e-5,
    gap_tol: float = 1e-4,
    certificate_stride: int = 10,
    restart: bool = True,
    backtracking: bool = True,
    lambda_path: Sequence[float] | None = None,
    continuation_tol_scale: float = 10.0,
    power_iter: int = 50,
    power_tol: float = 1e-6,
) -> FistaResult: ...
```

`lambda_path=None` 表示直接求解目标 `lam`。若提供路径，路径必须严格递减、每项为正、最后一项等于 `lam`；前序阶段复用上一阶段解作为暖启动。`max_iter` 是每个阶段的上限，最终阶段未达到双证书时返回 `status="max_iter"`，不得报告收敛。

```python
@dataclass(frozen=True)
class Certificate:
    stage: int
    iteration: int
    lam: float
    primal: float
    dual: float
    kkt_residual: float
    relative_gap: float


@dataclass(frozen=True)
class FistaResult:
    x: NDArray
    status: Literal["converged", "max_iter"]
    iterations: int
    total_iterations: int
    lipschitz: float
    restart_count: int
    certificate: Certificate
    history: tuple[Certificate, ...]
```

结果保存不可变历史记录；`x` 是最终阶段的非负解。证书历史只包含实际计算证书的迭代点，`Certificate.iteration` 是该 λ 阶段内的迭代编号。

## 4. 算法

### 4.1 近端与步长

将二次数据项作为光滑部分

\[
f(x)=\frac12\|Ax+b-y\|_2^2,\qquad \nabla f(x)=A^T(Ax+b-y),
\]

将非负约束、L1 与 L2 合并到近端：

\[
\operatorname{prox}_{t}(z)
=\frac{\max(0,z-t\lambda)}{1+t\mu},
\qquad t=1/L.
\]

`L` 是 `||A||₂²` 的可用上界。用户未提供 `lipschitz` 时，以固定随机种子对 `AᵀA` 做幂迭代得到初始估计。幂迭代只提供估计而非数学上界，因此默认启用回溯：若

\[
f(x^+) > f(z)+\langle\nabla f(z),x^+-z\rangle
+\frac{L}{2}\|x^+-z\|_2^2
\]

超出浮点舍入容差，则将 `L` 加倍并重新计算近端步。回溯得到的 `L` 在后续迭代中保留，不自动降回估计值。零算子使用 `L=1` 的有限默认值。用户提供明显偏小的 `L` 时，默认回溯仍应能恢复稳定步长。

### 4.2 FISTA 与自适应重启

标准动量系数为

\[
t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}2,
\]

动量点为 `z = x_k + β(x_k - x_{k-1})`。采用 O'Donoghue–Candès 梯度重启判据：当

\[
\langle \nabla f(z),x^+-x_k\rangle>0
\]

时放弃该外推步，改从 `z=x_k` 重新计算梯度与近端步，并重置动量。`restart=False` 时使用标准 FISTA。重启不是收敛证书，最终仍只由 KKT 与对偶间隙判定。

### 4.3 λ 延拓

提供辅助函数：

```python
def lambda_max(A: LinearOperator | NDArray, y: NDArray, *, b: NDArray | None = None) -> float: ...
```

对本问题，`x=0` 时 `g₀=Aᵀ(b-y)`，因此

\[
\lambda_{\max}=\max(0,-\min_i(g_{0,i})).
\]

`λ ≥ λmax` 时零向量满足最优性条件。延拓路径由调用者显式给出，避免把路径长度和下降比例硬编码成未经验证的启发式默认值。中间阶段使用 `continuation_tol_scale` 倍的目标容差；达到当前阶段双证书或当前阶段 `max_iter` 后进入下一阶段。最终阶段始终使用用户指定的目标容差。

## 5. 收敛证书

### 5.1 原问题与投影 KKT

令

\[
r=Ax+b-y,\qquad g=A^Tr+\mu x.
\]

由于 `x≥0`，原目标直接计算为

\[
p=\frac12\|r\|_2^2+\lambda\sum_i x_i+\frac\mu2\|x\|_2^2.
\]

KKT 条件为：

- `x_i>0`：`g_i+λ=0`；
- `x_i=0`：`g_i+λ≥0`。

归一化违反度与调研保持一致：

\[
\epsilon_{KKT}=
\frac{\max\left(\max_{x_i>0}|g_i+\lambda|,
\max_{x_i=0}(-\lambda-g_i)^+\right)}
{\max(\lambda,\|g\|_\infty)}.
\]

支撑按实际非零元素判定，不用任意幅值阈值伪删近零坐标。求解器保证输出非负；证书同时检查输入解的有限性和非负性。

### 5.2 可行对偶与相对间隙

弹性网采用增广算子推导对偶。取对偶变量

\[
\nu=\theta r,\qquad \eta=\theta\sqrt{\mu}x.
\]

对偶可行约束为

\[
\theta(A^Tr+\mu x)\ge-\lambda\mathbf 1.
\]

令 `q=Aᵀr+μx`，取

\[
\theta=\min\left(1,\min_{q_i<0}\frac{\lambda}{-q_i}\right),
\]

不存在负 `q_i` 时 `θ=1`。因此构造的对偶点始终可行。对偶值为

\[
d=-\frac{\theta^2}{2}\left(\|r\|_2^2+\mu\|x\|_2^2\right)
+\theta(b-y)^Tr.
\]

相对间隙为

\[
\operatorname{gap}=\frac{p-d}{\max(1,|p|,|d|)}.
\]

只允许把浮点舍入量级的微小负间隙截为零；材料性负间隙必须视为证书或求解错误，不能静默掩盖。

最终收敛必须同时满足 `kkt_residual <= kkt_tol` 和 `relative_gap <= gap_tol`。达到 `max_iter`、目标值停滞或动量重启均不能单独声明收敛。

### 5.3 证书计算成本

调研中“每步免费产出证书”的说法需要修正：加速 FISTA 在动量点 `z` 计算梯度，而当前点 `x` 的 KKT 和对偶证书需要 `Ax` 与 `Aᵀ(Ax+b-y)`。因此默认每 `certificate_stride` 次迭代及最终迭代计算一次证书，每次增加一对 `A/Aᵀ`；不把它宣称为零成本。ISTA 或恰好需要当前点梯度的迭代才能直接复用。

实现时同步修正 README 与调研中的相应表述，并注明弹性网对偶来自增广算子。

## 6. 错误处理与数值约定

- `lam <= 0`、`mu < 0`、非正 `max_iter`、非正容差、非法路径等抛出 `ValueError`。
- `y/b/x0` 必须有限、形状一致；`x0` 必须非负，不做静默投影。
- 迭代出现 NaN/Inf 或回溯无法得到有限步长时抛出 `FloatingPointError`。
- 所有内部向量使用 `float64`；输出解不复用调用者的 `x0` 内存。
- 达到迭代上限返回最后解和真实证书，不抛出伪异常，也不标记收敛。

## 7. 测试设计（TDD）

每个生产行为先写一个因功能缺失而失败的测试，确认失败原因后再实现。测试至少覆盖：

1. **算子**：稠密矩阵正演/伴随、随机向量伴随恒等式、形状错误。
2. **近端**：非负软阈值、零区间、弹性网分母。
3. **λmax**：大于和小于阈值时的零解/非零解行为。
4. **证书**：标量解析最优点的 KKT 接近零；随机小问题的对偶可行性和 `p≥d`；弹性网使用同一构造。
5. **基本求解**：小规模非负 LASSO 的输出非负且双证书达标；大 λ 返回精确零解。
6. **弹性网**：`μ>0` 的解满足对应 KKT 与对偶证书，不能误用纯 LASSO 梯度。
7. **回溯**：故意低估 `lipschitz` 后仍收敛到相同证书口径。
8. **重启**：纯判据测试覆盖触发与不触发分支，相关矩阵积分用例在开启重启时满足双证书。
9. **延拓**：路径按阶段递减、暖启动历史完整、最终使用目标 λ 和目标容差。
10. **停机诚实性**：极小 `max_iter` 返回 `max_iter`，不得仅因目标下降而报告收敛。
11. **参数错误**：负 λ、负 μ、负初值、形状不一致、非有限输入均被拒绝。

独立正确性锚点优先使用标量/小问题解析 KKT 与证书，不在本轮引入 LARS、glmnet、SciPy 优化器或另一个同构 FISTA 作为“标准答案”。

## 8. 验收标准

- pytest 全量通过，且每项新功能都有实现前观察到的预期失败记录。
- 确定性小算例最终同时满足 `ε_KKT ≤ 1e-5`、`relative_gap ≤ 1e-4`。
- 求解器不依赖显式 `AᵀA` 或 `A` 的稠密存储；稠密矩阵只是算子的一种适配。
- README 给出最小调用示例，文档中的证书成本与弹性网公式和代码一致。
- 仓库文本不包含机器相关绝对路径、个人信息或密钥。
- 不宣称已验证真实 FFT 算子、28.6 万体素性能或 EF-Spin 重建效果；这些属于后续集成验收。
