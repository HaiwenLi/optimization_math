# optimization_math

通用型 L1/L2 稀疏优化框架：非负稀疏重建问题的**可插拔求解器** + **与算法无关的统一收敛证书**。

## 动机

L1 类问题（LASSO / 弹性网 / 非负变体）有大量成熟求解器——FISTA、ADMM、内点法、贪心类等——但多数实现把"问题定义、算子、求解器、收敛判定"绑死在一起，换个任务就得重写。本项目把这四层解耦：

| 层 | 内容 | 通用性 |
|---|---|---|
| 问题层 | 数据 `y` + 算子 `A` + 正则类型，纯配置 | 任意任务 |
| 算子层 | `A` 只暴露正演/伴随接口；卷积类算子用 FFT 隐式实现，不显式组装矩阵 | 覆盖去卷积 / CT / MRI / 编码成像等 |
| 求解器层 | 可插拔后端，同一问题跑任意求解器 | 一处实现，处处复用 |
| 证书层 | 投影 KKT 残差 + 相对对偶间隙，与算法解耦 | 统一验收标准 |

## 问题形式

```
min_{o ≥ 0}  ½‖A o − y‖₂² + λ‖o‖₁        （弹性网：再加 (μ/2)‖o‖₂²）
```

收敛不以"迭代数达到上限"或"残差变小"单独判定，一律以证书为准。

## 安装与测试

```powershell
python -m pip install -e ".[test]"
python -m pytest -v
```

运行时只依赖 NumPy；测试使用 pytest。

## 快速上手

```python
import numpy as np
from optimization_math.solvers.fista import solve_fista

A = np.array([[1.0, 1.0], [0.0, 1.0]])
y = np.array([2.0, 1.0])
result = solve_fista(
    A, y, 0.1,
    lambda_path=[1.0, 0.5, 0.1],
    certificate_stride=10,
)
assert result.status == "converged"
print(result.x)
print(result.certificate.kkt_residual, result.certificate.relative_gap)
```

示例输出（非负 LASSO，λ 从 1.0 延拓到 0.1，共三个阶段）：

```text
[0.9000017  0.99999895]
6.476165656835956e-06 9.430809100041682e-07
```

- `status` 只有在最终 λ 阶段同时满足 `kkt_residual <= kkt_tol` 与 `relative_gap <= gap_tol` 时才为 `"converged"`；达到 `max_iter` 时如实返回 `"max_iter"`，不以目标下降冒充收敛。
- 证书按 `certificate_stride` 间隔与最终迭代点评估，并非每步免费产出：加速 FISTA 在动量点计算梯度，接受迭代点的证书需要额外一对 `A/Aᵀ`（ISTA 的当前点梯度可直接复用）。
- 弹性网（`mu > 0`）、已知背景 `b`、显式 `lipschitz`、回溯与自适应重启等完整参数见 `solve_fista` 签名；黑盒算子只需实现 `shape` / `matvec` / `rmatvec`。

## 求解器路线图

| 优先级 | 求解器 | 说明 |
|---|---|---|
| P0 | 非负 FISTA（已实现） | 加速 + 自适应重启 + 回溯 + λ 延拓暖启动；证书按可配置间隔与最终迭代点评估 |
| P0 | 三分裂 ADMM（split Bregman） | CG+FFT 内层；z 步非负闭式；支持背景块联合估计 |
| P1 | 工作列贪心 | 支撑定位器；与上两者互供 warm start；终检 KKT 通过即全局最优 |
| P2 | 小算例 ground truth | LARS / l1_ls / glmnet 在降采样算例上三方对照 |

统一可选后处理：支撑内 NNLS 幅度去偏（数值秩检查 + 残差保护）。

## 证书化对比协议

同一问题 → 多求解器 → 五指标横向验收：**ε_KKT、相对对偶间隙、支撑 F1、幅度 RMSE、耗时**。求解器的差异不是"答案不同"（凸问题最优解唯一），而是速度、证书强度与支撑定位能力的权衡。

## 目录结构

```
optimization_math/
├── docs/
│   ├── survey/                      # 算法调研
│   └── superpowers/                 # 设计规格（specs）与实施计划（plans）
├── src/optimization_math/
│   ├── operators.py                 # 带形状检查的线性算子接口
│   ├── certificates.py              # 投影 KKT + 相对对偶间隙证书
│   └── solvers/
│       └── fista.py                 # 非负 FISTA（回溯 / 重启 / λ 延拓）
└── tests/                           # 含证书断言的测试
```

## 文档

- [L1/L2 求解算法调研（2026-11-13）](docs/survey/2026-11-13-L1-L2-稀疏求解算法调研.md) — 11 类算法族、统一收敛证书推导、算法对比总表、43 条可核验文献、P0/P1/P2 选型建议
- [非负 FISTA P0 设计规格（2026-11-15）](docs/superpowers/specs/2026-11-15-nonnegative-fista-design.md) — 接口、算法、证书与错误处理的完整设计
- [实施计划（2026-11-15）](docs/superpowers/plans/2026-11-15-nonnegative-fista.md) — TDD 任务分解与验收记录

## 约定

- 代码与文档一律使用相对路径
- 仓库内容涉及未决专利事项，默认保持私有
