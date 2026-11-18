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

运行时只依赖 NumPy；测试使用 pytest 与 SciPy（QP 独立参考解）。生成基准图表还需安装 `python -m pip install -e ".[test,benchmark]"`，其中包含 Matplotlib。

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

> 注：示例中的具体数字可能因 NumPy/BLAS 平台与版本不同而略有差异；验收标准是基于容差的断言（`kkt_residual <= kkt_tol`、`relative_gap <= gap_tol`），而非逐位复现输出。

- `status` 只有在最终 λ 阶段同时满足 `kkt_residual <= kkt_tol` 与 `relative_gap <= gap_tol` 时才为 `"converged"`；达到 `max_iter` 时如实返回 `"max_iter"`，不以目标下降冒充收敛。
- 证书按 `certificate_stride` 间隔与最终迭代点评估，并非每步免费产出：加速 FISTA 在动量点计算梯度，接受迭代点的证书需要额外一对 `A/Aᵀ`（ISTA 的当前点梯度可直接复用）。
- 弹性网（`mu > 0`）、已知背景 `b`、显式 `lipschitz`、回溯与自适应重启等完整参数见 `solve_fista` 签名；黑盒算子只需实现 `shape` / `matvec` / `rmatvec`。

### 三分裂 ADMM

```python
import numpy as np
from optimization_math.solvers.admm import solve_admm

A = np.array([[1.0, 1.0], [0.0, 1.0]])
y = np.array([2.0, 1.0])
result = solve_admm(A, y, 0.1, rho=1.0, certificate_stride=10)
assert result.status == "converged"
assert result.certificate.kkt_residual <= 1e-5
assert result.certificate.relative_gap <= 1e-4
print(result.status)
print(result.x)
```

- 三分裂：`x` 承载数据项与弹性网二次项、`z` 承载非负 + L1、约束 `x=z`；`x` 步用黑盒 CG 求解 `(AᵀA+(μ+ρ)I)x = Aᵀ(y-b)+ρ(z-u)`，不组装 `AᵀA`。
- `result.x` 返回非负 `z`；证书在 `z` 上复用与 FISTA 相同的 `compute_certificate`。收敛要求归一化原始/对偶残差与 `kkt_residual ≤ kkt_tol`、`relative_gap ≤ gap_tol` **同时**满足，残差缩小、CG 收敛或 `max_iter` 都不单独判定收敛。
- 本轮实现覆盖**已知背景 `b`** 的核心三分裂 ADMM；背景块 `B c` 联合估计作为后续独立工作另行规格化。

## 求解器路线图

| 优先级 | 求解器 | 说明 |
|---|---|---|
| P0 | 非负 FISTA（已实现） | 加速 + 自适应重启 + 回溯 + λ 延拓暖启动；证书按可配置间隔与最终迭代点评估 |
| P0 | 三分裂 ADMM（已实现·已知 b） | 黑盒 CG 内层；z 步非负闭式；可选自适应 ρ；背景块联合估计待单独规格 |
| P1 | 工作列贪心 | 支撑定位器；与上两者互供 warm start；终检 KKT 通过即全局最优 |
| P2 | 小算例 ground truth | LARS / l1_ls / glmnet 在降采样算例上三方对照 |

统一可选后处理：支撑内 NNLS 幅度去偏（数值秩检查 + 残差保护）。

## 算法调研扩展（经典问题覆盖）

以「已知真值 + 评价指标 + 自动判定」统一范式，对五类经典优化问题各完成调研文档、求解器、证书与基准（详见 [总路线图](docs/survey/2026-11-17-算法调研扩展总路线图.md)）：

| 类型 | 求解器 | 证书 | 基准判定 |
|---|---|---|---|
| T1 带符号 LASSO/弹性网 | `fista_signed.py` | 软阈值 KKT + 对偶 gap | F1 对比非负版 |
| T2 稀疏 logistic | `fista_logistic.py` | logistic 梯度 KKT + Fenchel 对偶 | λ-F1 非单调 |
| T3 1D TV 去噪 | `chambolle_tv.py` | Chambolle 原始-对偶 gap | λ-误差非单调 |
| T4 盒约束 QP/NNLS | `qp_box.py` | box KKT + 二次下界 gap | 边界活动识别 |
| T5 group/低秩 | `fista_group.py` / `fista_nuclear.py` | 组范数 KKT / 近端不动点 | 组联合选择 / 秩恢复 |
| 进阶：安全筛选 | `screening.py` | 复用带符号证书（对偶点/θ/gap）| 安全剔除 92–95% 坐标 |
| 进阶：主-对偶 | `pdhg.py` | 复用带符号证书（同 FISTA 验收）| 与 FISTA 收敛到同一最优解 |

## 证书化对比协议

同一问题 → 多求解器 → 五指标横向验收：**ε_KKT、相对对偶间隙、支撑 F1、幅度 RMSE、耗时**。凸问题不保证最优解唯一；只有满足严格凸性等条件时才能比较解向量的一致性，否则应比较目标值与证书。

### 标准问题：独立最优解与算法横向对比

```powershell
python -m pip install -e ".[test,benchmark]"
python -W error -m benchmarks.bench_standard --out-dir benchmarks/out/standard
```

- **8 类问题、96 个已知最优解实例、177 次算法运行**：非负 FISTA/ADMM、带符号 FISTA/PDHG、QP 与 logistic 对比 SciPy；TV、group 与核范数用独立解析解验收。
- 3 个随机种子，条件目标 1/10/100；同一实例、同一证书精度，分别比较最优目标误差、解误差、迭代数、实测正演/伴随调用与时间。数据生成真值的恢复误差单列，不冒充最优化误差。
- 这是小规模正确性测试（向量维度 8、核范数矩阵 4×4、两点 TV），不是大型成像性能结论。
- [参考解推导与协议](docs/benchmarks/standard-optimization-comparisons.md) · [实测结果与图像](docs/benchmarks/2026-11-17-standard-results/README.md) · [新增缺陷与修复记录](docs/reviews/2026-11-17-standard-validation-followup.md)

## 目录结构

```
optimization_math/
├── benchmarks/                        # 已知真值基准（指标 + 自动断言 + 图表）
│   ├── metrics.py                     # 恢复指标（rel.L2 / F1 / RMSE / 稳定性）
│   ├── bench_signed_lasso.py          # T1
│   ├── bench_logistic.py              # T2
│   ├── bench_tv.py                    # T3
│   ├── bench_qp.py                    # T4
│   ├── bench_structured.py            # T5（group + 核范数）
│   ├── bench_screening.py             # GAP-safe 筛选降维效果
│   ├── bench_pdhg.py                  # 主-对偶 vs FISTA
│   ├── standard_problems.py           # 标准问题：独立最优解 fixtures
│   ├── bench_standard.py              # 标准问题跨算法对比 runner
│   └── standard_report.py             # 结果报告（JSON/Markdown/图）
├── docs/
│   ├── survey/                        # 算法调研（含 T1–T5）
│   └── superpowers/                   # 设计规格（specs）与实施计划（plans）
├── src/optimization_math/
│   ├── operators.py                   # 带形状检查的线性算子接口
│   ├── operators_difference.py        # 一阶差分算子 D（隐式）
│   ├── certificates.py                # 非负 LASSO 证书
│   ├── certificates_signed.py         # 带符号 LASSO 证书
│   ├── certificates_logistic.py       # logistic Fenchel 对偶证书
│   ├── certificates_qp.py             # box-QP 二次下界证书
│   ├── certificates_group.py          # group LASSO 证书
│   ├── certificates_nuclear.py        # 核范数（谱）证书
│   ├── screening.py                   # GAP-safe 安全筛选（列范数 + 安全球）
│   └── solvers/
│       ├── _common.py                 # 共享校验/估计助手（私有）
│       ├── fista.py                   # 非负 FISTA
│       ├── admm.py                    # 三分裂 ADMM
│       ├── fista_signed.py            # 带符号 FISTA
│       ├── fista_logistic.py          # 稀疏 logistic FISTA
│       ├── chambolle_tv.py            # 1D TV（Chambolle 投影）
│       ├── qp_box.py                  # 盒约束 QP（投影梯度）
│       ├── fista_group.py             # group LASSO FISTA
│       ├── fista_nuclear.py           # 核范数 FISTA
│       └── pdhg.py                    # 主-对偶 Chambolle-Pock
└── tests/                             # 含证书断言的测试
    ├── test_standard_comparisons.py   # 独立最优解对比矩阵验收
    ├── test_standard_regressions.py   # group 分区 / TV 偏移数值回归
    └── test_standard_report.py        # 报告渲染与失败保留
```

## 文档

- [L1/L2 求解算法调研（2026-11-13）](docs/survey/2026-11-13-L1-L2-稀疏求解算法调研.md) — 11 类算法族、统一收敛证书推导、算法对比总表、43 条可核验文献、P0/P1/P2 选型建议
- [非负 FISTA P0 设计规格（2026-11-15）](docs/superpowers/specs/2026-11-15-nonnegative-fista-design.md) — 接口、算法、证书与错误处理的完整设计
- [实施计划（2026-11-15）](docs/superpowers/plans/2026-11-15-nonnegative-fista.md) — TDD 任务分解与验收记录
- [三分裂 ADMM 核心设计规格（2026-11-16）](docs/superpowers/specs/2026-11-16-three-split-admm-design.md) — 已知背景 b 的 ADMM / split Bregman 同族核心求解器设计
- [实施计划（2026-11-16）](docs/superpowers/plans/2026-11-16-three-split-admm.md) — TDD 任务分解与验收记录
- [算法调研扩展总路线图（2026-11-17）](docs/survey/2026-11-17-算法调研扩展总路线图.md) — 五类经典优化问题的调研/求解器/证书/基准索引
- [带符号 LASSO/弹性网（T1）](docs/survey/2026-11-17-带符号-LASSO-弹性网.md) · [稀疏 logistic（T2）](docs/survey/2026-11-17-稀疏-logistic-回归.md) · [Fused LASSO/TV 去噪（T3）](docs/survey/2026-11-17-Fused-LASSO-TV-去噪.md) · [盒约束 QP/NNLS（T4）](docs/survey/2026-11-17-盒约束-QP-NNLS.md) · [group LASSO/核范数低秩（T5）](docs/survey/2026-11-17-group-LASSO-核范数低秩.md)
- [L2+L1 进阶算法族综述（2026-11-17）](docs/survey/2026-11-17-L2L1-进阶算法族综述.md) — 主-对偶（PDHG）、拟牛顿（OWL-QN/Prox-QN）、坐标下降、半光滑牛顿、safe screening 的收敛率与选型
- [GAP-safe 筛选与主-对偶 PDHG 落地（2026-11-17）](docs/survey/2026-11-17-GAP-safe筛选与主对偶PDHG落地.md) — 两个算法族的原理、实现口径、实测对比（含 screening 安全口径澄清）
- [本地正确性审查（2026-11-17）](docs/reviews/2026-11-17-local-correctness-review.md) — 已复现缺陷、修复证据与剩余限制

## 约定

- 代码与文档一律使用相对路径
- 本仓库为公开仓库；不得提交个人信息、密钥、机器绝对路径或未获授权的专利材料。
