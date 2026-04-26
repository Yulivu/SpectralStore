# 实验记录

这个文档集中记录 SpectralStore 的实验：实验问题、设置、方法、结果，以及相对总设计文档 `SpectralStore.md` 还欠什么。具体产物放在 `experiments/results/`，关键结论同步记录在这里，避免结果散落。

## 实验一：理论验证

### 对照 `SpectralStore.md`

`SpectralStore.md` 中实验一的目标是验证非对称张量谱分解的 entrywise 误差是否达到理论预测量级，以及相对于对称化方法是否有可观测优势。

| 项目 | 总文档要求 |
| --- | --- |
| 数据集 | Synthetic-SBM、Synthetic-Spiked |
| SBM 节点数变化 | 固定 `T=20, r=5`，`n` 从 `500` 到 `10000` |
| SBM 时间长度变化 | 固定 `n=2000, r=5`，`T` 从 `5` 到 `100` |
| Spiked | 做同样两组实验，并额外变化 `SNR={0.5,1,2,5}` |
| 重复次数 | 每个设置 50 次独立重复 |
| 对比方法 | SpectralStore、SymSVD、DirectSVD、Tensor Unfolding + SVD、CP-ALS |
| 指标 | 最大 entrywise 误差、平均 entrywise 误差、Frobenius 误差 |
| 图表 | log-log 图，拟合幂律指数，并与 `O(sqrt(log n / (nT)))` 对照 |

### 当前位置

实验一相关脚本已经归到同一个目录：

```powershell
python scripts\exp1\run_exp1_theory_validation.py
python scripts\exp1\run_exp1_theory_validation.py --refresh-from-results
python scripts\exp1\run_lowsnr_diagnostic.py
python scripts\exp1\plot_theory_from_results.py
```

当前主要结果目录：

```text
experiments/results/exp1/
  results.csv
  summary.md
  sbm_n_sweep.png
  sbm_t_sweep.png
  spiked_snr_sweep.png
  sbm_n_sweep_with_theory.png
  sbm_t_sweep_with_theory.png
```

### 当前已完成的主实验设置

| 组别 | 当前设置 | 当前方法 | 重复 |
| --- | --- | --- | ---: |
| SBM 节点数变化 | `T=20, r=5, K=5, p=0.30, q=0.05`，`n={500,1000,2000,5000}` | `spectralstore_asym`, `sym_svd`, `direct_svd`, `tensor_unfolding_svd` | 10 |
| SBM 时间长度变化 | `n=2000, r=5, K=5, p=0.30, q=0.05`，`T={5,10,20,50}` | `spectralstore_asym`, `sym_svd`, `direct_svd`, `tensor_unfolding_svd` | 10 |
| Spiked SNR 变化 | `n=1000, T=20, r=5`，`SNR={0.5,1,2,5}` | `spectralstore_asym`, `sym_svd`, `direct_svd`, `tensor_unfolding_svd` | 10 |

指标均对比真实期望矩阵 `M*`，不是对比观测矩阵 `A`：

- 最大 entrywise 误差
- 平均 entrywise 误差
- relative Frobenius 误差

图表已经加入：

- `sbm_n_sweep.png`：理论参考线 `O(√(logn/nT))`，缩放到 `spectralstore_asym` 第一个点。
- `sbm_t_sweep.png`：理论参考线 `O(1/√T)`，缩放到 `spectralstore_asym` 第一个点。
- `results.csv`：新增 `max_entrywise_power_law_slope`，记录 `sbm_n` 和 `sbm_t` 的 log-log 幂律拟合斜率。

当前拟合斜率大约只有 `-0.03` 到 `-0.04`，明显弱于理论参考斜率 `-0.5`。这说明当前参数范围内还没有看到理论预测的 max-entrywise 衰减速率，或者最大误差被有限样本、模型偏差、实现细节等因素主导。

### 低信噪比补充诊断

低信噪比诊断现在已经并入实验一结果：

- `experiments/results/exp1/results.csv` 中的 `sweep=sbm_low_snr`
- `experiments/results/exp1/summary.md` 中的 `Low-SNR Synthetic-SBM Diagnostic` 小节

设置：

| 参数 | 值 |
| --- | ---: |
| 数据集 | Synthetic-SBM |
| `n` | 2000 |
| `T` | 20 |
| rank `r` | 5 |
| 社区数 `K` | 5 |
| `p_in` | 0.15 |
| `p_out` | 0.10 |
| temporal jitter | 0.08 |
| directed | true |
| 重复次数 | 10 |
| 随机种子 | 401 到 410 |

结果：

| 方法 | 单次耗时 | 最大 entrywise | 平均 entrywise | relative Frobenius |
| --- | ---: | ---: | ---: | ---: |
| `spectralstore_asym` | `2.32581 +/- 0.0914` | `0.200803 +/- 0.00624` | `0.00837345 +/- 0.000152` | `0.0981681 +/- 0.00151` |
| `sym_svd` | `2.17164 +/- 0.1` | `0.193957 +/- 0.00934` | `0.00670017 +/- 0.000215` | `0.0808153 +/- 0.00201` |
| `direct_svd` | `2.12424 +/- 0.0739` | `0.192348 +/- 0.00882` | `0.00726718 +/- 0.000182` | `0.0865301 +/- 0.00176` |

解读：在低信噪比设置 `p_in=0.15, p_out=0.10` 下，当前 `spectralstore_asym` 没有优于 SVD baseline。`direct_svd` 的最大 entrywise 最好，`sym_svd` 的平均 entrywise 和 Frobenius 最好。这是实验一的风险点，后续论文表述必须区分默认强信号设置和低信噪比设置。

### 相对总文档还欠什么

| 欠缺项 | 当前状态 | 原因 | 后续补法 |
| --- | --- | --- | --- |
| `n=10000` | 当前最大 `n=5000` | dense 路径太慢，之前完整运行数小时未完成 | 等 sparse path 或单独少量重复补点 |
| `T=100` | 当前最大 `T=50` | 运行时间增加，先保留趋势点 | 主趋势确认后单独补 `T=100` |
| 50 次重复 | 当前 10 次 | 控制运行时间 | 最终关键设置补到 50 次 |
| CP-ALS | 当前实验一不跑 | TensorLy CP-ALS 在大 dense tensor 上不可行 | 小规模补 CP-ALS，或等 sparse/masked CP |
| Spiked 的 `n/r` 变化 | 当前只做 SNR 变化 | 先对齐最关键 SNR 轴 | 后续补 `n={200,500,1000,5000}` 和 `r={3,5,10}` |
| 理论衰减结论 | 当前斜率远弱于 `-0.5` | 可能未进入理论区间或实现/参数不匹配 | 需要更大规模、更多 T、以及理论构造核对 |

因此，实验一现在是“结果已集中、主轴可读、风险点已记录”的阶段，但还不是 `SpectralStore.md` 定义的完整论文级实验一。
