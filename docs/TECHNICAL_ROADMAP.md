# SpectralStore 技术路线差距与工程路线图

本文档对照 `SpectralStore.md`，记录当前仓库已经完成的能力、距离完整技术路线的缺口、优先级，以及下一阶段验收标准。它的定位是工程路线图：每一项都应能被拆成可实现、可测试、可复现实验的任务。

## 当前已完成

- 仓库结构已经按可读性拆分：`data/`、`scripts/`、`experiments/`、`tests/smoke/`。
- 已有基础因子化存储：`FactorizedTemporalStore` 存储 `left/right/temporal/lambdas/residuals`。
- 已有第一版查询层：`LINK_PROB`、`TOP_NEIGHBOR`、`TEMPORAL_TREND`、`ANOMALY_DETECT` 的基础接口。
- 已有第一版压缩器：`AsymmetricSpectralCompressor`、`SymmetricSVDCompressor`、`DirectSVDCompressor`。
- 已有 Bitcoin-OTC 下载、加载和 preliminary 实验。
- 已有 Synthetic-SBM 生成、真值矩阵评估和 preliminary 实验。

## 完整差距清单

| 模块 | 当前状态 | 缺口 |
| --- | --- | --- |
| 存储层 | 有内存态因子化 store 和 CSR residual 字段 | 缺 serialization、compression ratio 统计、residual 元数据、uncertainty 参数 |
| 压缩引擎 | 有 dense mean/SVD 原型和多 split 非对称 ensemble | 缺真正张量展开、ARD/VI、鲁棒交替优化、增量更新、稀疏大图 SVD |
| 查询层 | Q1/Q2/Q4/Q5 有基础返回 | Q3 `COMMUNITY` 缺失；Q1/Q2/Q4/Q5 缺误差返回、query optimizer、batch API |
| 索引层 | 只有空包 | PQ/MIPS、时间索引、社区倒排索引均未实现 |
| 误差保证 | 有经验 entrywise/Frobenius 指标 | 缺 entrywise bound、节点度参数、噪声估计、query error bound |
| 数据层 | 有 Bitcoin-OTC 和 Synthetic-SBM | 缺 Bitcoin-Alpha、UCI、Enron、OGB、Reddit、Stack Overflow 等加载器 |
| Baseline | 有 SymSVD/DirectSVD | 缺 NMF、CP/Tucker、BPTF、RPCA+SVD、图摘要和动态图方法 |
| 实验 | 有两个 preliminary 实验 | `SpectralStore.md` 的 7 组正式实验大多未做 |

## 优先级路线

- **P0: Synthetic-Attack + Robust residual**  
  目标是做出 `SpectralStore-Full` 与 `NoRobust` 的第一组消融，让方法进入它最擅长的对抗扰动场景。
- **P1: entrywise bound 初版**  
  加入经验噪声估计、节点度参数和可返回的 query error bound。当前已完成第一步：MAD 自适应 residual threshold，用于避免无攻击时固定比例剥离 residual。
- **P2: query optimizer + residual correction**  
  让查询根据误差容忍度决定是否读取 residual。
- **P3: PQ index + query latency experiment**  
  将 `TOP_NEIGHBOR` 从 dense scan 推向近似 MIPS。
- **P4: ARD rank selection**  
  用自动秩选择替代手动 rank，并和交叉验证对比。
- **P5: 正式大规模实验与更多 baseline**  
  补齐 `SpectralStore.md` 的数据集、baseline 和论文级实验。

## 下一阶段实现计划

下一阶段聚焦 **鲁棒攻击实验 + Robust residual compressor**。

需要新增 Synthetic-Attack 数据生成能力：

- 支持 `random_flip`、`targeted_cross_community`、`sparse_outlier_edges`。
- `snapshots` 表示受攻击观测图，`expected_snapshots` 保持干净真值。
- 记录 attack metadata，用于 anomaly precision/recall。

需要新增 `RobustAsymmetricSpectralCompressor`：

- 先做初始低秩估计。
- 计算残差矩阵。
- 按固定阈值或分位数分离 sparse residual。
- 用去 residual 后的矩阵重新拟合。
- 输出 CSR residual，供 `ANOMALY_DETECT` 和 residual correction 使用。

需要新增实验：

- 目录：`experiments/preliminary/synthetic_attack/`
- 脚本：`scripts/run_preliminary_synthetic_attack.py`
- 对比：`spectralstore_full`、`spectralstore_no_robust`、`baseline_sym_svd`、`baseline_direct_svd`
- 指标：max entrywise error、mean entrywise error、relative Frobenius error、anomaly precision/recall、residual sparsity/residual nnz

## 验收标准

- `pytest -p no:cacheprovider` 全部通过。
- `python scripts/run_preliminary_synthetic_attack.py` 可直接运行。
- `experiments/preliminary/synthetic_attack/results/summary.md` 包含 Full vs NoRobust vs baseline 表格。
- Synthetic-Attack 生成器返回正确尺寸，且 attack severity 增大时扰动边数量增加。
- Robust compressor 能产生 residuals，且 residual 数量等于时间快照数。
