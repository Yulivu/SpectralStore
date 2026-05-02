# 阶段 A-D 系统路线说明

更新日期：2026-05-02

当前第一目标不是立即写论文，而是把 SpectralStore 当作一个可运行的时序图 AQP 存储与查询系统彻底跑通，判断它是否能自然形成值得写的方向。A-D 路线对应的是“从压缩器结果，转向系统结果”。

## 总体判断

- 旧主线 Exp1/Exp2/Exp4/Exp5 已经证明系统能跑，但大多仍偏重重构误差、存储比和机制诊断。
- 新路线把实验中心移到：
  - 压缩域 Q1-Q5 查询是否可用；
  - 索引路径是否带来 NoIndex/Index 的系统差异；
  - 鲁棒残差是否真的修正查询和捕获异常；
  - ARD 是不是能修成可靠机制。
- 大规模运行应放在 AutoDL/HPC，本地只做小规模 smoke test。

## 阶段 A：Exp3 查询中心 benchmark

- 新增脚本：
  - `scripts/exp3/run_query_benchmark.py`
- 新增配置：
  - `experiments/configs/exp3/query_benchmark.yaml`
- 输出目录：
  - `experiments/results/exp3/query_benchmark/`
- 对应系统架构：
  - `FactorizedTemporalStore`
  - `QueryEngine`
  - Q1-Q5 查询接口
  - 残差参与查询的执行路径
- 覆盖查询：
  - Q1：`LINK_PROB(u, v, t)`
  - Q2：`TOP_NEIGHBOR(u, t, k)`
  - Q3：`COMMUNITY(t)`
  - Q4：`TEMPORAL_TREND(u, v, t1, t2)`
  - Q5：`ANOMALY_DETECT(t, threshold)`
- 主要指标：
  - Q1 RMSE/MAE
  - Q2 recall@k、overlap、latency
  - Q3 NMI
  - Q4 RMSE/MAE
  - Q5 precision/recall/F1
  - `compressed_vs_raw_sparse_ratio`
  - `mean_latency_ms`
- 能回答的问题：
  - SpectralStore 是否已经是查询系统，而不是只有压缩器？
  - factor-only 和 factor+residual 在查询层面的差异是什么？
  - 查询质量和存储比之间是否存在值得分析的折中？

## 阶段 B：索引/NoIndex 对比

- 实现方式：
  - 作为 Exp3 的 Q2 子路径输出，不单独拆成另一个实验。
- 对应系统架构：
  - `src/spectralstore/index/exact_mips.py`
  - `src/spectralstore/index/ann_mips.py`
  - `QueryEngine.top_neighbor`
- 对比路径：
  - `scan_factor_only`
  - `scan_residual`
  - `exact_index`
  - `exact_index_residual`
  - `ann_index`
  - `ann_index_residual`
- 当前定位：
  - `ExactMIPSIndex` 是精确因子空间 top-k 扫描索引。
  - `RandomProjectionANNMIPSIndex` 是轻量 ANN/PQ 占位原型，不等同于最终论文级 PQ。
- 能回答的问题：
  - 压缩域索引是否比直接 dense reconstruction 扫描更有系统意义？
  - 残差 rerank 是否会牺牲太多 latency？
  - 未来是否值得继续实现真正 PQ/MIPS？

## 阶段 C：Exp4_v2 残差-查询鲁棒性

- 新增脚本：
  - `scripts/exp4_v2/run_residual_query_robustness.py`
- 新增配置：
  - `experiments/configs/exp4_v2/residual_query_robustness.yaml`
- 输出目录：
  - `experiments/results/exp4_v2/residual_query_robustness/`
- 名称要求：
  - 使用 `Exp4_v2`。
  - 旧 Exp4 保留为历史诊断，不再把它解释为残差修正证据。
- 对应系统架构：
  - robust sparse residual separation
  - residual store
  - Q1/Q4 residual correction
  - Q5 anomaly detection
  - storage budget diagnostics
- 与旧 Exp4 的区别：
  - 旧 Exp4 主要测攻击下低秩因子表示和社区 NMI 是否崩溃。
  - Exp4_v2 直接测残差是否命中攻击边、是否改善攻击边上的 Q1/Q4 查询、是否支持 Q5 异常检测。
- 主要指标：
  - `residual_precision`
  - `residual_recall`
  - `residual_f1`
  - `q5_precision`
  - `q5_recall`
  - `q5_f1`
  - `q1_attack_rmse_factor_only`
  - `q1_attack_rmse_factor_residual`
  - `q1_attack_rmse_improvement`
  - `q4_attack_rmse_improvement`
  - `compressed_vs_raw_sparse_ratio`
- 能回答的问题：
  - 鲁棒残差是不是一个有价值的纠错层？
  - 残差是在修正查询，还是只是在降低训练重构误差？
  - 在存储预算下，保留残差是否值得？

## 阶段 D：ARD 小闭环诊断与修复

- 修改文件：
  - `src/spectralstore/compression/spectral.py`
- 新增脚本：
  - `scripts/exp5/run_ard_diagnostic.py`
- 新增配置：
  - `experiments/configs/exp5/ard_diagnostic.yaml`
- 输出目录：
  - `experiments/results/exp5/ard_diagnostic/`
- 当前修复：
  - ARD shrinkage 只用于选择有效分量。
  - 选择后使用保留的原始 SVD 分量重新投影回最终 store。
  - 这样避免 shrinkage 尺度污染最终重构。
- 本地 smoke 观察：
  - 精确低秩数据上，ARD 从 `eff=3, frobenius≈1.72` 修回 `eff=3, frobenius≈0.057`。
  - SBM observed 场景中，ARD 仍倾向保留 max rank，说明 SBM 的“真秩”定义和噪声结构仍需单独诊断。
- 能回答的问题：
  - ARD 当前问题是否主要来自尺度 bug？
  - 修复后它能否至少通过精确低秩小闭环？
  - 它是否能进入主线，还是只能作为机制诊断？

## HPC 运行入口

当前 A-D 推荐入口：

```bash
cd /root/autodl-tmp/SpectralStore
conda activate spectralstore
export SPECTRAL_OUTPUT_DIR=/root/autodl-tmp/spectral_outputs
bash scripts/hpc/run_system_direction.sh
```

会依次运行：

- Exp3 查询与索引 benchmark
- Exp4_v2 残差-查询鲁棒性
- Exp5 ARD diagnostic

日志位置：

```text
/root/autodl-tmp/spectral_outputs/logs/
```

结果位置：

```text
experiments/results/exp3/query_benchmark/
experiments/results/exp4_v2/residual_query_robustness/
experiments/results/exp5/ard_diagnostic/
```

## 下一步解释原则

- 如果 Exp3 显示压缩域查询 latency/存储比有清晰收益，系统主线成立。
- 如果 Exp4_v2 显示残差能以可控存储命中异常并修正 Q1/Q4/Q5，鲁棒 AQP 主线成立。
- 如果 ARD 只在精确低秩成立、在 SBM 和真实图上不稳定，就不要把 ARD 作为论文核心贡献。
- 如果索引只在小规模不明显，需要继续做真正 PQ/MIPS；但这属于系统增强，不一定是当前论文主线的必要条件。
