# SpectralStore 科研进度报告

更新时间：2026-04-27

本报告以仓库当前磁盘状态为准，重点区分“当前主结果”和“legacy/diagnostic 结果”。旧 Exp1 结果仍可能留在 `experiments/results/exp1*`，但当前理论验证应以 Exp1-v2 及机制诊断为准。

## 1. 当前系统实现

### 1.1 压缩与存储

当前已经实现：

- `FactorizedTemporalStore`
  - 低秩左右因子、时间因子、奇异值
  - CSR residual 存储
  - storage accounting
  - NPZ round-trip
  - entrywise bound metadata
- compression registry
  - `spectralstore_asym`
  - `spectralstore_split_asym_unfolding`
  - `spectralstore_unfolding_asym`
  - `spectralstore_robust`
  - `sym_svd`
  - `direct_svd`
  - `rpca_svd`
  - `cp_als`
  - `tucker_als` / `tucker_hosvd`

注意：

- `sym_svd` 和 `direct_svd` 按 baseline 视为稳定实现。
- CP/Tucker 是 TensorLy-backed dense baseline，大规模 dense tensor 上不可作为主线依赖。

### 1.2 Query 层

当前已经实现：

- Q1 `LINK_PROB`
- Q2 `TOP_NEIGHBOR`
- Q3 `COMMUNITY`
- Q4 `TEMPORAL_TREND`
- Q5 `ANOMALY_DETECT`
- `QueryEngine.from_config`
- Q1 bounded result:

```text
estimate
bound
used_residual
method
```

校准参数：

- `query.bound_C`
- `robust.threshold_scale`

这两个是 YAML/config 参数，不是理论常数。

### 1.3 数据与实验框架

当前已经接入：

- Bitcoin-OTC
- Synthetic-SBM
- Synthetic-Spiked
- Synthetic-Attack
- sparse corruption masks
- ogbl-collab preliminary
- Exp1-v2 theory-regime SBM
- temporal-correlated SBM

实验框架支持：

- YAML/OmegaConf
- strict override key validation
- CSV append
- resume / skip completed settings
- `metrics.json`
- `summary.md`
- `resolved_config.yaml`
- `run_metadata.json`

## 2. 当前应引用的主结果

### 2.1 Exp1-v2 theory-regime

当前 Exp1 理论验证以 Exp1-v2 为准。

结果目录：

```text
experiments/results/exp1_v2/standard/
experiments/results/exp1_v2/hetero/
```

旧目录：

```text
experiments/results/exp1/
experiments/results/exp1_smoke/
experiments/results/exp1_pilot/
experiments/results/exp1_low_snr_smoke/
```

这些是 legacy/diagnostic，不再作为当前理论验证主结果。

#### Exp1-v2 standard

设置：

- noise: `iid`
- regimes: `medium_snr`, `low_snr`
- methods:
  - `spectralstore_asym`
  - `spectralstore_split_asym_unfolding`
  - `sym_svd`
  - `direct_svd`

关键结果：

| 现象 | 数值 | 解释 |
|---|---:|---|
| closest slope to -0.5 | `-0.378889` | 来自 `n_sweep / spectralstore_asym / low_snr / iid` |
| asym max-error ratio vs `sym_svd` | `1.20647` | 大于 1，说明 mean max error 没有优势 |
| asym variance ratio vs `sym_svd` | `6.71843` | 方差更大 |

结论：

- `spectralstore_asym` 在部分 slope 上比 `sym_svd` 更接近理论形状。
- 但当前 standard 结果不支持 asym 在 mean max-entrywise error 或 variance 上优于 `sym_svd`。

#### Exp1-v2 hetero

设置：

- noise: `heteroskedastic_entry`
- regimes: `medium_snr`, `low_snr`

关键结果：

| 现象 | 数值 | 解释 |
|---|---:|---|
| closest slope to -0.5 | `-0.51133` | 来自 `T_sweep / spectralstore_asym / medium_snr / heteroskedastic_entry` |
| asym max-error ratio vs `sym_svd` | `1.25189` | 大于 1，mean max error 仍无优势 |
| asym variance ratio vs `sym_svd` | `7.57832` | 方差仍更大 |

结论：

- heteroskedastic setting 中 slope 更接近理论参考。
- 但误差均值和方差仍不支持 asym 优势。

### 2.2 Asym temporal dependence

结果目录：

```text
experiments/results/exp_asym_temporal_dependence/
```

目标：

验证 temporal dependence 是否会让 asym 机制产生价值。

模型：

```text
A_t = M* + H_t
H_t = alpha H_{t-1} + epsilon_t
```

关键结果：

| alpha | asym error ratio vs `sym_svd` | asym variance ratio vs `sym_svd` |
|---:|---:|---:|
| `0.0` | `1.01348` | `1.6929` |
| `0.3` | `1.00975` | `1.43124` |
| `0.6` | `1.00588` | `1.39263` |
| `0.9` | `0.999609` | `1.53983` |

结论：

- `alpha=0` 复现 iid 下 asym 无明显优势。
- 随 alpha 增大，error ratio 有下降趋势。
- `alpha=0.9` 出现极弱 mean error 优势。
- variance ratio 始终大于 1，没有观察到 variance 优势。

### 2.3 Asym mechanism audit

结果目录：

```text
experiments/results/asym_mechanism_audit_full/
```

目标：

审计非对称机制是否真实生效，分构造层、分解层、后处理层、方差层。

关键结果：

| 问题 | 当前结论 | 证据 |
|---|---|---|
| asym matrix 是否真的非对称 | 是 | mean `asymmetry_norm=0.58916` |
| U/V 是否真的不同 | 是 | mean `||U-V||_F/||U||_F=0.365997` |
| 输出是否被强制对称化 | 否 | mean `output_asymmetry_norm=0.130484`, `forced_symmetric=False` |
| `spectralstore_asym` 与 split asym 是否等价 | 几乎等价 | mean `reconstruction_diff_asym_vs_split=2.56789e-15` |
| asym variance 是否小于 sym | 否 | mean `asym_variance_ratio_vs_sym=1.87308` |

解释：

- 非对称构造确实存在。
- U/V 也没有 collapse。
- 输出没有被后处理强制对称化。
- 但 `spectralstore_asym` 与 `spectralstore_split_asym_unfolding` 当前几乎完全等价。
- 更大的方差更可能来自 sample split 降低有效样本量、U/V 随 seed 更敏感、理论条件与当前 synthetic 设置不完全匹配，而不是后处理把 asym gain 抹掉。

### 2.4 Calibration

结果目录：

```text
experiments/preliminary/thinking_alignment/results_n500_t20_r3_calibration/calibration/
```

关键结论：

| 参数 | 当前经验默认 | 说明 |
|---|---:|---|
| `query.bound_C` | `3` | 来自当前 coverage calibration，不是全局理论常数 |
| `robust.threshold_scale` | `3` | 来自 sparse-corruption threshold calibration，不是所有数据集最优 |

### 2.5 Exp2 Bitcoin-OTC

结果目录：

```text
experiments/results/exp2/
```

当前用途：

- compression ratio sweep
- storage ratio
- max/mean/frobenius error
- held-out RMSE

注意：

- 这是系统/真实数据实验，不是 Thinking entrywise theory 的主验证。
- 当前真实数据 preliminary 中 `sym_svd` / `direct_svd` 往往不弱于 `spectralstore_asym`。

### 2.6 Exp4 robustness

结果目录：

```text
experiments/results/exp4/
```

当前用途：

- random attack
- targeted attack
- RPCA+SVD baseline
- robust residual diagnostics

注意：

- random/targeted attack 是 robustness diagnostic。
- Thinking 严格 sparse corruption 模型更应引用 alignment sparse-corruption check。

## 3. 当前主结论

1. 系统层面已经跑通：
   - compression registry
   - query engine
   - bound calibration
   - robust residual
   - resume experiments

2. 理论验证层面，当前不能宣称：

```text
spectralstore_asym stably outperforms sym_svd
```

3. 更准确的说法是：

```text
spectralstore_asym 在部分 setting 下呈现更接近理论的 scaling slope，
但 mean max-entrywise error 和 variance 证据尚不支持其稳定优于 sym_svd。
```

4. 机制审计说明：
   - asym construction 是真实的。
   - U/V 是不同的。
   - 输出没有被强制对称化。
   - 当前 `spectralstore_asym` 与 split asym 几乎等价。

5. robust residual 是当前最强的正向结果链之一：
   - sparse corruption 下 residual recovery 明确有效。
   - threshold calibration 给出了可配置经验默认。

## 4. 已知不准确或需要谨慎的表述

下面这些表述不应继续使用：

| 旧表述 | 问题 | 修正 |
|---|---|---|
| “Exp1 已完成理论验证” | 旧 Exp1 是 legacy，且 slope/优势证据不足 | 改为 Exp1-v2 是当前理论验证主结果，但结论保守 |
| “asym 优于 sym_svd” | 当前主要结果不支持稳定优势 | 改为部分 slope 更像理论，但 error/variance 不占优 |
| “split asym 是独立新机制并明显不同于 asym” | audit 显示两者 reconstruction 几乎等价 | 改为 split asym 是构造对齐版本，但当前与 asym 几乎等价 |
| “temporal dependence 已证明 asym 有用” | alpha=0.9 只有极弱 error 优势，variance 仍更大 | 改为 temporal dependence 下 error ratio 有下降趋势，但证据仍弱 |
| “bound C=3 是理论常数” | C=3 是经验校准值 | 改为 configurable calibrated default |
| “threshold_scale=3 是全局最优” | 只对当前 sparse-corruption sweep 成立 | 改为 configurable calibrated default |
| “Exp2 是理论验证” | Exp2 是真实数据系统/storage sweep | 改为 Exp2 是系统实验 |
| “Exp4 random/targeted attack 等于 Thinking sparse corruption” | attack 设计不同 | 改为 robustness diagnostic；严格 sparse corruption 用 alignment check |

## 5. 下一步建议

1. 对外叙事以 Exp1-v2 + temporal dependence + mechanism audit + calibration 组成理论诊断链。
2. 不再引用旧 Exp1 作为主理论实验，只保留 legacy/diagnostic 角色。
3. 如果继续追 asym 优势，应优先解释：
   - 为什么 split/asym 几乎等价；
   - 为什么 asym variance 更大；
   - 哪些 temporal/noise regime 会让 error ratio 接近或低于 1。
4. robustness 主线可以继续推进，因为它当前证据更稳定。

