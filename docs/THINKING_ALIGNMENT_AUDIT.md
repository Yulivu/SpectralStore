# Thinking Alignment Audit

Last updated: 2026-05-01

This document maps every substantive claim in `Thinking.docx` onto the current
codebase and marks each as complete, partial, or not implemented.  It serves as
the baseline for "先把现有实现按Thinking理论彻底做好".

Legend: ✅ complete / ⚠️ partial (explained) / ❌ not implemented / 🔵 deferred

---

## 1.  Architecture (Thinking §2.2)

| Layer | Thinking spec | Status | Where |
|---|---|---|---|
| Factored Store | U(n×r), V(n×r), W(T×r), λ(1×r), sparse residual CSR | ✅ | `FactorizedTemporalStore` + `TemporalCOOResidualStore` |
| Compression Engine | batch + incremental | ⚠️ | batch ✅, incremental ❌ |
| Query Processor | compressed-domain → bound → residual fallback | ✅ | `*_optimized()` paths + `BoundedQueryResult` |
| Index Layer | PQ + temporal B-tree + inverted | ⚠️ | ExactMIPS ✅, RandomProjectionANN placeholder ✅, PQ ❌, temporal-index ❌, inverted-index ❌ |

---

## 2.  Core Algorithm (Thinking §2.3)

| Stage | Thinking description | Status | Evidence |
|---|---|---|---|
| **I — Tensor unfold + asym construction** | mode-3 unfold n×nT, random bipartition T₁/T₂, upper-tri from T₁-mean, lower-tri from T₂-mean | ⚠️ | random bipartition + triangular stitching ✅ (`_asymmetric_basis`); mode-3 unfold not the primary path — `TensorUnfoldingSVDCompressor` uses mode-1+mode-2 |
| **II — Bayesian ARD rank** | Gamma(α₀,β₀) prior, VI, SVD warm-start, ELBO convergence → prune | ✅ | `_ard_variational_shrinkage()`, `rank_selection_mode=ard` |
| **III — Robust alternating separation** | hard-threshold Ŝ_t, re-decompose A_t − Ŝ_t, iterate | ✅ | `RobustAsymmetricSpectralCompressor`, `AlternatingRobustAsymmetricSpectralCompressor` |
| **IV — Incremental update** | project A_{T+1} → update temporal vector → residual check → full refactor if needed | ❌ | no `partial_fit` / `update` API; store is `frozen=True` |

---

## 3.  Query API (Thinking §2.1 & §2.4)

| Query | Status | Bound-formula alignment |
|---|---|---|
| Q1 LINK_PROB | ✅ | `link_prob_with_error()` returns `BoundedQueryResult`; bound is adaptive, not the exact `C·σ_max·√(r log n/(nT))·(1/√μ_u + 1/√μ_v)` formula |
| Q2 TOP_NEIGHBOR | ✅ | ExactMIPS + RandomProjectionANN |
| Q3 COMMUNITY | ✅ | KMeans on weighted embeddings |
| Q4 TEMPORAL_TREND | ✅ | cached + error-aware variants |
| Q5 ANOMALY_DETECT | ✅ | direct sparse-residual lookup |

---

## 4.  Datasets (Thinking §3)

### 4.1 Synthetic

| Dataset | Status |
|---|---|
| Synthetic-SBM | ✅ |
| Synthetic-Spiked | ✅ |
| Synthetic-Attack — Random | ✅ |
| Synthetic-Attack — Targeted | ✅ |
| Synthetic-Attack — Injection | ❌ |

### 4.2 Real-world (small/medium)

| Dataset | Status |
|---|---|
| Bitcoin-OTC | ✅ |
| Bitcoin-Alpha | ✅ |
| UCI Messages | ❌ |
| Enron Email | ❌ |

### 4.3 Real-world (medium/large)

| Dataset | Status |
|---|---|
| ogbl-collab | ✅ |
| Reddit Hyperlink | ❌ |
| DBLP | ❌ |
| Stack Overflow | ❌ |

### 4.4 Knowledge graphs

| Dataset | Status |
|---|---|
| FB15k-237 | ❌ |
| ICEWS | ❌ |

---

## 5.  Baselines (Thinking §4)

| Baseline | Category | Status |
|---|---|---|
| SymSVD | §4.1 spectral | ✅ |
| DirectSVD | §4.1 spectral | ✅ |
| NMF | §4.1 spectral | ❌ |
| CP-ALS | §4.2 tensor | ✅ |
| Tucker-ALS | §4.2 tensor | ✅ |
| BPTF | §4.2 tensor | ❌ |
| COSTCO | §4.2 tensor | ❌ |
| RPCA+SVD | §4.5 robust | ✅ |
| DynGEM | §4.3 dynamic-graph | ❌ |
| EvolveGCN | §4.3 dynamic-graph | ❌ |
| ROLAND | §4.3 dynamic-graph | ❌ |
| TGN | §4.3 dynamic-graph | ❌ |
| SWeG | §4.4 graph-summary | ❌ |
| SSumM | §4.4 graph-summary | ❌ |
| Spectral Sparsification | §4.4 graph-summary | ❌ |
| TCM | §4.4 graph-summary | ❌ |
| Pro-GNN | §4.5 robust | ❌ |
| GCN-Jaccard | §4.5 robust | ❌ |

---

## 6.  Experiments (Thinking §5)

| Experiment | Thinking goal | Status | Notes |
|---|---|---|---|
| **Exp1** entrywise bound | SBM + Spiked, n/T sweep, log-log power-law fit | ✅ | `run_exp1_theory_validation.py`, output → `exp1_v2/` |
| **Exp2** compression accuracy + storage | Bitcoin/ogbl-collab/Reddit, ratio 1%–50% | ⚠️ | Bitcoin sweep ✅, ogbl-collab ❌, Reddit ❌ |
| **Exp3** query latency | Q1–Q5 vs Raw/Neo4j, ogbl-collab/Reddit/StackOverflow | ⚠️ | latency benchmark exists, Neo4j ❌, Reddit/StackOverflow loaders ❌ |
| **Exp4** robustness | Random/Targeted/Injection 0%–30%, NoRobust ablation | ⚠️ | Random + Targeted ✅, Injection ❌, ablation partially available |
| **Exp5** ARD rank selection | ARD vs cross-validation, SBM + ogbl-collab | ⚠️ | ARD core ✅, standalone Exp5 script ❌ |
| **Exp6** scalability | n 1K→1M, StackOverflow 10K→500K | ❌ | no script, `>=50K` still pending |
| **Exp7** ablation | NoAsym/NoBayes/NoRobust/NoResidual/NoIndex | ⚠️ | individual variants usable, no unified ablation runner |

---

## 7.  Implementation Recommendations (Thinking §6)

| Recommendation | Status |
|---|---|
| Python + NumPy/SciPy core | ✅ |
| TensorLy (CP/Tucker) | ✅ |
| scikit-learn (KMeans, NMF) | ✅ (KMeans), NMF ❌ |
| PyTorch Geometric Temporal | ❌ |
| OGB library | ✅ |
| Separate `data_loader` / `compression` / `query_engine` / `index` / `evaluation` / `baselines` modules | ✅ all except `baselines` |
| YAML config management | ✅ OmegaConf |
| Reproducibility (seed, env recording) | ✅ `set_reproducibility_seed` + `run_metadata.json` |

---

## 8.  Summary

### ✅ Fully implemented

- Asymmetric construction (random bipartition + triangular stitching)
- ARD variational rank selection with ELBO diagnostics
- Robust alternating sparse separation (`spectralstore_robust`, `spectralstore_asym_alternating_robust`)
- Factorized storage + COO/CSR residuals + storage gate (`drop_residual` / `raise` / `diagnostic`)
- Q1–Q5 all queries with error bounds and optimized paths
- Sparse-native paths for four compressors
- Experiment contract (YAML → CSV + `metrics.json` + `summary.md`)

### ❌ Not implemented (strategically deferred)

- Incremental/online update (Thinking §2.3 Stage IV)
- 13 external baselines (NMF, BPTF, DynGEM, EvolveGCN, ROLAND, TGN, SWeG, SSumM, SpectralSparsification, TCM, Pro-GNN, GCN-Jaccard, COSTCO)
- 7 additional dataset loaders (UCI, Enron, Reddit, DBLP, StackOverflow, FB15k-237, ICEWS)
- Injection attack mode for Exp4
- Standalone Exp5/Exp6/Exp7 scripts
- PQ index

### ⚠️ Partial or formally divergent

- Entrywise bound formula uses adaptive bound, not the exact Thinking formula
- `TensorUnfoldingSVDCompressor` uses mode-1/2 unfold, not mode-3
- Asymmetric + ARD + robust not yet combined into a single unified offline loop
- Exp2 covers only Bitcoin, not ogbl-collab or Reddit
- Exp3 lacks Neo4j/Cypher baseline and Reddit/StackOverflow data
