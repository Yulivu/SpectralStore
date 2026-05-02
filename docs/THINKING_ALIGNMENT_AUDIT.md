# Thinking Alignment Audit

Last updated: 2026-05-02

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
| **I — Tensor unfold + asym construction** | mode-3 unfold n×nT, random bipartition T1/T2, upper-tri from T1-mean, lower-tri from T2-mean | ✅ | `_mode3_thinking_tensor_basis()` + `_thinking_asymmetric_basis_with_mode3()` — mode-3 unfold is the primary path for `UnifiedThinkingSpectralCompressor`; legacy mode-1+2 Thinking helper removed |
| **II — Bayesian ARD rank** | Gamma(alpha0,beta0) prior, VI, SVD warm-start, ELBO convergence → prune | ✅ | `_ard_variational_shrinkage()`, `rank_selection_mode=ard` |
| **III — Robust alternating separation** | hard-threshold S_t, re-decompose A_t - S_t, iterate | ✅ | `UnifiedThinkingSpectralCompressor` |
| **I–III unified offline loop** | mode-3 + asym + ARD + robust in a single alternating loop | ✅ | `_run_unified_thinking_loop()` — ARD runs inside each robust iteration; `UnifiedThinkingSpectralCompressor` registered as `spectralstore_thinking` |
| **IV — Incremental update** | project A_{T+1} → update temporal vector → residual check → full refactor if needed | ❌ | no `partial_fit` / `update` API; store is `frozen=True` |

---

## 3.  Query API (Thinking §2.1 & §2.4)

| Query | Status | Bound-formula alignment |
|---|---|---|
| Q1 LINK_PROB | ✅ | `link_prob_with_error()` returns `BoundedQueryResult`; `spectralstore_thinking` populates exact `bound_sigma_max`, `bound_mu`, `bound_constant` → `store.entrywise_bound()` computes exact Thinking formula `C*sigma_max*sqrt(r*log(n)/(n*T))*(1/sqrt(mu_u)+1/sqrt(mu_v))` |
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
| NMF | §4.1 spectral | ✅ |
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

**Total baselines implemented: 8 of 18** |

---

## 6.  Experiments (Thinking §5)

| Experiment | Thinking goal | Status | Notes |
|---|---|---|---|
| **Exp1** entrywise bound | SBM + Spiked, n/T sweep, log-log power-law fit | ✅ | `run_exp1_theory_validation.py`, output -> `exp1/` |
| **Exp2** compression accuracy + storage | Bitcoin/ogbl-collab/Reddit, ratio 1%–50% | ⚠️ | Bitcoin sweep ✅, ogbl-collab ❌, Reddit ❌ |
| **Exp3** query latency | Q1–Q5 vs Raw/Neo4j, ogbl-collab/Reddit/StackOverflow | ⚠️ | Q1-Q5 compressed-domain benchmark ✅, Q2 NoIndex/Exact/ANN ✅, Neo4j ❌, Reddit/StackOverflow loaders ❌ |
| **Exp4 / Exp4_v2** robustness | Random/Targeted/Injection 0%–30%, NoRobust ablation | ⚠️ | legacy Exp4 attack modes ✅; Exp4_v2 residual anomaly + query correction ✅; NoRobust ablation still partial |
| **Exp5** ARD rank selection | ARD vs cross-validation, SBM + ogbl-collab | ⚠️ | ARD core ✅, rank script ✅, small-loop diagnostic ✅, SBM/real-data reliability still unresolved |
| **Exp6** scalability | n 1K→1M, StackOverflow 10K→500K | ❌ | no script, `>=50K` still pending |
| **Exp7** ablation | NoAsym/NoBayes/NoRobust/NoResidual/NoIndex | ⚠️ | individual variants usable, no unified ablation runner |

---

## 7.  Implementation Recommendations (Thinking §6)

| Recommendation | Status |
|---|---|
| Python + NumPy/SciPy core | ✅ |
| TensorLy (CP/Tucker) | ✅ |
| scikit-learn (KMeans, NMF) | ✅ (KMeans + NMF) |
| PyTorch Geometric Temporal | ❌ |
| OGB library | ✅ |
| Separate `data_loader` / `compression` / `query_engine` / `index` / `evaluation` / `baselines` modules | ✅ all except `baselines` |
| YAML config management | ✅ OmegaConf |
| Reproducibility (seed, env recording) | ✅ `set_reproducibility_seed` + `run_metadata.json` |

---

## 8.  Summary

### ✅ Fully implemented

- Asymmetric construction (random bipartition + triangular stitching)
- **Mode-3 tensor unfold** (`_mode3_thinking_tensor_basis`, primary path in `UnifiedThinkingSpectralCompressor`)
- ARD variational rank selection with ELBO diagnostics
- **Unified offline loop**: mode-3 + asym + ARD + robust in `_run_unified_thinking_loop()`
- Robust alternating sparse separation inside `spectralstore_thinking`
- **Exact entrywise bound formula**: `store.entrywise_bound()` computes `C*sigma_max*sqrt(r*log(n)/(n*T))*(1/sqrt(mu_u)+1/sqrt(mu_v))` — populated by `spectralstore_thinking`
- Factorized storage + COO/CSR residuals + storage gate (`drop_residual` / `raise` / `diagnostic`)
- Q1–Q5 all queries with error bounds and optimized paths
- Sparse-native paths for stable SVD baselines; `spectralstore_thinking` sparse mode deferred
- Experiment contract (YAML → CSV + `metrics.json` + `summary.md`)
- NMF baseline
- Exp4 all three attack modes: Random, Targeted, Injection
- Exp4_v2 residual anomaly capture and Q1/Q4/Q5 correction benchmark
- Exp5 ARD rank selection script
- Exp5 ARD small-loop diagnostic script
- **Current compressor**: `spectralstore_thinking` (`UnifiedThinkingSpectralCompressor`)

### ❌ Not implemented (strategically deferred)

- Incremental/online update (Thinking §2.3 Stage IV)
- External baselines deferred: BPTF, COSTCO, DynGEM, EvolveGCN, ROLAND, TGN, SWeG, SSumM, SpectralSparsification, TCM, Pro-GNN, GCN-Jaccard
- 7 additional dataset loaders (UCI, Enron, Reddit, DBLP, StackOverflow, FB15k-237, ICEWS)
- Standalone Exp6/Exp7 scripts
- PQ index

### ⚠️ Partial or formally divergent

- Exp2 covers only Bitcoin, not ogbl-collab or Reddit
- Exp3 lacks Neo4j/Cypher baseline and Reddit/StackOverflow data
- ARD rank selection now avoids the shrinkage-scale reconstruction bug, but still
  needs reliability evidence beyond exact low-rank diagnostics.
