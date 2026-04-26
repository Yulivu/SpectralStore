# Thinking Alignment Checks

This run checks implementation behavior against Thinking.docx and SpectralStore.md without changing the research target.

## Split Asym Sanity

- Thinking triangular construction compliant: `True`
- mean T1/T2 noise correlation: `0.00159278`
- mean reconstruction difference vs spectralstore_asym: `7.20957e-15`
- equivalence diagnosis: equivalent: spectralstore_asym already uses one split triangular mean; SBM diagonal is zero

## Entrywise Bound Coverage

| method | coverage | violation rate | median bound/error | max violation margin |
|---|---:|---:|---:|---:|
| direct_svd | 0.94431 | 0.0556895 | 3.80039 | 0.344452 |
| spectralstore_asym | 0.914207 | 0.0857933 | 3.09571 | 0.358594 |
| spectralstore_split_asym_unfolding | 0.914207 | 0.0857933 | 3.09571 | 0.358594 |
| sym_svd | 0.9591 | 0.0409001 | 4.44472 | 0.345014 |

Coverage near 1.0 supports using the bound in Q1; large median bound/error indicates looseness.
Diagnosis: C=1 bound has nonzero violations in this smoke setting; treat it as an uncalibrated theoretical diagnostic before query-layer SLA use.

## Robust Sparse Corruption

| setting | max err | precision | recall | f1 | residual storage |
|---|---:|---:|---:|---:|---:|
| block_sparse_spike|0.02|direct_svd | 1.00784 | 0 | 0 | 0 | 0 |
| block_sparse_spike|0.02|rpca_svd | 1.02678 | 0 | 0 | 0 | 0 |
| block_sparse_spike|0.02|spectralstore_asym | 1.05859 | 0 | 0 | 0 | 0 |
| block_sparse_spike|0.02|spectralstore_robust | 0.387471 | 0.998 | 1 | 0.998999 | 0.031002 |
| block_sparse_spike|0.02|sym_svd | 0.983519 | 0 | 0 | 0 | 0 |
| block_sparse_spike|0.0|direct_svd | 0.369315 | 0 | 0 | 0 | 0 |
| block_sparse_spike|0.0|rpca_svd | 0.375574 | 0 | 0 | 0 | 0 |
| block_sparse_spike|0.0|spectralstore_asym | 0.384051 | 0 | 0 | 0 | 0 |
| block_sparse_spike|0.0|spectralstore_robust | 0.394755 | 0 | 0 | 0 | 0.031002 |
| block_sparse_spike|0.0|sym_svd | 0.369776 | 0 | 0 | 0 | 0 |
| signed_spike|0.02|direct_svd | 0.395243 | 0 | 0 | 0 | 0 |
| signed_spike|0.02|rpca_svd | 0.396075 | 0 | 0 | 0 | 0 |
| signed_spike|0.02|spectralstore_asym | 0.421691 | 0 | 0 | 0 | 0 |
| signed_spike|0.02|spectralstore_robust | 0.3864 | 0.998 | 1 | 0.998999 | 0.031002 |
| signed_spike|0.02|sym_svd | 0.397433 | 0 | 0 | 0 | 0 |
| signed_spike|0.0|direct_svd | 0.369315 | 0 | 0 | 0 | 0 |
| signed_spike|0.0|rpca_svd | 0.375574 | 0 | 0 | 0 | 0 |
| signed_spike|0.0|spectralstore_asym | 0.384051 | 0 | 0 | 0 | 0 |
| signed_spike|0.0|spectralstore_robust | 0.394755 | 0 | 0 | 0 | 0.031002 |
| signed_spike|0.0|sym_svd | 0.369776 | 0 | 0 | 0 | 0 |
| sparse_spike|0.02|direct_svd | 0.485298 | 0 | 0 | 0 | 0 |
| sparse_spike|0.02|rpca_svd | 0.478954 | 0 | 0 | 0 | 0 |
| sparse_spike|0.02|spectralstore_asym | 0.499628 | 0 | 0 | 0 | 0 |
| sparse_spike|0.02|spectralstore_robust | 0.385384 | 0.998 | 1 | 0.998999 | 0.031002 |
| sparse_spike|0.02|sym_svd | 0.486869 | 0 | 0 | 0 | 0 |
| sparse_spike|0.0|direct_svd | 0.369315 | 0 | 0 | 0 | 0 |
| sparse_spike|0.0|rpca_svd | 0.375574 | 0 | 0 | 0 | 0 |
| sparse_spike|0.0|spectralstore_asym | 0.384051 | 0 | 0 | 0 | 0 |
| sparse_spike|0.0|spectralstore_robust | 0.394755 | 0 | 0 | 0 | 0.031002 |
| sparse_spike|0.0|sym_svd | 0.369776 | 0 | 0 | 0 | 0 |

Robust is aligned with Thinking.docx when residuals recover sparse S_t with useful precision/recall without excessive residual storage.
Diagnosis: positive-corruption robust mean F1 is `0.998999`; no-corruption robust residual storage is `0.031002`, so threshold calibration should be checked for false residual storage.


## CSV Outputs

- split_asym_sanity: `split_asym_sanity.csv`
- entrywise_bound_coverage: `entrywise_bound_coverage.csv`
- robust_sparse_corruption: `robust_sparse_corruption.csv`
