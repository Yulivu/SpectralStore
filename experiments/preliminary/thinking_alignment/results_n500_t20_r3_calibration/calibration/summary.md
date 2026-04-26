# Thinking Alignment Checks

This run checks implementation behavior against Thinking.docx and SpectralStore.md without changing the research target.

## Entrywise Bound Calibration

| method | C | coverage | violation rate | mean bound/error | median bound/error | max violation margin |
|---|---:|---:|---:|---:|---:|---:|
| direct_svd | 1 | 0.94431 | 0.0556895 | 33.4209 | 3.80039 | 0.344452 |
| direct_svd | 1.5 | 0.9829 | 0.0170996 | 50.1313 | 5.70059 | 0.33202 |
| direct_svd | 2 | 0.994693 | 0.00530707 | 66.8417 | 7.60079 | 0.319588 |
| direct_svd | 3 | 0.997937 | 0.00206307 | 100.263 | 11.4012 | 0.294777 |
| direct_svd | 5 | 0.998 | 0.002 | 167.104 | 19.002 | 0.245299 |
| spectralstore_asym | 1 | 0.914207 | 0.0857933 | 26.718 | 3.09571 | 0.358594 |
| spectralstore_asym | 1.5 | 0.968532 | 0.0314685 | 40.077 | 4.64357 | 0.345866 |
| spectralstore_asym | 2 | 0.988735 | 0.0112651 | 53.436 | 6.19142 | 0.333138 |
| spectralstore_asym | 3 | 0.997556 | 0.00244353 | 80.154 | 9.28714 | 0.307681 |
| spectralstore_asym | 5 | 0.998 | 0.002 | 133.59 | 15.4786 | 0.257033 |
| spectralstore_split_asym_unfolding | 1 | 0.914207 | 0.0857933 | 26.718 | 3.09571 | 0.358594 |
| spectralstore_split_asym_unfolding | 1.5 | 0.968532 | 0.0314685 | 40.077 | 4.64357 | 0.345866 |
| spectralstore_split_asym_unfolding | 2 | 0.988735 | 0.0112651 | 53.436 | 6.19142 | 0.333138 |
| spectralstore_split_asym_unfolding | 3 | 0.997556 | 0.00244353 | 80.154 | 9.28714 | 0.307681 |
| spectralstore_split_asym_unfolding | 5 | 0.998 | 0.002 | 133.59 | 15.4786 | 0.257033 |
| sym_svd | 1 | 0.9591 | 0.0409001 | 38.9604 | 4.44472 | 0.345014 |
| sym_svd | 1.5 | 0.989298 | 0.0107023 | 58.4405 | 6.66708 | 0.332634 |
| sym_svd | 2 | 0.996506 | 0.0034936 | 77.9207 | 8.88944 | 0.320253 |
| sym_svd | 3 | 0.997992 | 0.0020076 | 116.881 | 13.3342 | 0.295491 |
| sym_svd | 5 | 0.998 | 0.002 | 194.802 | 22.2236 | 0.246207 |

### Calibration Decisions

- direct_svd: min C for coverage >= 0.99 is `2`; min C for coverage >= 0.995 is `3`.
- spectralstore_asym: min C for coverage >= 0.99 is `3`; min C for coverage >= 0.995 is `3`.
- spectralstore_split_asym_unfolding: min C for coverage >= 0.99 is `3`; min C for coverage >= 0.995 is `3`.
- sym_svd: min C for coverage >= 0.99 is `2`; min C for coverage >= 0.995 is `2`.
- Recommended default C: `3` across bounded methods in this sweep.
- Looseness should be judged by mean/median bound-over-error; very large ratios mean the bound is safe but weak for query precision.

## Residual Threshold Calibration

| method | scale | corruption | rate | f1 | precision | recall | residual storage | false positive residual ratio | storage ratio |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| direct_svd | 1 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| direct_svd | 1 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| direct_svd | 1 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| direct_svd | 1 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| direct_svd | 1 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| direct_svd | 1 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| direct_svd | 1.5 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| direct_svd | 1.5 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| direct_svd | 1.5 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| direct_svd | 1.5 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| direct_svd | 1.5 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| direct_svd | 1.5 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| direct_svd | 2 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| direct_svd | 2 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| direct_svd | 2 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| direct_svd | 2 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| direct_svd | 2 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| direct_svd | 2 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| direct_svd | 3 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| direct_svd | 3 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| direct_svd | 3 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| direct_svd | 3 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| direct_svd | 3 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| direct_svd | 3 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| direct_svd | 5 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| direct_svd | 5 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| direct_svd | 5 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| direct_svd | 5 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| direct_svd | 5 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| direct_svd | 5 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| rpca_svd | 1 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008229 |
| rpca_svd | 1 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008229 |
| rpca_svd | 1 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008229 |
| rpca_svd | 1 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008229 |
| rpca_svd | 1 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008229 |
| rpca_svd | 1 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008229 |
| rpca_svd | 1.5 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008229 |
| rpca_svd | 1.5 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008229 |
| rpca_svd | 1.5 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008229 |
| rpca_svd | 1.5 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008229 |
| rpca_svd | 1.5 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008229 |
| rpca_svd | 1.5 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008229 |
| rpca_svd | 2 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008229 |
| rpca_svd | 2 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008229 |
| rpca_svd | 2 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008229 |
| rpca_svd | 2 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008229 |
| rpca_svd | 2 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008229 |
| rpca_svd | 2 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008229 |
| rpca_svd | 3 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008229 |
| rpca_svd | 3 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008229 |
| rpca_svd | 3 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008229 |
| rpca_svd | 3 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008229 |
| rpca_svd | 3 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008229 |
| rpca_svd | 3 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008229 |
| rpca_svd | 5 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008229 |
| rpca_svd | 5 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008229 |
| rpca_svd | 5 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008229 |
| rpca_svd | 5 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008229 |
| rpca_svd | 5 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008229 |
| rpca_svd | 5 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008229 |
| spectralstore_asym | 1 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| spectralstore_asym | 1 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| spectralstore_asym | 1 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| spectralstore_asym | 1 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| spectralstore_asym | 1 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| spectralstore_asym | 1 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| spectralstore_asym | 1.5 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| spectralstore_asym | 1.5 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| spectralstore_asym | 1.5 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| spectralstore_asym | 1.5 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| spectralstore_asym | 1.5 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| spectralstore_asym | 1.5 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| spectralstore_asym | 2 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| spectralstore_asym | 2 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| spectralstore_asym | 2 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| spectralstore_asym | 2 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| spectralstore_asym | 2 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| spectralstore_asym | 2 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| spectralstore_asym | 3 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| spectralstore_asym | 3 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| spectralstore_asym | 3 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| spectralstore_asym | 3 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| spectralstore_asym | 3 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| spectralstore_asym | 3 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| spectralstore_asym | 5 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| spectralstore_asym | 5 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| spectralstore_asym | 5 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| spectralstore_asym | 5 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| spectralstore_asym | 5 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| spectralstore_asym | 5 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| spectralstore_robust | 1 | block_sparse_spike | 0 | 0 | 0 | 0 | 0.002445 | 0.000962 | 0.00347292 |
| spectralstore_robust | 1 | block_sparse_spike | 0.02 | 0.93533 | 0.878562 | 1 | 0.0350842 | nan | 0.0361121 |
| spectralstore_robust | 1 | signed_spike | 0 | 0 | 0 | 0 | 0.002445 | 0.000962 | 0.00347292 |
| spectralstore_robust | 1 | signed_spike | 0.02 | 0.971547 | 0.944729 | 1 | 0.0326978 | nan | 0.0337257 |
| spectralstore_robust | 1 | sparse_spike | 0 | 0 | 0 | 0 | 0.002445 | 0.000962 | 0.00347292 |
| spectralstore_robust | 1 | sparse_spike | 0.02 | 0.999346 | 0.998692 | 1 | 0.0309812 | nan | 0.0320091 |
| spectralstore_robust | 1.5 | block_sparse_spike | 0 | 0 | 0 | 0 | 0.001002 | 0 | 0.00202993 |
| spectralstore_robust | 1.5 | block_sparse_spike | 0.02 | 1 | 1 | 1 | 0.030942 | nan | 0.0319699 |
| spectralstore_robust | 1.5 | signed_spike | 0 | 0 | 0 | 0 | 0.001002 | 0 | 0.00202993 |
| spectralstore_robust | 1.5 | signed_spike | 0.02 | 1 | 1 | 1 | 0.030942 | nan | 0.0319699 |
| spectralstore_robust | 1.5 | sparse_spike | 0 | 0 | 0 | 0 | 0.001002 | 0 | 0.00202993 |
| spectralstore_robust | 1.5 | sparse_spike | 0.02 | 1 | 1 | 1 | 0.030942 | nan | 0.0319699 |
| spectralstore_robust | 2 | block_sparse_spike | 0 | 0 | 0 | 0 | 0.001002 | 0 | 0.00202993 |
| spectralstore_robust | 2 | block_sparse_spike | 0.02 | 1 | 1 | 1 | 0.030942 | nan | 0.0319699 |
| spectralstore_robust | 2 | signed_spike | 0 | 0 | 0 | 0 | 0.001002 | 0 | 0.00202993 |
| spectralstore_robust | 2 | signed_spike | 0.02 | 1 | 1 | 1 | 0.030942 | nan | 0.0319699 |
| spectralstore_robust | 2 | sparse_spike | 0 | 0 | 0 | 0 | 0.001002 | 0 | 0.00202993 |
| spectralstore_robust | 2 | sparse_spike | 0.02 | 1 | 1 | 1 | 0.030942 | nan | 0.0319699 |
| spectralstore_robust | 3 | block_sparse_spike | 0 | 0 | 0 | 0 | 0.001002 | 0 | 0.00202993 |
| spectralstore_robust | 3 | block_sparse_spike | 0.02 | 1 | 1 | 1 | 0.030942 | nan | 0.0319699 |
| spectralstore_robust | 3 | signed_spike | 0 | 0 | 0 | 0 | 0.001002 | 0 | 0.00202993 |
| spectralstore_robust | 3 | signed_spike | 0.02 | 1 | 1 | 1 | 0.030942 | nan | 0.0319699 |
| spectralstore_robust | 3 | sparse_spike | 0 | 0 | 0 | 0 | 0.001002 | 0 | 0.00202993 |
| spectralstore_robust | 3 | sparse_spike | 0.02 | 1 | 1 | 1 | 0.030942 | nan | 0.0319699 |
| spectralstore_robust | 5 | block_sparse_spike | 0 | 0 | 0 | 0 | 0.001002 | 0 | 0.00202993 |
| spectralstore_robust | 5 | block_sparse_spike | 0.02 | 0.000527499 | 0.666667 | 0.000263861 | 0.0010099 | nan | 0.00203783 |
| spectralstore_robust | 5 | signed_spike | 0 | 0 | 0 | 0 | 0.001002 | 0 | 0.00202993 |
| spectralstore_robust | 5 | signed_spike | 0.02 | 0.0357745 | 1 | 0.0182131 | 0.0015473 | nan | 0.00257522 |
| spectralstore_robust | 5 | sparse_spike | 0 | 0 | 0 | 0 | 0.001002 | 0 | 0.00202993 |
| spectralstore_robust | 5 | sparse_spike | 0.02 | 0.00336631 | 1 | 0.00168671 | 0.0010525 | nan | 0.00208042 |
| sym_svd | 1 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| sym_svd | 1 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| sym_svd | 1 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| sym_svd | 1 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| sym_svd | 1 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| sym_svd | 1 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| sym_svd | 1.5 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| sym_svd | 1.5 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| sym_svd | 1.5 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| sym_svd | 1.5 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| sym_svd | 1.5 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| sym_svd | 1.5 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| sym_svd | 2 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| sym_svd | 2 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| sym_svd | 2 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| sym_svd | 2 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| sym_svd | 2 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| sym_svd | 2 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| sym_svd | 3 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| sym_svd | 3 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| sym_svd | 3 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| sym_svd | 3 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| sym_svd | 3 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| sym_svd | 3 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| sym_svd | 5 | block_sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| sym_svd | 5 | block_sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| sym_svd | 5 | signed_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| sym_svd | 5 | signed_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |
| sym_svd | 5 | sparse_spike | 0 | 0 | 0 | 0 | 0 | 0 | 0.0008168 |
| sym_svd | 5 | sparse_spike | 0.02 | 0 | 0 | 0 | 0 | nan | 0.0008168 |

### Calibration Decisions

- Best tradeoff scale by mean F1 minus no-corruption residual storage: `3` (F1 `1`, residual storage `0.001002`, false positive ratio `0`).
- High-F1 and low-storage interval exists at scale(s): `1, 1.5, 2, 3`.
- Current default threshold does not over-trigger residual storage in this smoke sweep.
- rpca_svd has no residual anomaly store, so its residual precision/recall remain zero/NaN by design.


## CSV Outputs

- entrywise_bound_calibration: `entrywise_bound_calibration.csv`
- residual_threshold_calibration: `residual_threshold_calibration.csv`
