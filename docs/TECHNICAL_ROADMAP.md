# SpectralStore 鎶€鏈矾绾垮樊璺濅笌宸ョ▼璺嚎鍥?
鏈枃妗ｅ鐓?`SpectralStore.md`锛岃褰曞綋鍓嶄粨搴撳凡缁忓畬鎴愮殑鑳藉姏銆佽窛绂诲畬鏁存妧鏈矾绾跨殑缂哄彛銆佷紭鍏堢骇锛屼互鍙婁笅涓€闃舵楠屾敹鏍囧噯銆傚畠鐨勫畾浣嶆槸宸ョ▼璺嚎鍥撅細姣忎竴椤归兘搴旇兘琚媶鎴愬彲瀹炵幇銆佸彲娴嬭瘯銆佸彲澶嶇幇瀹為獙鐨勪换鍔°€?
## 褰撳墠宸插畬鎴?
- 浠撳簱缁撴瀯宸茬粡鎸夊彲璇绘€ф媶鍒嗭細`data/`銆乣scripts/`銆乣experiments/`銆乣tests/smoke/`銆?- 宸叉湁鍩虹鍥犲瓙鍖栧瓨鍌細`FactorizedTemporalStore` 瀛樺偍 `left/right/temporal/lambdas/residuals`銆?- 宸叉湁绗竴鐗堟煡璇㈠眰锛歚LINK_PROB`銆乣TOP_NEIGHBOR`銆乣TEMPORAL_TREND`銆乣ANOMALY_DETECT` 鐨勫熀纭€鎺ュ彛銆?- 宸叉湁绗竴鐗堝帇缂╁櫒锛歚AsymmetricSpectralCompressor`銆乣SymmetricSVDCompressor`銆乣DirectSVDCompressor`銆?- 宸叉湁 Bitcoin-OTC 涓嬭浇銆佸姞杞藉拰 preliminary 瀹為獙銆?- 宸叉湁 Synthetic-SBM 鐢熸垚銆佺湡鍊肩煩闃佃瘎浼板拰 preliminary 瀹為獙銆?
## 瀹屾暣宸窛娓呭崟

| 妯″潡 | 褰撳墠鐘舵€?| 缂哄彛 |
| --- | --- | --- |
| 瀛樺偍灞?| 鏈夊唴瀛樻€佸洜瀛愬寲 store 鍜?CSR residual 瀛楁 | 缂?serialization銆乧ompression ratio 缁熻銆乺esidual 鍏冩暟鎹€乽ncertainty 鍙傛暟 |
| 鍘嬬缉寮曟搸 | 鏈?dense mean/SVD 鍘熷瀷鍜屽 split 闈炲绉?ensemble | 缂虹湡姝ｅ紶閲忓睍寮€銆丄RD/VI銆侀瞾妫掍氦鏇夸紭鍖栥€佸閲忔洿鏂般€佺█鐤忓ぇ鍥?SVD |
| 鏌ヨ灞?| Q1/Q2/Q4/Q5 鏈夊熀纭€杩斿洖 | Q3 `COMMUNITY` 缂哄け锛決1/Q2/Q4/Q5 缂鸿宸繑鍥炪€乹uery optimizer銆乥atch API |
| 绱㈠紩灞?| 鍙湁绌哄寘 | PQ/MIPS銆佹椂闂寸储寮曘€佺ぞ鍖哄€掓帓绱㈠紩鍧囨湭瀹炵幇 |
| 璇樊淇濊瘉 | 鏈夌粡楠?entrywise/Frobenius 鎸囨爣 | 缂?entrywise bound銆佽妭鐐瑰害鍙傛暟銆佸櫔澹颁及璁°€乹uery error bound |
| 鏁版嵁灞?| 鏈?Bitcoin-OTC 鍜?Synthetic-SBM | 缂?Bitcoin-Alpha銆乁CI銆丒nron銆丱GB銆丷eddit銆丼tack Overflow 绛夊姞杞藉櫒 |
| Baseline | 鏈?SymSVD/DirectSVD | 缂?NMF銆丆P/Tucker銆丅PTF銆丷PCA+SVD銆佸浘鎽樿鍜屽姩鎬佸浘鏂规硶 |
| 瀹為獙 | 鏈変袱涓?preliminary 瀹為獙 | `SpectralStore.md` 鐨?7 缁勬寮忓疄楠屽ぇ澶氭湭鍋?|

## 浼樺厛绾ц矾绾?
- **P0: Synthetic-Attack + Robust residual**  
  鐩爣鏄仛鍑?`SpectralStore-Full` 涓?`NoRobust` 鐨勭涓€缁勬秷铻嶏紝璁╂柟娉曡繘鍏ュ畠鏈€鎿呴暱鐨勫鎶楁壈鍔ㄥ満鏅€?- **P1: entrywise bound 鍒濈増**  
  鍔犲叆缁忛獙鍣０浼拌銆佽妭鐐瑰害鍙傛暟鍜屽彲杩斿洖鐨?query error bound銆傚綋鍓嶅凡瀹屾垚绗竴姝ワ細MAD 鑷€傚簲 residual threshold锛岀敤浜庨伩鍏嶆棤鏀诲嚮鏃跺浐瀹氭瘮渚嬪墺绂?residual銆?- **P2: query optimizer + residual correction**  
  璁╂煡璇㈡牴鎹宸蹇嶅害鍐冲畾鏄惁璇诲彇 residual銆?- **P3: PQ index + query latency experiment**  
  灏?`TOP_NEIGHBOR` 浠?dense scan 鎺ㄥ悜杩戜技 MIPS銆?- **P4: ARD rank selection**  
  鐢ㄨ嚜鍔ㄧЗ閫夋嫨鏇夸唬鎵嬪姩 rank锛屽苟鍜屼氦鍙夐獙璇佸姣斻€?- **P5: 姝ｅ紡澶ц妯″疄楠屼笌鏇村 baseline**  
  琛ラ綈 `SpectralStore.md` 鐨勬暟鎹泦銆乥aseline 鍜岃鏂囩骇瀹為獙銆?
## 涓嬩竴闃舵瀹炵幇璁″垝

涓嬩竴闃舵鑱氱劍 **椴佹鏀诲嚮瀹為獙 + Robust residual compressor**銆?
闇€瑕佹柊澧?Synthetic-Attack 鏁版嵁鐢熸垚鑳藉姏锛?
- 鏀寔 `random_flip`銆乣targeted_cross_community`銆乣sparse_outlier_edges`銆?- `snapshots` 琛ㄧず鍙楁敾鍑昏娴嬪浘锛宍expected_snapshots` 淇濇寔骞插噣鐪熷€笺€?- 璁板綍 attack metadata锛岀敤浜?anomaly precision/recall銆?
闇€瑕佹柊澧?`RobustAsymmetricSpectralCompressor`锛?
- 鍏堝仛鍒濆浣庣З浼拌銆?- 璁＄畻娈嬪樊鐭╅樀銆?- 鎸夊浐瀹氶槇鍊兼垨鍒嗕綅鏁板垎绂?sparse residual銆?- 鐢ㄥ幓 residual 鍚庣殑鐭╅樀閲嶆柊鎷熷悎銆?- 杈撳嚭 CSR residual锛屼緵 `ANOMALY_DETECT` 鍜?residual correction 浣跨敤銆?
闇€瑕佹柊澧炲疄楠岋細

- 鐩綍锛歚experiments/preliminary/synthetic_attack/`
- 鑴氭湰锛歚scripts/preliminary/run_preliminary_synthetic_attack.py`
- 瀵规瘮锛歚spectralstore_full`銆乣spectralstore_no_robust`銆乣baseline_sym_svd`銆乣baseline_direct_svd`
- 鎸囨爣锛歮ax entrywise error銆乵ean entrywise error銆乺elative Frobenius error銆乤nomaly precision/recall銆乺esidual sparsity/residual nnz

## 楠屾敹鏍囧噯

- `pytest -p no:cacheprovider` 鍏ㄩ儴閫氳繃銆?- `python scripts/preliminary/run_preliminary_synthetic_attack.py` 鍙洿鎺ヨ繍琛屻€?- `experiments/preliminary/synthetic_attack/results/summary.md` 鍖呭惈 Full vs NoRobust vs baseline 琛ㄦ牸銆?- Synthetic-Attack 鐢熸垚鍣ㄨ繑鍥炴纭昂瀵革紝涓?attack severity 澧炲ぇ鏃舵壈鍔ㄨ竟鏁伴噺澧炲姞銆?- Robust compressor 鑳戒骇鐢?residuals锛屼笖 residual 鏁伴噺绛変簬鏃堕棿蹇収鏁般€?# 2026-04-24 System-First Update

Completed in the current system skeleton pass:

- `COMMUNITY(t)` now has a preliminary Synthetic-SBM evaluation loop with NMI and ARI.
- Storage accounting now reports dense and CSR-sparse raw denominators plus ratios against both.
- `TOP_NEIGHBOR` has a prototype exact factor-space MIPS index path.
- Indexed `TOP_NEIGHBOR` now supports CSR residual candidate reranking for corrected scores.
- A reproducible query latency microbenchmark now covers Q1/Q2/Q4/Q5 on raw dense, factorized, residual-corrected, and indexed Q2 paths.
- `FactorizedTemporalStore` now has NPZ round-trip serialization for factors, CSR residuals, diagnostics, and bound metadata.
- Tensor compressors now normalize component scales into `lambdas` and support energy-based effective-rank pruning via `tensor_rank_energy`.
- The SVD compressor path now supports iterative ARD-like deterministic rank pruning, temporal refit, and effective-rank diagnostics.
- Experiment scripts now default to YAML configs through Hydra/OmegaConf and emit `run_metadata.json` for reproducibility.

Near-term follow-up:

- Add versioned serialization manifests before broadening benchmark scale.
- Move CP/Tucker from dense reconstruction prototypes toward sparse/observed-edge objectives.
- Upgrade deterministic ARD-like pruning to a posterior/VI formulation.
- Add Hydra config groups, multirun sweep launchers, and optional MLflow/W&B tracking.
- Keep CP/Tucker and current latency scripts marked as preliminary/prototype until the system path is stable.

