# ogbl-collab Preliminary Results

- nodes: 1000
- temporal snapshots: 10
- year_start: 2008
- years: 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017
- held-out observed edges: 10203

| method | train rel. Frob | max entrywise | mean entrywise | train edge RMSE | held-out RMSE | held-out MAE | MRR | Hits@10 | Hits@50 | compressed bytes | sparse ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| spectralstore_asym | 0.8698 | 17.3131 | 0.0054 | 1.2106 | 1.1966 | 0.9299 | 0.7462 | 0.9068 | 1.0000 | 128704 | 0.2446 |
| sym_svd | 0.8416 | 16.0000 | 0.0057 | 1.1437 | 1.1314 | 0.8660 | 0.7721 | 0.9217 | 1.0000 | 128704 | 0.2446 |
| direct_svd | 0.8410 | 16.0000 | 0.0057 | 1.1471 | 1.1411 | 0.8723 | 0.7687 | 0.9208 | 1.0000 | 128704 | 0.2446 |
