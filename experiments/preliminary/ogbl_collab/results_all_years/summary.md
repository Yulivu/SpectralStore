# ogbl-collab Preliminary Results

- nodes: 1000
- temporal snapshots: 37
- year_start: None
- years: 1978, 1981, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017
- held-out observed edges: 9765

| method | train rel. Frob | max entrywise | mean entrywise | train edge RMSE | held-out RMSE | held-out MAE | MRR | Hits@10 | Hits@50 | compressed bytes | sparse ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| spectralstore_asym | 0.8920 | 17.9935 | 0.0014 | 1.2985 | 1.3072 | 1.0411 | 0.7312 | 0.9163 | 1.0000 | 130432 | 0.2128 |
| sym_svd | 0.8461 | 16.7290 | 0.0015 | 1.1978 | 1.2017 | 0.9013 | 0.7644 | 0.9390 | 1.0000 | 130432 | 0.2128 |
| direct_svd | 0.8470 | 17.0485 | 0.0015 | 1.2011 | 1.2122 | 0.9057 | 0.7629 | 0.9363 | 1.0000 | 130432 | 0.2128 |
