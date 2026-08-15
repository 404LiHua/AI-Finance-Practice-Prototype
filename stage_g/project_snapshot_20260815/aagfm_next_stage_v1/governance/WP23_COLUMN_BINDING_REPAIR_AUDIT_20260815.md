# WP23 market-scale column-binding repair

The failed market-volatility replay was caused by positional CSV column binding, not by a model gain/loss or a cross-source market regime. `pandas.read_csv` returns positional columns in source order; the previous implementation named the requested positions in request order and therefore swapped adjusted close with the adjustment factor.

The corrected materializers bind `[1,2,11,15]` explicitly as `trade_date`, `raw_close`, `adjust_factor`, and `qfq_close`. The archived development panel now matches after the inherited 2018-07-06 warm-up exception across 256 post-warm-up dates, with maximum absolute error `9.71445146547012e-17` under `atol=1e-14`.

Corrected RG2, market-scale, and C1 label-free shadow artifacts were produced for five future origins (2026-07-17 through 2026-08-14), 300 keys each. Future labels were not opened, no metrics were calculated, GPU was not used, and the production registry was not modified. Earlier pre-repair outputs are superseded and must not be used for evaluation.
