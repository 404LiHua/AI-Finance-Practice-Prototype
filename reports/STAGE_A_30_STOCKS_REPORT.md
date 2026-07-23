# Stage A: 30-stock data pipeline acceptance report

Report date: 2026-07-23

## Scope

- Stocks: 30
- Date range: 2022-06-03 to 2023-06-02
- Raw weekly prices: JiuZhang Quant, unadjusted
- Model prices: BaoStock, forward-adjusted
- Structured events: CSMAR special treatment and capital changes
- Split: chronological train/validation/test with purge weeks

## Acceptance results

- Raw selected rows: 1498
- Duplicate stock/date rows: 0
- Bad dates: 0
- Missing numeric cells: 0
- Invalid OHLC rows: 0
- Previous-close continuity differences (informational): 2
- BaoStock coverage: 100.0%
- CSMAR weekly-state coverage: 100.0%
- Text/event records aligned to panel: 18
- Training text rows used to fit TF-IDF: 15
- TF-IDF vocabulary size: 512
- SVD dimensions: 8
- Text clusters: 4
- Train samples: 688
- Validation samples: 120
- Test samples: 210

## Baseline verification

- Validation MAE: 0.039396
- Validation RMSE: 0.055820
- Validation direction accuracy: 40.00%
- Test MAE: 0.035948
- Test RMSE: 0.052976
- Test direction accuracy: 52.86%

## Traceability

- JiuZhang source files indexed: 5000
- BaoStock downloaded files: 60
- CSMAR source ZIP files: 4
- Every source file is recorded with SHA-256; CSMAR event rows retain source ZIP and row number.

## Known limitations

- The exported CSMAR category provides special-treatment/capital events, not broad daily financial news.
- Event text is sparse: most normal stocks have no special-treatment event during one year.
- CSMAR exports are marked for Jiaxing University use only and are excluded from Git.
- The random-forest model is a data-path verification baseline, not the proposed graph-frequency model.
- Stage B should reproduce stronger time-series baselines before expanding beyond 30 stocks.

## Gate decision

**PASS.** Fatal quality counts are zero, source coverage is 100%, text features are fitted from training data only, and all splits are non-empty.
