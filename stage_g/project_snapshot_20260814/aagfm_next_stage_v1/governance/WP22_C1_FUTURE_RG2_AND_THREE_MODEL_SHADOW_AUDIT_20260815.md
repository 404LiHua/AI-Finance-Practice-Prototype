# WP22 C1 future RG2 and three-model shadow audit

Status: `NON_PRODUCTION_LABEL_FREE_SHADOW_ONLY`

## CSMAR capital-event lineage

The frozen historical `corporate_actions.csv.gz` was compared row-for-row with the current raw CSMAR capital-change archive for the frozen 300-stock universe.

- Historical rows: 9,528
- Exact raw-archive matches: 9,528
- Historical-only rows: 0
- Archive-only rows: 1,265, all dated after the historical panel end date (`2023-06-01`)
- Source-row provenance matches: 9,528 / 9,528

The audit result is `PASS_HISTORICAL_EXACT_RAW_EXTENDED`. This establishes that the raw archive is a strict forward extension of the historical capital-event source used by the development state materialization; it does not by itself authorize production replacement.

## Future RG2 materialization

For origin `2026-07-17`, the RG2 state panel has 300 keys and 18 finite state/graph features. Its mandatory `2023-05-05` development-formula replay matched every feature with maximum absolute error `0.0`. The panel SHA-256 is `cbeea9ea9213a617d9d02437a0043636b691f732508ec6f3cc186a753d1e7bbf`.

## Aligned label-free predictions

The incumbent, C0 and C1 predictions were independently validated as 300-key, one-origin (`2026-07-17`) sets with normalized three-class probabilities, then sealed together.

| Slot | Identity |
| --- | --- |
| Incumbent | `RG_OBGNET_CONFIRMED_SAFE_V1_1` |
| C0 | `REV8_C0_TARGET_ADAPTED_HETEROSKEDASTIC_ORDINAL` |
| C1 | `REV8_C1_RG2_STATE_AUGMENTED_HETEROSKEDASTIC_ORDINAL` |

No target labels, FRESH payloads, SCREENING, FINAL, or production registry were read or changed. The seal is pending a separately authorized, independent T2-label evaluation. Until that gate passes, C1 remains a shadow candidate and the active production model remains unchanged.
