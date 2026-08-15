# V4 future FRESH preflight (2026-08-15)

This compact, public package prepares the pre-registered V4 independent validation of `AA_GFMNET_CROSS_SECTIONAL_NEUTRALIZED_RESIDUAL_TCN_V1` against the T2 production anchor.

It contains no market data, real FRESH labels, predictions, model weights, or scores.  It has passed CPU-only synthetic tests, but is intentionally **not materializable** until 2026-09-11 or later, a frozen delivery attestation, per-file hashes, a PIT fundamental export, and a consistent 299/300-stock universe are available.

## Contents

- `WORK_PACKAGE_13_V4_FUTURE_ANCHOR_ELIGIBLE_FRESH_PRECHECK_V1.md` — scope and operating boundary;
- `scripts/` — V4-only builder, label-free audit, and synthetic regression tests;
- `WORK_PACKAGE_14_V4_PRECONSUMPTION_PREDICTION_BINDING_V1.md` and companion scripts — exact eight-origin prediction binding before any custodian label access;
- `WORK_PACKAGE_15_V4_LABEL_FREE_E2E_RUNTIME_SMOKE_TEST_V1.md` and `test_csn_v4_label_free_e2e_runtime_v1.py` — synthetic end-to-end candidate-GPU/anchor-CPU runtime and real-output binding verification;
- `WORK_PACKAGE_16_PRODUCTION_T2_ALIGNMENT_AND_V4_PROMOTION_REVOCATION_V1.md` and `V4_PRODUCTION_T2_TARGET_SEMANTICS_CONFLICT_FREEZE_V1.json` — production-semantic gate; current V4 is not a promotion experiment;
- `governance/data_requests/20260815_PRODUCTION_T2_ALIGNMENT_DATA_REQUEST_V1.md` — exact archive/reconstruction delivery requirements for regaining production-comparable status;
- `promote_csn_candidate_atomically_v1.py` — atomic promotion now requires an independent production-T2 semantic-alignment PASS receipt;
- `governance/data_requests/` — delivery-attestation template;
- `audits/` — synthetic test receipts and the fail-closed preflight decision.

The builder fixes eight Monday 09:30 origins (2026-07-20 through 2026-09-07), uses the final trading day strictly before each origin as anchor, and uses the fourth trading day after origin (normally Friday) as H4 target.  It does not authorize scoring, promotion, automatic trading, or a production-model replacement.
