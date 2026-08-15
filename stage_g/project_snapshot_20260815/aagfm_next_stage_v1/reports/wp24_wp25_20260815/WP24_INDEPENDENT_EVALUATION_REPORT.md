# WP24 H4/T2 多任务共享主干开发评价

状态：`FAIL_DEVELOPMENT_JOINT_GATE_RESEARCH_ONLY`

本评价读取的只有冻结开发折的验证标签；未读取未来、FRESH、SCREENING 或 FINAL 标签，未修改生产注册表。

| 折 | MAE | 零预测 MAE | 周 IC | MCC | incumbent MCC | Brier | incumbent Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| REV2_RO_01 | 0.088193 | 0.085163 | 0.110566 | 0.025716 | 0.057743 | 0.604306 | 0.588357 |
| REV2_RO_02 | 0.097774 | 0.093042 | 0.109331 | 0.044368 | 0.060989 | 0.601935 | 0.590017 |
| REV2_RO_03 | 0.093491 | 0.084417 | 0.045498 | 0.075488 | 0.073427 | 0.602486 | 0.590405 |
| REV2_RO_04 | 0.090946 | 0.091344 | 0.163811 | 0.126521 | 0.081459 | 0.586930 | 0.576846 |
| REV2_RO_05 | 0.072352 | 0.071539 | 0.173127 | 0.134188 | 0.102295 | 0.592630 | 0.590003 |
| REV2_RO_06 | 0.068575 | 0.063042 | 0.007443 | 0.075658 | 0.081041 | 0.607815 | 0.602253 |

## 门槛

- regression_positive_ic_folds: `6`，通过=`True`
- regression_median_mae_delta_vs_zero: `0.0038812818798793225`，通过=`False`
- classification_positive_mcc_vs_naive_prior: `6`，通过=`True`
- classification_worst_mcc_delta_vs_incumbent: `-0.03202650594826249`，通过=`False`
- classification_worst_brier_delta_vs_incumbent: `0.01594928049446387`，通过=`False`

生产结论：即使开发门通过，也只产生研究候选；未来独立 T2 评价和生产替换仍需独立授权。
