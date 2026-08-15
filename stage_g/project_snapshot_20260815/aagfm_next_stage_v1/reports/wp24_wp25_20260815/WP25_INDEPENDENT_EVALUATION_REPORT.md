# WP25 锚定残差 H4/T2 开发评价

状态：`PASS_DEVELOPMENT_JOINT_GATE`

本评价读取的只有冻结开发折的验证标签；未读取未来、FRESH、SCREENING 或 FINAL 标签，未修改生产注册表。

| 折 | MAE | 零预测 MAE | 周 IC | MCC | incumbent MCC | Brier | incumbent Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| REV2_RO_01 | 0.086541 | 0.085163 | 0.067902 | 0.050042 | 0.057743 | 0.591905 | 0.588357 |
| REV2_RO_02 | 0.092516 | 0.093042 | 0.120792 | 0.070294 | 0.060989 | 0.587226 | 0.590017 |
| REV2_RO_03 | 0.084746 | 0.084417 | 0.137451 | 0.078672 | 0.073427 | 0.589221 | 0.590405 |
| REV2_RO_04 | 0.089436 | 0.091344 | 0.229197 | 0.113399 | 0.081459 | 0.571400 | 0.576846 |
| REV2_RO_05 | 0.070109 | 0.071539 | 0.223422 | 0.126741 | 0.102295 | 0.585357 | 0.590003 |
| REV2_RO_06 | 0.063575 | 0.063042 | 0.137853 | 0.099259 | 0.081041 | 0.601312 | 0.602253 |

## 门槛

- regression_positive_ic_folds: `6`，通过=`True`
- regression_median_mae_delta_vs_zero: `-9.817704505576258e-05`，通过=`True`
- classification_positive_mcc_vs_naive_prior: `6`，通过=`True`
- classification_worst_mcc_delta_vs_incumbent: `-0.007700806894398726`，通过=`True`
- classification_worst_brier_delta_vs_incumbent: `0.0035481740476189794`，通过=`True`

生产结论：即使开发门通过，也只产生研究候选；未来独立 T2 评价和生产替换仍需独立授权。
