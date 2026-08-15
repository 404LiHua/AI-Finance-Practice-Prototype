# AA-GFMNet 宸ヤ綔鍖?03锛氱嫭绔嬮噸绠楀紑鍙戝熀绾?
## 璧勪骇缁戝畾

- 鍐荤粨寮犻噺锛歚M5_FROZEN_TENSOR.npz`锛屽舰鐘?`(154,300,36)`锛屾敹鎹爣璁?`COMPLETE`銆?- H4/T2 鏍囩锛?6,200 閿紝鏍囩鍧囧湪 origin 鍚庡疄鐜般€?- 璇ヨ矾绾挎槸鐙珛 recomputation development track锛屼笉鏄師 incumbent锛屼笉鏄寮?37-origin鍗忚鎴愮哗銆?
## 鎵ц绛栫暐

- expanding folds validation starts锛?4/79/94/109/124/139銆?- 姣忔姌 purge=11 鍛紝validation=15 鍛ㄣ€?- Ridge H4銆丆PU XGBoost H4銆乀2 proportional-odds 鍜岄浂鏀剁泭 naive銆?- CPU 闄愬埗涓?绾跨▼锛孏PU鏈娇鐢紝淇濇寔鍗曟満鍝嶅簲銆?
## 缁撴灉

鍏姌 pooled锛?
- Ridge H4 MAE 0.02962锛孯MSE 0.04802锛孯ank IC 0.00609銆?- XGBoost H4 MAE 0.02967锛孯MSE 0.04809锛孯ank IC 0.01045銆?- T2 MCC 0.09858锛涜瀹炵幇鐨?Brier 浠嶉渶鍗曠嫭鏍″噯鏍告煡锛屼笉鑳界敤浜庣敓浜у彲闈犳€с€?
閫愭姌 Rank IC 鏄剧ず鏂瑰悜涓嶇ǔ瀹氾細

`Ridge: +0.0246, -0.0129, -0.0178, +0.0306, -0.0285, +0.0093`锛?`XGBoost: +0.0297, -0.0019, -0.0281, +0.0777, -0.0106, +0.0019`銆?
## 鍐崇瓥

褰撳墠涓嶆弧瓒斥€滃己妯″瀷鈥濊瘉鎹紝涔熶笉鍏佽澶嶆潅妯″潡琛ュ伩銆備笅涓€姝ュ厛鍋氾細

1. 鏍囩瀹炵幇鏃ャ€佺浉瀵规敹鐩婂畾涔夊拰 36 涓壒寰佺殑閫愬瓧娈?PIT/缂哄け瀹¤锛?2. 閫愭姌鏍囧噯鍖栥€佹í鎴潰鍘绘瀬鍊煎拰鎸?origin 鐨?rank-normalization 瀵圭収锛?3. 浠呭湪 TRAIN/VALIDATION 鍐呭仛灏忚寖鍥淬€侀娉ㄥ唽鐨?H4 涓诲共鍊欓€夛紱
4. T2 姒傜巼鏍″噯鍜屽彲闈犳€ц瘎浼扮嫭绔嬪畬鎴愶紱
5. 鑻ョǔ瀹氭€т粛澶辫触锛屼繚鐣欒礋璇佹嵁锛屼笉鍚敤鍥俱€侀鍩熴€侀瞾妫掓垨鏂囨湰妯″潡銆?
缁撴灉鏀舵嵁锛歔BASELINE_RESULTS.json](audits/154_origin_recomputation_baselines_v1/BASELINE_RESULTS.json)


