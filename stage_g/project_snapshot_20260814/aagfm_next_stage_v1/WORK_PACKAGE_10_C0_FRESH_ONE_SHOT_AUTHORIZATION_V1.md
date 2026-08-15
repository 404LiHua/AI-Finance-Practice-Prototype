# 宸ヤ綔鍖?10锛欳0 涓€娆℃€?FRESH 楠岃瘉鎺堟潈锛圴1 鑽夋锛?
**鐘舵€?*锛歚READY_FOR_EXPLICIT_USER_AUTHORIZATION`  
**鍓嶇疆鑺傜偣**锛歚PASS_C0_DEVELOPMENT_GATE_ELIGIBLE_FOR_SEPARATE_FRESH_AUTHORIZATION`  
**鐩爣**锛氫粎楠岃瘉宸插喕缁撶殑 C0锛屼笉鍏佽渚濇嵁 FRESH 缁撴灉鏀瑰彉妯″瀷銆佺壒寰併€佹牎鍑嗐€侀槇鍊笺€佺瀛愭垨鍊欓€夐泦鍚堛€?
## 1. 寰呴獙璇佸璞?
鍞竴鍊欓€夛細`REV8_C0_TARGET_ADAPTED_HETEROSKEDASTIC_ORDINAL`

寮€鍙戦樁娈靛凡缁忛€氳繃鍏ㄩ儴鍐荤粨闂紝浣嗗苟涓嶅畬鍏ㄦ敮閰?incumbent锛氬叾寮€鍙戞眹鎬?MCC 鐣ラ珮锛屾眹鎬?IC 涓?Brier 鐣ュ急銆傚洜姝ゆ湰鑺傜偣涓嶈兘棰勮 C0 鑳滃嚭锛涘繀椤诲悓鏃惰緭鍑?C0銆佹棦鏈夌敓浜?incumbent 鍜?naive 瀵圭収鐨勪笉鍙€嗙粨鏋溿€?
寮€鍙戦棬璇佹嵁锛?
`aagfm_next_stage_v1/audits/wp09_t2_c0_heteroscedastic_ordinal_eval_v1/C0_INDEPENDENT_EVALUATION_DECISION.json`

## 2. 鍞竴鍏佽鎵撳紑鐨勫瘑灏佽緭鍏?
| 鏁版嵁 | SHA-256 | 鏉冮檺 |
|---|---|---|
| `fresh1_confirmation.csv.gz` | `7706f1dbeebc1e065fbf55266443393a87adbf54702c4162c7d4c02841b30226` | 涓€娆℃€у彧璇昏瘎鍒?|
| `fresh2_confirmation.csv.gz` | `f55914c7527df38c95fc294e29d6d9644f91c1e6e6c26ebe6184eaa39b6b61e4` | 涓€娆℃€у彧璇昏瘎鍒?|
| `fresh3_incumbent_confirmation.csv.gz` | `dc4b7f60157b35054a845c2223f6e50a680801a9fbbbf1bd1f371ccb4bedb688` | 涓€娆℃€у彧璇昏瘎鍒?|

杈撳叆鐩綍鍥哄畾涓猴細

`research_tracks/pit_information_incremental_v1/sources/production_t2_reconstruction_20260814_v1/`

鍏佽璇诲彇鍏?`trade_date`銆乣stock_code`銆乣sample_key_sha256`銆乣target_valid`銆乣ordinal_target`銆乣target_return_h4` 鍜屽凡鍐荤粨 14 椤?RG3 鐗瑰緛銆備笉寰楄鍙栧叾浠栨暟鎹泦锛涗笉寰楄拷鍔犮€佸垹闄ゃ€佽鐩栨垨閲嶆柊鐗╁寲 FRESH銆?
## 3. 鍥哄畾鎵ц鏂规硶

1. 鍏堥獙璇佷笁涓?FRESH 鏂囦欢瀛楄妭 SHA銆乻chema銆佸敮涓€閿€佸郊姝ゅ強寮€鍙戞牱鏈殑 key 涓嶉噸鍙狅紱澶辫触鍗冲仠姝€?2. 浠呯敤宸插喕缁撶殑寮€鍙戞牱鏈嫙鍚?C0 鏈€缁堟ā鍨嬶紱鍏跺昂搴︾己澶卞鐞嗐€佸弬鏁般€佸唴灞傛牎鍑嗙綉鏍煎拰閫夋嫨瑙勫垯蹇呴』涓?WP09 protocol 瀹屽叏涓€鑷淬€?3. 杩愯鏃㈡湁鐢熶骇 incumbent 鐨?*鍙棰勬祴**浠ュ強涓ょ naive 瀵圭収锛涗笉鏀瑰啓 incumbent 鐨勫弬鏁版垨娉ㄥ唽琛ㄣ€?4. 瀵逛笁浠?FRESH 棰勬祴鍏ㄩ儴灏佸瓨銆佸啓 SHA 鍚庯紝鍙厑璁哥嫭绔嬭瘎浼板櫒璇诲彇涓€娆℃爣绛惧強鎸囨爣銆?5. 鎸囨爣锛氭瘡涓?FRESH 绐楀彛涓庡悎骞剁獥鍙ｇ殑 MCC銆丅rier銆佹瘡鍛?Spearman IC銆佹牎鍑嗗拰鍥哄畾闂ㄦ帶锛涚粺璁℃柟娉曟部鐢?WP09 鐨?8 鍛ㄥ潡缃崲銆佺Щ鍔ㄥ潡 bootstrap銆佸崟鍊欓€?BH銆?
## 4. 绂佹椤?
- 涓嶅緱浠ヤ换鎰?FRESH 缁撴灉閲嶆柊璁粌銆佽皟鍙傘€佹崲鍊欓€夈€佹崲鏍″噯鎴栨崲闃堝€硷紱
- 涓嶅緱璇诲彇 SCREENING銆丗INAL銆佷换浣曟湭澹版槑 holdout 鎴栧閮ㄦ暟鎹紱
- 涓嶅緱鎶婂紑鍙戦棬閫氳繃绉颁负鐢熶骇鏇挎崲锛?- 涓嶅緱鍦ㄧ嚎閮ㄧ讲銆佺敓浜ф敞鍐屾垨鑷姩涓嬪崟锛?- 涓嶅緱閲嶅紑 R26銆佸姩鎬佸浘銆丯umericV2銆丄AGFM V3/V4/V5銆佹鏂囪瀺鍚堛€佸熀鏈潰娈嬪樊绛夊喕缁撳け璐ヨ矾绾裤€?
## 5. 鍐崇瓥瑙勫垯

鍙湁涓嬪垪鏉′欢鍚屾椂婊¤冻锛屾墠鍙?*寤鸿**杩涘叆鍗曟満 shadow/paper-trading 涓庢敞鍐屽鎵瑰噯澶囷細

- 涓変釜 FRESH 绐楀彛鍧囧畬鏁淬€佸搱甯屽拰 key 鍚堝悓閫氳繃锛?- C0 鍦ㄩ娉ㄥ唽鐨?FRESH 闂ㄤ腑閫氳繃锛?- 鏃犱换浣曠獥鍙ｅ彂鐢熺‖鎬ч闄┿€丳IT銆佹牎鍑嗘垨鍙潬鎬уけ璐ワ紱
- 缁撴灉鐢辩嫭绔嬭瘎浼板櫒鐢熸垚涓旈娴嬪皝瀛?SHA 鏈彉锛?- 褰㈡垚鏄庣‘鐨勨€滈€夋嫨 C0 / 淇濈暀 incumbent鈥濈粨璁恒€傝嫢 C0 鏈兘涓ユ牸鑳滃嚭鎴栨寚鏍囧啿绐侊紝鍒欎繚鐣?incumbent锛屼笉浠ュ钩鍧囨寚鏍囪ˉ鍋垮け璐ョ獥鍙ｃ€?
鍗充娇閫氳繃锛屼篃浠嶉渶绾搁潰浜ゆ槗/鍥炴斁銆佹ā鍨嬪崱銆佸彲澶嶇幇鍗曟満鏈嶅姟鍜屽師瀛愭敞鍐岃〃瀹℃壒锛涙湰宸ヤ綔鍖呮湰韬笉鎵瑰噯鐢熶骇鏇挎崲銆?
## 6. 鍚姩绛惧彂

鍙湁鐢ㄦ埛鏄庣‘纭涓嬪垪鍐呭鍚庢墠鑳芥墽琛岋細

```text
[鈭歖 鎴戞巿鏉冧唬鐞嗕竴娆℃€ф墦寮€骞惰瘎鍒嗕笂杩颁笁浠?FRESH 鏂囦欢
[鈭歖 鎴戠‘璁?FRESH 缁撴灉涓嶄細鐢ㄤ簬閲嶆柊璁粌銆佽皟鍙傛垨鏂板鍊欓€?[鈭歖 鎴戠‘璁ゆ湰娆′笉璇诲彇 SCREENING銆丗INAL 鎴栧叾浠栧瘑灏侀泦
[鈭歖 鎴戠‘璁ゆ湰娆′笉杩涜鐢熶骇鏇挎崲銆佺嚎涓婇儴缃叉垨鑷姩浜ゆ槗
```


