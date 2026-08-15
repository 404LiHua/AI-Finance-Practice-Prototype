# 宸ヤ綔鍖?08锛氱敓浜?T2 鍙屽熀绾跨嫭绔嬭缁冩巿鏉冿紙V1 鑽夋锛?
**鐘舵€?*锛歚READY_FOR_EXPLICIT_USER_AUTHORIZATION`  
**鍓嶇疆璇佹嵁**锛歚PASS_REPRODUCIBLE_TRAIN_AND_RG3_SOURCE_BINDING_FRESH_SEALED`  
**鏈伐浣滃寘涓嶆巿鏉?*锛欶RESH 璇诲彇銆丼CREENING/FINAL 璇诲彇銆佺敓浜ф浛鎹€佺嚎涓婇儴缃层€佸浘鍒嗘敮銆佷簨浠跺垎鏀€佽繛缁?H4 澶淬€?
## 1. 鐩爣涓庡彛寰?
鏈妭鐐瑰彧寤虹珛鍙鏌ョ殑鐢熶骇 T2 寮€鍙戝熀绾匡紝涓嶉噸璁?鏇挎崲鐜版湁鐢熶骇妯″瀷銆?
```text
鐩爣锛歍2_MARKET_RELATIVE_FIXED
Horizon锛? 涓懆搴︽闀匡紙鍛ㄤ簲 close origin 鍚庯級
杩炵画鐩爣锛歳aw_target_return_h4 - same-origin valid-stock median(raw_target_return_h4)
闃堝€硷細卤0.01锛岃竟鐣屽綊 NEUTRAL
绫诲埆锛欴OWN=0 / NEUTRAL=1 / UP=2
```

涓ョ灏?TextCU 鐨?4 涓氦鏄撴棩缁濆 H4 鐮旂┒鏍囩褰撲綔璇ュ彛寰勭殑鏇夸唬鍝併€?
## 2. 涓嶅彲鍙樿緭鍏ヤ笌鍏佽鑼冨洿

| 杈撳叆 | SHA-256 | 鍏佽鐢ㄩ€?|
|---|---|---|
| rev8 TRAIN 鐩爣 | `aadada6cbdcaaefd0edd0df1a66b176daba6b015cee6d9f6867cf95d6d92204c` | 浠呭紑鍙戣缁?鍒囧垎 |
| RG3 鐗瑰緛 | `04f6b11b7296aa1d92bdc0a97d652565672ac90b1a01601b67f9a629989a9525` | 浠?14 涓棩鎶€鏈壒寰?|
| 璁粌鏍锋湰婧?| `60c1322c228a7a2e52b9c0b5dec054ba5a95de242170cca17759a7fb631091b6` | 浠呭凡鍐荤粨 TRAIN/VALIDATION 寮€鍙戝垏鍒?|
| 鏍锋湰鍒囧垎鍚堝悓 | `8b52e673a20422ad0f2ae688fced69c15830b72c9e44f17d5d9d516ac650bb53` | 6 鎶?expanding銆? 鍛?purge + 7 鍛?embargo |

杈撳叆蹇呴』鏉ヨ嚜 `sources/production_t2_source_binding_20260814_v1/` 鎴栧叾宸查獙璇侀噸鐗╁寲浜х墿銆傛柊鑴氭湰涓嶅緱璇诲彇浠?`fresh`銆乣screening`銆乣final`銆乣sealed` 鍛藉悕鎴栫櫥璁扮殑 payload銆?
## 3. 鍥哄畾妯″瀷棰勭畻

鏈巿鏉冨彧鍏佽浠ヤ笅涓や釜鍙傝€冨熀绾匡紝鏃犱换浣曠壒寰佹悳绱€佺獥鍙ｆ悳绱€乻eed 鎼滅储鎴栬瀺鍚堬細

1. `NAIVE_PRIOR` 涓?`NAIVE_NEUTRAL`锛氫綔涓?BEST_NAIVE锛屾寜棰勫厛瑙勫畾鐨勭嫭绔嬫寚鏍囧紩鎿庨€夎緝浼樿€咃紱
2. `INCUMBENT_ORDINAL_T2_DEV`锛?4 椤?RG3 鏃ユ妧鏈壒寰併€佸悓鏂瑰樊 proportional-odds銆乣location_l2=0.001`銆?
涓嶅厑璁歌缁冧换浣曞浘銆侀鐜囥€佹枃鏈€佷簨浠躲€佸熀鏈潰銆佹贩鍚堥棬鎺ф垨杩炵画鏀剁泭鍒嗘敮銆傚凡鍐荤粨澶辫触璺嚎涓嶅緱鍊熷熀绾胯妭鐐归噸寮€銆?
## 4. 杩愯涓庡璁¤姹?
- 鍏堣繍琛?CPU-only 鏁版嵁/閿?鍒囧垎/PIT dry-run锛屽啀杩愯璁粌锛?- GPU 濡傜‘鏈夊繀瑕侊紝浠呯敤浜庨娉ㄥ唽鐨勫€欓€夎缁冿紱鍙屽熀绾块粯璁?CPU锛?- 姣忔杩愯杈撳嚭鍒版柊鐨勪笉鍙鐩栫洰褰曪紝璁板綍浠ｇ爜銆佽緭鍏ャ€佺幆澧冦€佸懡浠ゅ拰 SHA-256锛?- 棰勬祴鍏堝皝瀛橈紝鎸囨爣鐢辩嫭绔嬭鍙栧櫒璇诲彇锛涗笉寰椾粠鎸囨爣鍙嶅悜璋冨弬锛?- 杈撳嚭蹇呴』娉ㄦ槑 `new_evaluation_claim=false`锛岄櫎闈炲悗缁彟鏈夌嫭绔嬫寚鏍囪鍙栨巿鏉冿紱
- 浠讳綍缂哄け閿€佸垏鍒嗛噸鍙犮€乫uture join銆佽緭鍏?SHA 涓嶅尮閰嶆垨绂佽璺緞璁块棶绔嬪嵆 fail-closed銆?
## 5. 鏈妭鐐圭殑浜х墿

```text
preconsumption/INPUT_AND_PATH_DENYLIST_AUDIT.json
preconsumption/SPLIT_AND_KEY_AUDIT.json
predictions_sealed/<baseline>/<fold>.parquet
run_receipts/<baseline>.json
SOURCE_AND_ENV_MANIFEST_SHA256.csv
```

涓嶅緱浜х敓鐢熶骇娉ㄥ唽琛ㄥ彉鏇淬€佺敓浜фā鍨嬭鐩栨垨閮ㄧ讲閰嶇疆銆?
## 6. 鍚庣画闂ㄦ

鍙屽熀绾垮缓绔嬪畬鎴愬彧浼氫骇鐢熲€滃彲姣旇緝鐨勫紑鍙戝弬鐓р€濓紝涓嶈嚜鍔ㄥ厑璁稿€欓€夎缁冦€傚€欓€夋ā鍨嬪繀椤诲彟琛岄娉ㄥ唽骞舵槑纭細鍞竴鍊欓€夈€佽缁冮绠椼€侀娴嬪皝瀛樸€佺嫭绔嬫寚鏍囪鍙栥€侀€氳繃闂ㄥ拰澶辫触鍚庣殑鍐荤粨瑙勫垯銆侳RESH 璇勫垎蹇呴』鍦ㄨ鍊欓€夐€氳繃寮€鍙戦棬鍚庡崟鐙敵璇蜂竴娆℃€ф巿鏉冦€?
## 7. 鍚姩绛惧彂

浠ヤ笅椤圭洰闇€瑕佺敤鎴锋槑纭‘璁ゅ悗锛屾墠鍙紑濮嬪疄闄呮暟鎹閰?鍩虹嚎璁粌锛?
```text
[鈭歖 纭浠呬娇鐢ㄧ 2 鑺傚喕缁撹緭鍏ヤ笌 TRAIN/VALIDATION 寮€鍙戝垏鍒?[鈭歖 纭涓嶈鍙栦换浣?FRESH / SCREENING / FINAL / sealed payload
[鈭歖 纭鏈浠呰缁冧袱涓浐瀹氬弻鍩虹嚎锛屼笖涓嶆敼鍙樼敓浜у唴鏍?[鈭歖 纭鐙珛鎸囨爣璇诲彇鍓嶄笉杩涜浠讳綍璋冨弬鎴栧€欓€夎缁?```


