# 鍗曟満 T2 杩愯淇濋殰闂幆锛?026-08-14锛?
## 缁撹

褰撳墠 active 鍐呮牳 `RG_OBGNET_CONFIRMED_SAFE_V1_1` 宸查€氳繃鏈満銆佸洖鐜湴鍧€鐨勭鍒扮杩愯楠屾敹锛?
```text
PASS_LOCAL_LOOPBACK_OPERATIONAL_ACCEPTANCE
```

杩欒瘉鏄庝釜浜虹數鑴戜笂鐨勫彧璇绘湰鍦版湇鍔″彲杩愯锛涗笉浠ｈ〃鍊欓€?C0 宸茶儨鍑猴紝涓嶄唬琛ㄥ厑璁稿缃戦儴缃层€佽嚜鍔ㄤ氦鏄撴垨鐢熶骇妯″瀷鍒囨崲銆?
## 绔埌绔疄娴?
鏈嶅姟浠呬复鏃剁粦瀹?`127.0.0.1` 鐨勯殢鏈虹鍙ｏ紝瀹屾垚鍚庝富鍔?shutdown銆?
| 椤圭洰 | 缁撴灉 |
|---|---:|
| `/api/health` | HTTP 200锛?.0464 绉?|
| `/api/model` | HTTP 200锛?.0013 绉?|
| `000001.SZ` 棣栨棰勬祴 | HTTP 200锛?.0461 绉掞紝缂撳瓨 `MISS` |
| 鍚屼竴棰勬祴閲嶆斁 | HTTP 200锛?.0358 绉掞紝缂撳瓨 `HIT`锛屼笁绫绘鐜囬€愪綅涓€鑷?|
| 闈炴硶鑲＄エ浠ｇ爜 | HTTP 400锛宖ail-closed |
| 鏈潵鏃ユ湡 `2099-01-01` | HTTP 400锛宖ail-closed |
| 妯″瀷鍥?浜嬩欢鍒嗘敮 | `0.0 / 0.0` |
| 鏈嶅姟绔?GPU policy | `disabled` |
| 鏍囩銆丗RESH銆乄P10 杈撳嚭銆佹寚鏍囪鍙?| 0 |
| 璁粌銆丟PU銆佽嚜鍔ㄤ氦鏄撱€佺敓浜ц祫浜т慨鏀?| 0 |

瀹為檯鏈嶅姟鍚姩鍏ュ彛涔熷凡鍦?`127.0.0.1:8766` 楠岃瘉鍋ュ悍妫€鏌ュ悗鍏抽棴锛涗笉浼氶仐鐣欑洃鍚繘绋嬨€?
## 鏈夌晫鍘嬪姏楠岃瘉

鍦ㄥ悓涓€鍥炵幆銆丆PU-only 杈圭晫鍐呭彟瀹屾垚 121 娆＄‘瀹氭€ч娴嬶紙100 娆￠『搴?+ 20 娆°€佷袱涓鎴风鐨勫苟鍙戣姹?+ 1 娆?warm-up锛夛細

| 鎸囨爣 | 缁撴灉 |
|---|---:|
| 涓綅寤惰繜 | 0.0610 绉?|
| P95 / P99 寤惰繜 | 0.1394 / 0.1613 绉?|
| 缂撳瓨鐘舵€?| 1 娆?`MISS`銆?20 娆?`HIT` |
| 杈撳嚭涓€鑷存€?| 121 娆″潎涓?warm-up 姒傜巼閫愪綅涓€鑷?|
| tracemalloc 褰撳墠澧為暱 | 181,299 bytes锛堥棬妲?8 MiB锛?|
| GPU / 鏍囩 / 璁粌 / 鑷姩浜ゆ槗 | 0 / 0 / 0 / 0 |

鍘嬪姏楠屾敹锛歔LOCAL_LOOPBACK_SOAK_DECISION.json](../local_service/operational_loopback_soak_20260814_v1/LOCAL_LOOPBACK_SOAK_DECISION.json)锛孲HA-256 `069aec4d5e7acb3cf2484dc9af4cafafe51ed4d5ace18868709e481f450280c6`銆?
## 鍐荤粨韬唤

- 杩愯楠屾敹绛栫暐锛歔LOCAL_T2_SINGLE_MACHINE_OPERATIONAL_ACCEPTANCE_V1.json](../local_service/LOCAL_T2_SINGLE_MACHINE_OPERATIONAL_ACCEPTANCE_V1.json)锛孲HA-256 `eed2d55119500b1dfecc3d1daa09d6f8f1328d9be3850c731e93a1063ee73a75`銆?- 杩愯楠屾敹鍐崇瓥锛歔LOCAL_LOOPBACK_OPERATIONAL_DECISION.json](../local_service/operational_loopback_audit_20260814_v1/LOCAL_LOOPBACK_OPERATIONAL_DECISION.json)锛孲HA-256 `8f807ba7665bd27836d23376cf47fad03be630be833c930de69cbe3dc72f9560`銆?- 褰撳墠鐜閿侊細[LOCAL_T2_RUNTIME_ENVIRONMENT_LOCK.json](../local_service/runtime_environment_lock_20260814_v1/LOCAL_T2_RUNTIME_ENVIRONMENT_LOCK.json)锛孲HA-256 `be2943bfe99e07b85736d3717d2f21f828037338d11c7828919136193c5175e8`銆?- 褰撳墠 registry SHA-256锛歚12d4ec2e7bcb933e316e5d6b8e20d685f94ab2f585f1516a335a062047e6ea13`銆?- 褰撳墠 model SHA-256锛歚d8e4316d0fab70d3785b775c695a1f3a31225edf441a3603e3830f7351c4e2e8`銆?
鐜閿佹槑纭垎绂荤‖浠跺伐浣滐細anchor HTTP/batch 浣跨敤 CPU锛汣UDA 12.8銆丷TX 5060 Laptop GPU銆乀orch/XGBoost/CuPy 鍙繚鐣欑粰鍙﹁棰勬敞鍐岀殑鐮旂┒璁粌浠诲姟銆傛湇鍔″惎鍔ㄥ叆鍙?`local_service/serve_t2_local_loopback_v1.py` 寮哄埗 GPU disabled 鍜?2 鏉?CPU 绾跨▼銆?
## 浣跨敤杈圭晫

鍚姩鏂瑰紡瑙侊細[local_service/README.md](../local_service/README.md)銆傛湇鍔′粎鍙粦瀹氭湰鏈?`127.0.0.1`锛涘缃戠洃鍚€佽韩浠介壌鏉冦€佸鍣ㄥ寲銆佺嚎涓婄洃鎺с€佽嚜鍔ㄤ氦鏄撳潎灏氭湭鎺堟潈鍜屽疄鏂姐€?
妯″瀷鏁堟灉鐨勪笅涓€纭棬浠嶆槸 WP11锛氳ˉ榻?300 鑲＄エ褰撴棩鏁版嵁銆佸皝瀛樿嚦灏?12 涓共鍑€ origin锛屽苟鐢辩嫭绔嬩繚绠¤€呬互涓€娆℃€ф巿鏉冨畬鎴愬悓鐩爣鏍囩璇勫垎銆傚湪姝や箣鍓嶏紝鍗曟満鏈嶅姟鍙互浣跨敤 active anchor 鍋氱爺绌舵紨绀猴紝浣嗕笉鑳界敤鏉ュ绉?C0 宸蹭紭浜庣敓浜фā鍨嬨€?

