# WP10 FRESH C0 灏哄害鐗瑰緛闃绘柇涓庤ˉ浠惰姹傦紙2026-08-14锛?
**鐘舵€侊細`BLOCKED_NO_FRESH_PREDICTION_SEALED_NO_LABEL_READ`**

## 宸叉墽琛屼笖宸查獙璇佺殑杈圭晫

鐢ㄦ埛宸叉巿鏉冧竴娆℃€?WP10 FRESH 楠岃瘉銆傛墽琛屽櫒鍦?FRESH 璇诲彇闃舵濮嬬粓鍙鍙?`trade_date`銆乣stock_code`銆乣sample_key_sha256` 涓?14 椤?RG3 鎶€鏈壒寰侊紱鏈鍙?`target_valid`銆乣ordinal_target`銆乣target_return_h4`銆乣raw_target_return_h4` 鎴栦换浣曟寚鏍囥€傛湭璇诲彇 SCREENING銆丗INAL 鎴栧叾浠?sealed holdout锛屾湭淇敼鐢熶骇娉ㄥ唽琛ㄣ€佺敓浜фā鍨嬫垨閮ㄧ讲銆?
涓や釜涓嶅彲瑕嗙洊鐩綍鍧囧凡淇濈暀锛屼笖鍏?`predictions_sealed/` 涓嬫病鏈夋枃浠躲€佹病鏈?seal manifest銆佹病鏈夎瘎浼拌緭鍑猴細

- `runs/wp10_c0_fresh_one_shot_v1/`锛歴chema preflight 鍋滄锛汧RESH 涓夋枃浠跺潎涓嶅惈 `market_volatility_4w` 鍒椼€?- `runs/wp10_c0_fresh_one_shot_v2/`锛氱敤鍙楁帶寮€鍙?`weekly_panel.csv.gz` 澶嶇畻寮€鍙戞湡鍚屽悕鐗瑰緛鍚庯紝FRESH date coverage preflight 鍋滄銆?
涓ゆ鍋滄鍧囧彂鐢熷湪鍐欏嚭绗竴浠介娴嬩箣鍓嶏紱娌℃湁 FRESH 璇勫垎銆佹爣绛捐鍙栥€佹ā鍨嬮€夋嫨鎴栫敓浜ф浛鎹€?
## 宸叉牳楠屼簨瀹?
| 椤圭洰 | 缁撴灉 |
|---|---|
| 鍐荤粨 C0 灏哄害杈撳叆 | `realized_volatility_20d`銆乣downside_volatility_60d`銆乣market_volatility_4w` |
| FRESH 涓夌獥鍙?| 2023-06-09..2024-05-31锛?024-07-05..2025-06-27锛?025-08-01..2026-06-26 |
| 鍙楁帶寮€鍙戝懆闈㈡澘瑕嗙洊 | 2018-06-08..2023-06-02 |
| 寮€鍙戞湡澶嶇畻 | 鍛ㄩ潰鏉?`market_volatility_4w` 涓庡紑鍙戞牱鏈殑鍚屽悕 C0 鐗瑰緛閫愰」涓€鑷达紙鍚己澶卞€硷級 |
| 椤圭洰鏃㈡湁璧勪骇瀹¤ | `PRODUCTION_T2_RECONSTRUCTION_ASSETS_RESTORED_20260814.md` 宸叉槑纭€滃巻鍙插競鍦哄熀鍑嗗師濮嬫棩绾垮強 SHA 鏈畾浣嶁€?|

鍥犳锛岀幇鏈夊懆闈㈡澘涓嶈兘瑕嗙洊浠讳竴 FRESH origin锛汧RESH 閲嶅缓鍖呬篃娌℃湁浼犻€掕鐗瑰緛銆傛妸鏁存缂哄け鐩存帴閫佸叆 WP09 鐨勨€滅湡瀹為潪鏈夐檺鍊尖啋鏍囧噯鍖栭浂鈥濆鐞嗭紝灞炰簬鎶?*渚涘簲缂哄け**浼鎴?*瑙傛祴缂哄け**锛屼細鏀瑰彉鍊欓€夌殑璇佹嵁鏉′欢锛岀姝㈡墽琛屻€?
## 闇€瑕佽幏鍙栫殑鏈€灏忚ˉ浠?
浜岄€変竴锛屼絾閮藉繀椤诲湪鍐嶆 FRESH 棰勬祴鍓嶅崟鐙喕缁撳苟楠屾敹銆?
1. 鎺ㄨ崘锛氫氦浠樺師濮?PIT 甯傚満/鍏ㄥ競鍦烘棩绾垮強鍘熷 `weekly_panel` 鐗╁寲鑴氭湰锛岃鐩栬嚦灏?2023-05-12 鑷?2026-06-26锛堝墠缃洓鍛ㄧ獥鍙ｉ渶瑕佸巻鍙诧級銆傝剼鏈繀椤昏兘浠ュ悓涓€鍙ｅ緞鐗╁寲 `market_volatility_4w`銆?2. 鍙帴鍙楋細浜や粯鍙惈鏍囩鏃犲叧灏哄害鐘舵€佺殑鍙楁帶鏂囦欢锛屼緥濡?`fresh_market_volatility_4w_v1.csv.gz`锛岃嚦灏戝寘鎷細
   - `trade_date`銆乣market_volatility_4w`銆乣source_trade_date`锛?   - 瑕嗙洊涓変釜 FRESH 鏂囦欢姣忎竴涓?origin date锛屾寜鏃ユ湡鍞竴锛?   - 鐢熸垚鑴氭湰銆佹墍鏈夊師濮嬭緭鍏ヨ矾寰勪笌 SHA-256銆佹枃浠?SHA-256銆丳IT/鍙緱鏃堕棿璇存槑锛?   - 璇佹槑鍏剁敓鎴愰€昏緫涓?2018-06-08..2023-06-02 鐨勫彈鎺?`weekly_panel.csv.gz` 閫愭棩鏈熺簿纭竴鑷达紱
   - 涓嶅惈浠讳綍 H4 鏍囩銆佹湭鏉?close銆乼arget return銆乷rdinal target銆丗RESH 璇勪及鎸囨爣鎴栨ā鍨嬭緭鍑恒€?
琛ヤ欢蹇呴』鏀惧叆鍙楁帶椤圭洰鐩綍锛屼笉鍙鐩栧師 FRESH 鏂囦欢锛涢殢鍚庣敓鎴愪竴浠藉寘鍚ˉ浠?SHA銆佺敓鎴愬櫒 SHA銆佸師濮嬫棩绾?SHA 鍜岄噸鏀句竴鑷存€х粨鏋滅殑 **WP10 preconsumption supplement freeze**锛屽浐瀹氬瓧娈典负 `node_id=WP10_FRESH_MARKET_VOLATILITY_SUPPLEMENT_FREEZE_V1`銆乣status=FROZEN_BEFORE_WP10_FRESH_PREDICTION_SEAL`銆乣market_state_sha256` 涓?`development_weekly_panel_sha256`銆傝 freeze 闇€瑕佺敤鎴风‘璁ゅ悗锛屾墠鑳戒粠鍏ㄦ柊鐨?`wp10_*_v3` 鐩綍閲嶆柊鍚姩涓€娆￠娴嬪皝瀛樸€?
## 鎭㈠鍚庣殑鍥哄畾椤哄簭

1. 楠岃瘉琛ヤ欢涓?freeze锛涢€愭棩鏈熼噸鏀惧紑鍙戞湡涓€鑷存€э紱楠岃瘉涓変唤 FRESH key 闅旂銆?2. 浠呯敤寮€鍙戦泦鎷熷悎鍐荤粨 C0锛屽苟灏佸瓨 C0銆乣NAIVE_PRIOR`銆乣NAIVE_NEUTRAL` 涓庡彧璇?incumbent 棰勬祴鍙?SHA銆?3. 浠呭湪鎵€鏈夐娴?SHA 宸插皝瀛樺悗锛岀敱鐙珛 evaluator 璇诲彇涓€娆℃爣绛惧苟鍑哄叿缁撴灉銆?4. incumbent 鍦ㄤ笁浠?FRESH 涓婂繀椤诲缁堟爣璁颁负 `INCUMBENT_PRODUCTION_IN_SAMPLE_REFERENCE_NOT_VALID_FOR_SELECTION`锛涘嵆浣?C0 閫氳繃锛屼篃鍙彲杩涘叆鍗曟満 shadow/paper 鍑嗗锛屼笉鑳芥浛鎹㈢敓浜с€?

