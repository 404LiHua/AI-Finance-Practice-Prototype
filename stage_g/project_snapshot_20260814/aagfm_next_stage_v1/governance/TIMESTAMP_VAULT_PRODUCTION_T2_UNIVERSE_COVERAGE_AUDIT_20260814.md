# timestamp-authority vault 瀵圭敓浜?T2 PIT 鑲＄エ姹犵殑瑕嗙洊瀹¤锛?026-08-14锛?
缁撹锛歚FAIL_CLOSED_UNIVERSE_COVERAGE_40_OF_312_NOT_USABLE_FOR_T2_EVENT_RESEARCH`

## 瀹¤鑼冨洿涓庢湭璇诲彇杈圭晫

鏈璁″彧璇诲彇鐢熶骇 T2 `samples.csv.gz` 鐨?`trade_date`銆乣stock_code`銆乣fold_id`銆乣split_role` 鍥涗釜闈炴爣绛惧瓧娈碉紝骞惰鍙?vault minute 鐩綍鐨勬枃浠跺悕銆傛病鏈夎鍙?target銆乷rdinal class銆佹敹鐩娿€丗RESH銆丼CREENING銆丗INAL銆佸叕鍛婃鏂囨垨妯″瀷鎸囨爣銆?
## 杈撳叆韬唤

- 鐢熶骇 T2 寮€鍙戞牱鏈細`data/rg1_4_materialized/samples.csv.gz`锛孲HA-256 `60c1322c228a7a2e52b9c0b5dec054ba5a95de242170cca17759a7fb631091b6`銆?- 鐢熶骇 T2 PIT 瀹囧畽绾﹀畾锛歚POINT_IN_TIME_UNIVERSE_CONTRACT.json`锛孲HA-256 `f64740b608b50532cf91859fa5c137f7f13acf71a6f958a00db6691f547306f8`锛涘畠瀹氫箟 312 鏀偣鏃舵垚鍛橈紙鏈€灏?300锛夛紝骞剁姝㈢敤褰撳墠瀛樻椿鑲＄エ鍥炲～銆?- vault 鐜版湁鍐荤粨鍙傛暟锛歚D:\tool\opencode\xiangmu\timestamp_authority\training_alignment\FROZEN_PARAMS.json`锛孲HA-256 `23defd8fcf4878e1a660808fd66e9ad226fe052d74c88e3546e339d48b645a55`銆?
## 鍙绠楃粨鏋?
鍦ㄧ敓浜?T2 寮€鍙戞湡 `2018-06-08` 鑷?`2023-06-02` 鐨勬牱鏈储寮曚腑锛?
| 椤圭洰 | 鏁伴噺 |
| --- | ---: |
| 鐢熶骇 T2 PIT 鑲＄エ浠ｇ爜 | 312 |
| 鐢熶骇 T2 origin | 257 |
| 鐜版湁 vault minute 鏂囦欢 | 302 |
| 缁忓叚鐮佷唬鐮佽鑼冨寲鍚庣殑閲嶅悎鑲＄エ | 40 |
| 鐢熶骇 T2 鏈鐩栬偂绁?| 272 |

鐢熶骇 T2 鏍锋湰 origin 鍧囦负鍛ㄤ簲锛岃繖涓庣幇鏈?vault 鐨勫懆涓€ 09:30 瀵归綈涔熶笉涓€鑷达紱涓ら」闂褰兼鐙珛銆?
## 寮哄埗鍚庢灉

1. 涓嶅緱鎶婄己灏?vault 鏂囦欢鐨?272 鏀偂绁ㄧ紪鐮佷负鈥滈浂浜嬩欢鈥濇垨鏅€氱己澶卞€硷紱杩欎唬琛ㄩ噰闆嗘睜涓嶈鐩栵紝涓嶆槸鍙娴嬬殑鏃犱簨浠躲€?2. 涓嶅緱鍩轰簬璇?vault 鏋勫缓鐢熶骇 T2 浜嬩欢鐗瑰緛銆佹畫宸€侀闄╂爣璁般€佽缁冦€佹牎鍑嗐€佸€欓€夐€夋嫨銆乻hadow 鎴?paper 缁撴灉銆?3. 姝ょ粨璁轰笉褰卞搷 vault 鍦ㄥ叾鍘?300 鏀€佸懆涓€ H4 鐮旂┒鑼冨洿鍐呯殑璇佹嵁韬唤锛涗絾瀹冧笉鑳藉鎺ㄤ负鐢熶骇 T2 瑕嗙洊銆?
## 鑾峰緱鍙敤 T2 浜嬩欢杈撳叆鐨勭簿纭潯浠?
闇€瑕佹柊鐨勩€佷笉鍙鐩栫殑 T2 涓撶敤 freeze锛屽悓鏃舵弧瓒筹細

- 瑕嗙洊鐢熶骇 PIT 瀹囧畽鐨勫叏閮?312 鏀紙鎴栫敱鐐规椂鍙氦鏄撹鍒欒瘉鏄庣殑姣忎竴 origin 鏈夋晥鎴愬憳锛夛紱
- 姣忎釜鍏憡鏈夊彲瀹¤ `published_at_utc` 鍜屾潵婧愬搱甯岋紱
- 瀵规瘡涓懆浜?origin 浣跨敤鍗曠嫭鍐荤粨鐨勩€佷弗鏍兼棭浜庡喅绛栨椂鐐圭殑鍙鎬ф埅姝紱
- 鏃犱簨浠朵笌鏃犺鐩栧垎寮€缂栫爜锛?- 杈撳叆鑲＄エ/鏃ユ湡瑕嗙洊銆丼HA銆丳IT 鍜岄敭鍩熷璁″叏閮ㄩ€氳繃鍚庯紝鎵嶅彲鐢宠鐙珛 train-only 淇″彿璇勪及鎺堟潈銆?

