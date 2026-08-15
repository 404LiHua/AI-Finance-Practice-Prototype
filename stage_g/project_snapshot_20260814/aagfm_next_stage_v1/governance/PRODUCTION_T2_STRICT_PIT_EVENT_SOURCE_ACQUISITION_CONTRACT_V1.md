# 鐢熶骇 T2 strict-PIT 浜嬩欢婧愯幏鍙栦笌楠屾敹鍚堝悓 V1

鐘舵€侊細`REQUIRED_BEFORE_T2_EVENT_FEATURE_MATERIALIZATION`

鏈悎鍚屽彧涓虹敓浜х洰鏍?`T2_MARKET_RELATIVE_FIXED` 寤虹珛浜嬩欢婧愯緭鍏ワ紱瀹冧笉鎺堟潈璇诲彇 FRESH銆佽缁冨€欓€夈€佹敼鍙樻敞鍐岃〃銆侀儴缃叉垨浜ゆ槗銆傛墍鏈夋暟鎹互鏂?freeze ID 浜や粯锛屼笉鑳借鐩栫幇鏈?H4/Monday vault銆?
## 浜や粯鐩綍

```text
<freeze_root>/
  FREEZE_MANIFEST.json
  T2_ORIGIN_CUTOFFS.csv
  PIT_UNIVERSE_MEMBERSHIP.csv
  SOURCE_COVERAGE_RECEIPTS.csv
  STRICT_PIT_EVENTS.parquet
  SHA256_MANIFEST.csv
```

鎵€鏈夋枃鏈负 UTF-8锛涙棩鏈熶负 `YYYY-MM-DD`锛涙椂闂翠负 ISO-8601 UTC锛涙瘡涓枃浠剁殑瀛楄妭 SHA-256 蹇呴』鍒楀湪 `SHA256_MANIFEST.csv`銆?
瀹¤鏃惰繕蹇呴』浼犲叆椤圭洰鏂圭嫭绔嬪喕缁撶殑闈炴爣绛?`T2_PRODUCTION_TRAIN_ORIGIN_REGISTRY_V1.csv`銆傛暟鎹柟鐨?`T2_ORIGIN_CUTOFFS.csv` 蹇呴』涓庡叾 `trade_date` 闆嗗悎瀹屽叏鐩稿悓锛涗笉寰楅€氳繃缂╃煭 origin 绐楀彛缁曡繃瑕嗙洊瀹¤銆?
## 蹇呴渶瀛楁

| 鏂囦欢 | 蹇呴渶瀛楁 | 瑙勫垯 |
| --- | --- | --- |
 | `T2_ORIGIN_CUTOFFS.csv` | `trade_date,cutoff_at_utc,cutoff_rule_id` | 閫愬懆鍞竴锛沗trade_date` 涓庢寮?T2 origin 涓€鑷达紱`cutoff_at_utc` 鏄瓥鐣ヨ礋璐ｄ汉绛惧彂鐨勭簿纭彲瑙佹€ц竟鐣岋紝Asia/Shanghai 鏈湴鏃ユ湡蹇呴』涓庤 origin 鍚屾棩銆?|
 | `PIT_UNIVERSE_MEMBERSHIP.csv` | `trade_date,stock_code,eligible,membership_effective_at` | 姣忎釜 origin 閮藉繀椤绘湁鎴愬憳琛岋紱鎴愬憳鐢熸晥鏃剁偣涓嶅緱鏅氫簬璇?origin cutoff锛涚姝㈠瓨娲昏€呭洖濉紱涓嶅彲鎶婇潤鎬?300 姹犲啋鍏?PIT 312 姹犮€?|
| `SOURCE_COVERAGE_RECEIPTS.csv` | `stock_code,coverage_start_date,coverage_end_date,coverage_status,source_system,source_snapshot_sha256` | 瀵规瘡涓彲鑳芥湁鏁堟垚鍛樿瘉鏄庢潵婧愬凡瑕嗙洊銆俙NO_EVENTS` 鍙湁鍦?coverage receipt 瑕嗙洊璇ユ湡闂存椂鎵嶅彲瑙ｉ噴涓哄彲瑙傛祴闆朵簨浠躲€?|
 | `STRICT_PIT_EVENTS.parquet` | `event_id,stock_code,published_at_utc,source_response_sha256,source_url` | `event_id` 鍏ㄥ眬鍞竴锛涙瘡涓€琛屾湁鐪熷疄鍙戝竷鏃堕棿銆侀潪绌烘潵婧?URL 涓庢潵婧愬搱甯岋紱鍐荤粨涓笉寰楀嚭鐜版櫄浜庢渶鍚庝竴涓娉ㄥ唽 origin cutoff 鐨勪簨浠躲€傛鏂囧彲涓嶄氦浠橈紱姝ｆ枃鍝堝笇鍙€夈€?|
| `FREEZE_MANIFEST.json` | `status,freeze_id,target_id,timezone,cutoff_authority,strict_inclusion_rule` | `target_id=T2_MARKET_RELATIVE_FIXED`锛涗弗鏍肩撼鍏ヨ鍒欏繀椤讳负 `published_at_utc <= cutoff_at_utc` 鎴栨洿淇濆畧鐨勫彲瀹¤绛変环瑙勫垯銆?|

## 蹇呴』鐢辫礋璐ｄ汉鏄庣‘绛惧彂鐨勯€夋嫨

1. 鍛ㄤ簲鏀剁洏鍚庡叕鍛婂睘浜庡綋鍓?origin 杩樻槸涓嬩竴 origin锛?2. 闈炰氦鏄撳懆銆佽妭鍋囨棩銆佷复鏃跺仠甯傜殑 origin 鍜?cutoff锛?3. `published_at_utc == cutoff_at_utc` 鐨勫綊灞烇紱
4. 鏉ユ簮鍙戠敓淇鏃剁殑鐗堟湰閾惧拰閲嶈窇绛栫暐銆?
娌℃湁杩欎簺瀛楁鏃讹紝涓嶅緱浠ュ懆涓€ H4 vault銆佹棩绾ф椂闂淬€佹姄鍙栨椂闂淬€侀粯璁?15:00 鎴栨帹娴嬭鍒欐浛浠ｃ€?
## 楠屾敹涓庡け璐ュ叧闂?
浣跨敤 `scripts/audit_t2_strict_pit_event_freeze_v1.py` 杩涜 CPU-only 棰勬秷璐瑰璁°€傚畠鑷冲皯蹇呴』楠岃瘉锛氳緭鍏?SHA銆乷rigin 鍞竴鎬с€佸懆浜?T2 origin銆乧utoff 鏈湴鏃ユ湡銆佹瘡涓?origin 鎴愬憳琛屻€佹垚鍛樼敓鏁堟椂鐐广€佹湁鏁堟垚鍛樺叏瑕嗙洊銆佷簨浠堕敭鍞竴銆乁TC 鏃堕棿鍙В鏋愩€佹潵婧?URL/鍝堝笇鏍煎紡銆佷簨浠朵笉鏅氫簬鏈€鍚?cutoff锛屼互鍙婃棤鏍囩/鏃?FRESH 杈圭晫銆?
浠讳竴澶辫触鍗宠緭鍑?`FAIL_CLOSED_T2_EVENT_FREEZE_PRECONSUMPTION`锛屼笉寰楃墿鍖栦簨浠剁壒寰佹垨鍚姩 GPU銆傞€氳繃涔熷彧鍏佽涓嬩竴姝ュ崟鐙娉ㄥ唽鐨?train-only 淇″彿瀹¤銆?

