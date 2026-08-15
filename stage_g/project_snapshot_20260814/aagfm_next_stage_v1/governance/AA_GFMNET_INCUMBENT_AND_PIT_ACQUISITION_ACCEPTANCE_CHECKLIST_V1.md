# AA-GFMNet 鍘?incumbent 涓?PIT 鏁版嵁鑾峰彇楠屾敹娓呭崟 V1

**鐘舵€?*锛歚ACQUISITION_AND_ACCEPTANCE_PACKAGE_READY`  
**鐢熸垚鏃ユ湡**锛?026-08-13  
**閫傜敤鑼冨洿**锛氬師 incumbent銆丳IT 鍩烘湰闈€佸叕鍛?鏂囨湰銆佸垎鏋愬笀棰勬祴鍙?Test/FRESH 鎺堟潈鏉愭枡  

> 鏈竻鍗曞畾涔夎幏鍙栥€佷氦浠樺拰楠屾敹瑕佹眰锛屼笉绛変簬渚涘簲鏂逛氦浠橈紝涓嶇瓑浜庤缁冩巿鏉冿紝涔熶笉绛変簬 FRESH/Test 鎺堟潈銆?
## 1. 褰撳墠浜嬪疄涓庝笉鍙浛浠ｅ璞?
### 1.1 褰撳墠鐢熶骇鍐呮牳

```text
RG_OBGNET_CONFIRMED_SAFE_V1_1
```

鐢熶骇妯″瀷 SHA-256锛?
```text
d8e4316d0fab70d3785b775c695a1f3a31225edf441a3603e3830f7351c4e2e8
```

### 1.2 鍘?incumbent

鐩爣妯″瀷锛歚stock_node_gwnet_fixed_industry_l8`  
褰撳墠鐘舵€侊細`WAITING_FOR_SUPPLIER_COMPLETE_DELIVERY`

鏈湴宸叉湁鐨勪笁鎶樻棫璧勪骇鍜屾灦鏋勯噸绠楀彧鑳界敤浜庯細

- 鏋舵瀯妫€鏌ワ紱
- 鐙珛鍔犺浇鐑熸祴锛?- 韬唤姣斿銆?
涓嶅緱鐢ㄤ簬锛?
- 337-origin 鍘?incumbent OOF锛?- 褰撳墠娈嬪樊瀹¤锛?- 鏂囨湰铻嶅悎璁粌锛?- 浠ｆ浛渚涘簲鏂瑰叚鎶樹氦浠樸€?
## 2. 鍘?incumbent 浜や粯鍖?
渚涘簲鏂瑰繀椤绘柊寤轰笉鍙鐩栨棫鐗堟湰鐨勭洰褰曪紝渚嬪锛?
```text
C:\Users\27793\Documents\project1\AI_Finance_Prototype\research_tracks\pit_information_incremental_v1\sources\incumbent_model_textcu_<delivery_version>\
```

姣忔琛ヤ氦蹇呴』閫掑鐗堟湰鍙凤紝骞堕噸鏂扮敓鎴?`SHA256_MANIFEST.csv` 鍜?`DELIVERY_RECEIPT.json`銆?
### 2.1 M01-M04锛氭ā鍨嬨€佹簮鐮併€佺幆澧?
| ID | 蹇呴渶鍒跺搧 | 楠屾敹瑕佹眰 | 褰撳墠鐘舵€?| 缂哄け鏃跺姩浣?|
|---|---|---|---|---|
| M01 | 鍏姌妯″瀷鏉冮噸鎴栧姩鎬佽缁冨寘 | 鍏釜楠岃瘉鍧楃嫭绔嬪彲鍔犺浇锛涜褰曟潈閲?SHA銆佺敓鎴愭椂闂淬€佸懡浠?| 缂哄け/鏃ц祫浜т粎韬唤鐢ㄩ€?| 渚涘簲鏂归€掍氦鍏姌鍒跺搧 |
| M02 | 姣忔姌瑙ｆ瀽閰嶇疆鍜?checkpoint 鏀舵嵁 | fit_end銆乻eed銆佽缁冭竟鐣屻€乧heckpoint 閫夋嫨榻愬叏 | 缂哄け | 渚涘簲鏂硅ˉ榻?|
| M03 | 瀹屾暣婧愮爜 | loader銆乼rainer銆乮nference銆乬raph builder銆乴abel builder銆乻plit builder | 缂哄け | 渚涘簲鏂归€掍氦瀹屾暣婧愮爜鏍戝拰鐗堟湰 SHA |
| M04 | 鐜閿佸畾 | Python/PyTorch/CUDA/cuDNN/NumPy/pandas/GPU/纭畾鎬ц缃?| 閮ㄥ垎宸叉湁 | 渚涘簲鏂归€掍氦 lockfile銆丟PU 鏀舵嵁 |

### 2.2 M05-M10锛氳偂绁ㄦ睜銆佸浘銆佺壒寰併€佹爣绛惧拰鍒囧垎

| ID | 蹇呴渶鍒跺搧 | 楠屾敹瑕佹眰 | 褰撳墠鐘舵€?| 缂哄け鏃跺姩浣?|
|---|---|---|---|---|
| M05 | 姝ｅ紡 300 鑲＄エ姹?| 椤哄簭涓庡喕缁撹偂绁ㄦ睜涓€鑷达紝SHA=`87922e5c556234048b8b90523b657a0fef59931dae1e6fe2cedbbe153ba93a21` | 宸叉湁鏈湴鏂囦欢 | 缁戝畾骞惰褰?SHA |
| M06 | 鍘熷琛屼笟鏄犲皠鍜?300脳300 閭绘帴鐭╅樀 | 鏄庣‘ self-loop銆佸悓琛岃繛鎺ャ€佸綊涓€鍖栥€乬raph order锛涚嫭绔嬮噸寤鸿宸?鈮?e-7 | 閮ㄥ垎宸叉湁 | 渚涘簲鏂归€掍氦鍘熷鏉ユ簮鍜岃剼鏈?|
| M07 | `337脳300脳6` PIT snapshot | 瀛楁椤哄簭鍥哄畾锛涙潵婧愭棩鏈熴€佺増鏈€乵ask銆丼HA 榻愬叏 | 鏈湴閲嶇畻闈炰緵搴旀柟鍘熶欢 | 渚涘簲鏂归€掍氦鍘熷蹇収 |
| M08 | `337脳8脳300脳6` sequence snapshot 鎴栫粍瑁呭櫒 | 姣忎釜搴忓垪浣嶇疆鏉ユ簮鏄庣‘锛涙弧瓒?PIT | 鏈湴閲嶇畻闈炰緵搴旀柟鍘熶欢 | 渚涘簲鏂归€掍氦搴忓垪鎴栧畬鏁寸粍瑁呭櫒 |
| M09 | H4 鏍囩鍜屾爣绛捐剼鏈?| 浜ゆ槗鏃ュ巻銆佸疄鐜版棩銆佹爣绛?SHA 榻愬叏锛涗笉寰楁贩鐢?direction 鏍囩 | 鏈湴閲嶇畻閮ㄥ垎鍙敤 | 渚涘簲鏂圭粦瀹氭垨鏄庣‘涓€鑷存€ц瘉鏄?|
| M10 | 337-origin 鍏潡 split registry | 璧风偣 180/198/216/234/252/270锛沺urge/embargo=11 | 鏈湴閲嶇畻鍙敤 | 渚涘簲鏂圭粦瀹氭垨绛剧讲瀹屽叏涓€鑷村０鏄?|

### 2.3 M11-M15锛歄OF銆佸鐜颁笌娌荤悊

| ID | 蹇呴渶鍒跺搧 | 楠屾敹瑕佹眰 | 褰撳墠鐘舵€?| 缂哄け鏃跺姩浣?|
|---|---|---|---|---|
| M11 | 瀹屾暣 101100 閿師 incumbent OOF | 缂哄け閿繚鐣欏苟鏍囪 `prediction_valid=false` | 缂哄け | 渚涘簲鏂归€掍氦 |
| M12 | Naive 閰嶅 OOF 涓庢寚鏍?| 涓庡悓閿師 incumbent 閰嶅锛涗笉鑳界敤鏃у熀绾垮啋鍏?| 閮ㄥ垎宸叉湁 | 鍏ㄩ儴杈撳叆閫氳繃鍚庨噸绠?|
| M13 | 鍥哄畾鏍锋湰鍔犺浇/棰勬祴娴嬭瘯 | 鍏姌鐙珛鍔犺浇骞跺鐜板浐瀹氭牱鏈娴?| 缂哄け | 渚涘簲鏂归€掍氦娴嬭瘯鍜岀粨鏋?|
| M14 | SHA256 manifest 涓?delivery receipt | manifest 涓嶈嚜寮曠敤锛況eceipt 璁板綍 manifest SHA | 缂哄け | 渚涘簲鏂归€掍氦 |
| M15 | 鍘?incumbent 涓€娆℃€у璁℃巿鏉?| 浠呭湪 M01-M14 纭棬鍏ㄩ€氳繃鍚庣敓鎴?| 鏈巿鏉?| 閲嶆柊鐢熸垚鏂版巿鏉冿紝涓嶅鐢ㄦ棫鎺堟潈 |

浠讳竴 M01-M14 纭棬澶辫触锛岀姸鎬佸繀椤讳负 `FAIL_CLOSED`锛屼笉寰楄繘鍏ユ畫宸€佹枃鏈瀺鍚堟垨澶栭儴鐣欏嚭鐮旂┒銆?
## 3. PIT 鏁版嵁鑾峰彇涓庨獙鏀?
### 3.1 宸叉湁鍙敤鏁版嵁

- CSMAR PIT 鍩烘湰闈㈤摼锛氬凡鏈夊璁￠€氳繃缁撹锛屽彲浣滀负鐙珛鏃犳爣绛捐緭鍏ワ紱
- 鏈湴涓ユ牸 PIT 鏁板€煎揩鐓с€佸簭鍒楀揩鐓у拰鏍囩鐩綍锛氬彲鐢ㄤ簬缁撴瀯鏍搁獙鎴栧崗璁粦瀹氾紝涓嶈兘鑷姩绛夊悓浜庝緵搴旀柟鍘熷 incumbent 蹇収锛?- `pit_announcement_text_v1` 鍙婄浉鍏虫枃鏈洰褰曪細蹇呴』閫愭潯鏍搁獙棣栨鍏紑鏃堕棿銆佺増鏈€佹鏂?SHA銆丱CR 鍜岃鐩栧悗锛屾墠鑳借繘鍏ユ寮忕壒寰併€?
### 3.2 浼樺厛鑾峰彇鏉ユ簮

1. CSMAR銆乄ind銆丷ESSET銆丆hoice 绛夊甫鍘嗗彶鐗堟湰鐨勬巿鏉冩暟鎹簱锛?2. 鍒嗘瀽甯堜竴鑷撮鏈?棰勬祴淇锛屽繀椤绘湁瀹為檯鍙戝竷鏃堕棿鍜屽巻鍙茬増鏈紱
3. 浜ゆ槗鎵€鍏憡姝ｆ枃锛屽繀椤绘湁棣栨鍙戝竷鏃堕棿銆佺増鏈摼鍜屾鏂?SHA锛?4. 鏈烘瀯/闄嗚偂閫氭寔浠撳巻鍙插揩鐓э紝蹇呴』鏈夊巻鍙插揩鐓у拰鍙戝竷鏃堕棿锛?5. 渚涘簲閾惧叧绯伙紝蹇呴』鏈夌敓鏁堛€佺粓姝㈡棩鏈熷拰鐗堟湰銆?
Tushare Pro 浠呭湪鏉冮檺瓒冲銆佽皟鐢ㄥ彲鎭㈠涓斿畬鏁村巻鍙茬増鏈璁￠€氳繃鏃朵綔涓烘寮忔潵婧愶紱褰撳墠鍘嗗彶缁撹涓烘潈闄愪笉瓒筹紝涓嶅緱鐩存帴鍚姩姝ｅ紡淇″彿瀹¤銆?
### 3.3 鏂囨湰/鍏憡鏈€灏忓瓧娈?
姣忚涓€鐗堟湰锛孭arquet 鎴?UTF-8 JSONL 鍧囧彲锛?
```text
stable_id / announcement_id / original_url
stock_code
source_exchange / source_name
published_at
crawl_at
document_version
content_sha256
title
body
supersedes_id / correction_relation
pdf_sha256
ocr_quality_flag
```

鍚屾椂鎻愪緵锛氬瓧娈靛瓧鍏搞€佽偂绁ㄤ唬鐮佹槧灏勩€佺己澶卞€煎畾涔夈€侀噰闆嗚寖鍥淬€侀仐婕忔棩蹇楀拰瀹屾暣 SHA-256 娓呭崟銆?
鎺ㄨ崘鐩綍锛?
```text
C:\Users\27793\Documents\project1\AI_Finance_Prototype\research_tracks\pit_information_incremental_v1\sources\pit_announcement_text_v1\
```

### 3.4 PIT 楠屾敹椤哄簭

```text
鍘熷鏂囦欢瀛樺湪鎬?鈫?瀛楁鍜岀紪鐮?鈫?鐗堟湰/鏇存/鎾ゅ洖閾?鈫?棣栨鍙戝竷鏃堕棿涓?origin cutoff
鈫?鑲＄エ鏄犲皠涓庤鐩?鈫?OCR/姝ｆ枃璐ㄩ噺
鈫?缂哄け涓庢棤浜嬩欢閿?鈫?SHA-256 manifest
鈫?train-only 鏉ユ簮娉ㄥ唽
```

浠讳綍鏈潵鏃堕棿琛屻€佸綋鍓嶉〉闈㈠洖濉€佹姄鍙栨椂闂存浛浠ｅ彂甯冩椂闂淬€佺増鏈鐩栧巻鍙叉垨缂哄け鑲＄エ鍒犻櫎锛屽潎涓虹‖澶辫触銆?
## 4. Test/FRESH 鑾峰彇瑕佹眰

Test/FRESH 蹇呴』鐢辩嫭绔嬩繚绠¤€呬竴娆℃€ф巿鏉冿紝鑷冲皯鎻愪緵锛?
- authorization id锛?- candidate id锛?- allowed scope锛?- forbidden scope锛?- issued_at / expires_at锛?- input manifest SHA-256锛?- ACL receipt SHA-256锛?- external signature锛?- one-shot consumption receipt锛?- irreversible score receipt銆?
鑾峰彇娴佺▼锛?
```text
鍊欓€夊喕缁?鈫?鐙珛淇濈鑰呯缃?鈫?棰勬秷璐瑰共璺?鈫?鍗曟璇诲彇
鈫?涓嶅彲閫嗚瘎鍒?鈫?鎺堟潈澶辨晥
```

鐢ㄦ埛鍙ｅご纭銆丄CL 鍙闂€佹ā鏉挎巿鏉冨拰鏃ф巿鏉冨潎涓嶈兘鏇夸唬鏈祦绋嬨€?
## 5. 姝ｅ湪鑾峰彇椤圭洰鐨勭櫥璁版柟寮?
姣忎釜澶栭儴鏁版嵁鍖呮垨渚涘簲鏂逛氦浠樺寘閮藉繀椤诲缓绔嬩互涓嬬櫥璁帮細

```text
delivery_version
source_owner
source_type
coverage_start / coverage_end
stock_coverage
origin_coverage
pit_timestamp_rule
artifact_root
manifest_sha256
receipt_path
authorization_id
status
next_action
```

鐘舵€佸彧鍏佽浣跨敤锛?
```text
MISSING
REQUESTED
RECEIVING
RECEIVED_UNVERIFIED
PASS_READY_FOR_TRAIN_ONLY
FAIL_CLOSED
```

## 6. 褰撳墠涓嬩竴鑺傜偣

```text
AA_GFMNET_BASELINE_AND_DATA_DELIVERY_ACCEPTANCE_V1
```

璇ヨ妭鐐瑰厛鍋氬彧璇讳氦浠橀獙鏀躲€佹暟鎹敞鍐屽拰鍩虹嚎閲嶅缓鍑嗗銆傛湭鑾峰緱瀹屾暣鍘?incumbent 浜や粯鎴栫嫭绔嬫巿鏉冨墠锛屼笉璁粌鑱斿悎妯″瀷锛屼笉璇诲彇鏂扮殑 Test/FRESH锛屼笉淇敼鐢熶骇鍐呮牳鎴栨敞鍐岃〃銆?

