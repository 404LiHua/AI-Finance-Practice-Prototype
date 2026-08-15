# AA-GFMNet 宸ヤ綔鍖?04锛氭爣绛捐涔変笌 PIT 瀹¤

## 瀹¤缁撴灉

瀹¤鏀舵嵁锛歚audits/154_origin_semantics_audit_v1/SEMANTICS_AUDIT.json`

- 寮犻噺褰㈢姸锛歚154脳300脳36`銆?- H4 鍜?T2 鏍囩瀹炵幇鏃堕棿鍧囨櫄浜?origin锛屾爣绛炬柟鍚戦€氳繃銆?- 浣嗘妧鏈壒寰佺殑 `source_trade_date == origin_date` 鍏?46,200 琛岋紝`source_trade_date > origin_date` 涓?0 琛屻€?- 鐮旂┒鍗忚瑕佹眰涓ユ牸 `source_trade_date < origin_date`锛涘洜姝ゅ綋鍓嶆妧鏈壒寰侀摼涓嶆弧瓒充弗鏍?PIT 闂ㄧ銆?
## 褰卞搷

姝ゅ墠 Ridge/XGBoost 缁撴灉鍙兘瑙嗕负鈥滃惈 PIT 杈圭晫椋庨櫓鐨勫紑鍙戣瘖鏂€濓紝涓嶈兘浣滀负姝ｅ紡妯″瀷鎬ц兘銆佺敵鎶ヤ功璇佹嵁鎴栫敓浜у€欓€変緷鎹€備笉鑳介€氳繃鍦ㄧ粨鏋滃眰闈㈣ˉ鍋胯闂銆?
## 蹇呴』鑾峰彇/淇鐨勫唴瀹?
渚涘簲鏂规垨鏁版嵁鏋勫缓鏂瑰繀椤绘彁渚涳細

1. origin 鍐崇瓥鏃跺埢鐨勭‘鍒囧畾涔夛紙鍛ㄤ竴 09:30 鎴栧叾浠?cutoff锛夛紱
2. 姣忎釜鐗瑰緛鐨勭湡瀹?source timestamp锛岃€屼笉鍙槸 source_trade_date锛?3. 璇佹槑褰撴棩鏀剁洏鏁版嵁鍦ㄥ喅绛栨椂鍒讳笉鍙敤锛屾垨閲嶆柊鐢熸垚涓ユ牸 `< origin_date` 鐨勭壒寰佸揩鐓э紱
4. 鏇存柊鍚庣殑鐗瑰緛 SHA-256銆佹瀯寤鸿剼鏈€丳IT 瀹¤鏀舵嵁銆?
鍦ㄦ涔嬪墠锛屽浘銆侀鍩熴€侀瞾妫掕缁冦€佹枃鏈瀺鍚堝拰鐢熶骇鏅嬬骇鍏ㄩ儴鏆傚仠锛涚敓浜у唴鏍镐繚鎸?`RG_OBGNET_CONFIRMED_SAFE_V1_1`銆?

