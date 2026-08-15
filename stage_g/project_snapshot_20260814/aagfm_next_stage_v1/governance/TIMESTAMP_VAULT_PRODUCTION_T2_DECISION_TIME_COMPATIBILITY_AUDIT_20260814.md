# timestamp-authority vault 涓庣敓浜?T2 鍐崇瓥鏃剁偣鍏煎鎬у璁★紙2026-08-14锛?
缁撹锛歚FAIL_CLOSED_DECISION_TIME_MISMATCH_REQUIRES_NEW_FROZEN_ALIGNMENT`

## 宸叉牳楠屼簨瀹?
- 澶栭儴 vault 璺緞鍙彧璇昏闂細`D:\tool\opencode\xiangmu\timestamp_authority\training_alignment\`銆?- 鍐荤粨鍙傛暟 SHA-256锛歚23defd8fcf4878e1a660808fd66e9ad226fe052d74c88e3546e339d48b645a55`銆?- 瀵归綈琛?SHA-256锛歚af752e854498429bcfef89a47f8be3ff7ac55336085b25d5443c873116c96ee2`锛?66 涓?weekly origin锛岃鍒欎负 `weekly Monday 09:30 Asia/Shanghai`锛屼粎閲囩敤 strict-PIT minute 浜嬩欢銆?- 浜嬩欢琛?SHA-256锛歚ba1fc3e6ba5cb8874f3c58f271f13242c1f24930ad4fc046054ccee0dc6f2a0e`锛?,623 涓簨浠躲€傛瘡涓簨浠惰褰曚粎鍚?`stock_code`銆乣announcement_id`銆乣published_at_utc`銆乣body_txt_sha256` 鍜?`ocr_verdict`锛屾湰瀹¤鏈鍙栨鏂囧唴瀹广€?- 宸叉湁 H4 origin 娉ㄥ唽琛ㄤ负鍛ㄤ竴 origin锛圫HA-256 `b68c38acc5a672dd852c8f41f0ce0604bf881a8e809884f36e6827706844a2f9`锛夛紝鍥犳涓婅堪 vault 瀵归綈鍙綔涓虹嫭绔?H4 鐮旂┒鏁版嵁瀹¤鐨勫€欓€夎緭鍏ャ€?- 鐢熶骇 T2 宸插喕缁撲负鍛ㄤ簲 trade-date/鍛ㄤ簲鏀剁洏鐨勫喅绛栫偣锛岄殢鍚庨娴嬪洓鍛ㄥ競鍦虹浉瀵规敹鐩婏紱瀹冧笉鏄懆涓€ 09:30 鐨?H4 鐮旂┒鍐崇瓥鐐广€?
## 涓嶅吋瀹瑰師鍥?
鑻ユ妸褰撳墠鈥滃懆涓€ 09:30 鎴鈥濈殑浜嬩欢姹囧叆浠ュ墠涓€鍛ㄥ懆浜旀敹鐩樹负杈撳叆鏃剁偣鐨勭敓浜?T2锛屽懆涓€鑷冲懆浜旀湡闂撮娆″叕寮€鐨勪簨浠朵細鍦ㄥ洖鐪嬩腑鍑虹幇鍦ㄨ鍛ㄤ簲涔嬪墠鐨勭壒寰佸唴銆傝繖鏋勬垚鏃堕棿绌胯秺锛屼笉鑳戒互缂哄け濉厖銆佸欢鍚庢爣绛炬垨浜嬪悗杩囨护琛ユ晳銆?
## 寮哄埗鍚庢灉

- 褰撳墠 ALIGN-FREEZE-001 涓嶈兘杩涘叆鐢熶骇 T2 鐨勮缁冦€佹牎鍑嗐€佸€欓€夐€夋嫨銆乻hadow銆乸aper 鎴栨湇鍔°€?- 涓嶈鍙?FRESH銆丼CREENING銆丗INAL锛屼篃涓嶆敼鍙樼敓浜у唴鏍告垨娉ㄥ唽琛ㄣ€?- 涓嶆妸璇ュけ璐ュ綊鍜庝簬淇″彿寮哄急锛涘畠鏄緭鍏ユ椂闂磋涔変笉鍏煎锛屾湭浜х敓浠讳綍妯″瀷鎸囨爣銆?
## 绔嬪嵆缂哄彛锛氭柊鍐荤粨瀵归綈鎵€闇€鐨勬潈濞佸畾涔?
鍦ㄤ换浣曠敓浜?T2 vault 淇″彿瀹¤鍓嶏紝蹇呴』鐢辨暟鎹?绛栫暐璐熻矗浜烘彁渚涘苟鍐荤粨锛?
1. 姣忎釜 T2 origin 鐨勭簿纭彲瑙佹€ф埅姝㈡椂鍒伙紙鏃ユ湡銆佹椂鍖恒€佹椂鍒嗙锛夛紝浠ュ強鏀剁洏鍚庡叕鍛婃槸鍚﹀睘浜庡綋鍛ㄨ緭鍏ワ紱
2. 闈炰氦鏄撳懆銆佽妭鍋囨棩鍜屽懆浜旈潪浜ゆ槗鏃ョ殑 origin 瑙勫垯锛?3. 涓庤缁?楠岃瘉/future 绐楀彛涓€鑷寸殑 T2 origin 鍒楄〃鍙婂叾 SHA-256锛?4. `published_at_utc == cutoff` 鐨勮竟鐣屽綊灞炶鍒欙紱
5. 鏂板榻愯剼鏈€佸叾 SHA-256銆乻trict-PIT 浜嬩欢绛涢€夊拰鏃犱簨浠?mask 鐨勪笉鍙彉鐗堟湰銆?
鍙栧緱杩欎簺鍐呭鍚庯紝鍙兘鍙﹀缓鏂扮洰褰曞拰鏂?freeze ID锛岄噸鏂扮敓鎴愬苟瀹¤ T2 涓撶敤瀵归綈琛紱涓嶅緱瑕嗙洊鎴栦慨鏀?ALIGN-FREEZE-001銆?

