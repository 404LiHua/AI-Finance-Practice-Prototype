# AA-GFMNet 宸ヤ綔鍖?01锛氭暟鎹绾︿笌 train-only 璇婃柇鍩虹嚎

## 鏈伐浣滃寘宸插畬鎴?
- 鍐荤粨缁熶竴濂戠害锛歚governance/AA_GFMNET_DATA_CONTRACT_V1.json`
- 鍙瀹¤鑴氭湰锛歚scripts/audit_train_only_inputs_v1.py`
- 杈撳叆瀹¤鏀舵嵁锛歚audits/train_only_input_contract_audit_v1/`
- 鍙璇婃柇鍩虹嚎鑴氭湰锛歚scripts/run_train_only_diagnostic_baseline_v1.py`
- 璇婃柇鏀舵嵁锛歚audits/train_only_diagnostic_baseline_v1/`

## 缁撹

褰撳墠涓や釜 NPZ 鐪熷疄鍙锛屼笖鍝堝笇宸茶褰曪細

- sequence cube锛?99 origins 脳 5796 stocks 脳 14 features锛汼HA-256 `8ff48ae656fa177a86928831efb01cded512c8fe4f999d22831fd919da289430`
- multiview cube锛?99 origins 脳 5796 stocks 脳 49 features锛汼HA-256 `aec078bb10b39ccf4b3761118f649aaef4e37315b3fb0077987ddd9f091366a0`

瀹冧滑鏄?train-only 鏋舵瀯/鏁版嵁璇婃柇璧勪骇锛屼笉鏄崗璁姹傜殑 337脳300 姝ｅ紡闈㈡澘锛屼篃涓嶆槸鍘?incumbent 浜や粯銆傚洜姝ゆ湰宸ヤ綔鍖呬笉鐢熸垚姝ｅ紡鍏姌 OOF銆佷笉璁粌鑱斿悎妯″瀷銆佷笉瑙︾ Test/FRESH銆佷笉鏀瑰彉鐢熶骇娉ㄥ唽琛ㄣ€?
## 璇婃柇缁熻锛堥潪妯″瀷鎴愮哗锛?
sequence cube 鐨勬湁鏁堣娴嬫暟涓?1,321,849锛涚浉瀵规敹鐩婂潎鍊?0.01390869銆佹爣鍑嗗樊 0.12592271锛涙寜 卤1% 闃堝€?UP/DOWN/NEUTRAL 姣斾緥鍒嗗埆涓?44.4617% / 44.3000% / 11.2383%銆傝繖浜涙暟鍊煎彧鐢ㄤ簬妫€鏌ユ爣绛惧垎甯冨拰鏁版嵁鍋ュ悍锛屼笉鑳戒綔涓烘硾鍖栨€ц兘鎴栬涓氶鍏堣瘉鎹€?
## 涓嬩竴姝ヨ繘鍏ユ潯浠?
渚涘簲鏂瑰繀椤昏ˉ榻愶細姝ｅ紡 337 origins銆佸喕缁?300 鑲＄エ姹犲強椤哄簭銆?01100 閿€佸叚鎶?split/purge/embargo銆丠4/T2 鏍囩鐢熸垚鑴氭湰鍜屽畬鏁?PIT 鏀舵嵁锛涘師 incumbent 浠嶉渶鎸?M01鈥揗14 gap register 瀹屾垚浜や粯銆傝ˉ榻愬悗鎵嶅彲杩愯姝ｅ紡 Naive銆佺敓浜ч敋鐐广€丠4 Ridge/XGBoost銆乀2 proportional-odds 鍜屽€欓€変富妯″瀷銆?
R26 渚濇嵁 `R26_CLOSURE_REPORT.md` 缁存寔 `FROZEN_TEST_FAIL` / `promotion_allowed=false`锛屼笉寰楁檵绾с€?

