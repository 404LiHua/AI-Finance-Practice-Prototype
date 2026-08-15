# AA-GFMNet 姝ｅ紡鏁版嵁涓庡師 incumbent 浜や粯瑙勬牸 V1

鏈枃浠舵槸鏁版嵁渚涘簲鏂圭殑鍙墽琛屼氦浠樻竻鍗曘€備氦浠樺墠涓嶅緱鐢熸垚姝ｅ紡妯″瀷鎴愮哗锛涙敹鍒板悗蹇呴』鍏堥€氳繃鑷姩楠屾敹銆?
## A. 姝ｅ紡鐮旂┒闈㈡澘

- 鍐荤粨 300 鑲＄エ鍙婇『搴忥紝蹇呴』缁戝畾 `M5_FORMAL_STOCK_UNIVERSE_300.csv` 鐨?SHA-256銆?- 337 涓?origin锛宱rigin 椤哄簭銆佹棩鏈熴€佹椂鍖哄拰 cutoff 鏄庣‘璁板綍銆?- 瀹屾暣閿泦鍚?`origin_id|stock_code`锛屽叡 101100 涓紱缂哄け閿繚鐣欏苟鐢?mask / `prediction_valid=false` 琛ㄧず銆?- H4 杩炵画鐩稿鏀剁泭鏍囩銆乀2 闃堝€兼淳鐢熻鍒欏拰鏍囩鐢熸垚鑴氭湰銆?- 鍏姌璧风偣 `180,198,216,234,252,270`锛宲urge/embargo=11 鍛紝split registry 鍜屾瘡鎶樿缁冭竟鐣屻€?- origin-level PIT 蹇収 `(337,300,6)`锛宻equence PIT 蹇収 `(337,8,300,6)` 鎴栧彲澶嶇幇缁勮鍣紱姣忎釜浣嶇疆璁板綍婧愪氦鏄撴棩銆佹潵婧愮増鏈拰 mask銆?- 鎵€鏈夋枃浠堕檮 SHA-256锛涚姝㈢敤 crawl time 鏇夸唬棣栨鍏紑鏃堕棿銆?
## B. 鍘?incumbent stock_node_gwnet_fixed_industry_l8

蹇呴』鍚屾椂浜や粯 M01鈥揗14锛氬叚鎶樻潈閲嶃€侀€愭姌閰嶇疆/閫夋嫨鏀舵嵁銆佸畬鏁存簮鐮併€佺幆澧冮攣銆佸師濮嬭涓氬浘銆丳IT 鐗瑰緛涓庡簭鍒椼€丠4 鏍囩銆乻plit/purge registry銆?01100 閿?OOF銆丯aive 閰嶅 OOF銆佸姞杞藉鐜版祴璇曘€丼HA manifest 涓?delivery receipt銆備换浣曚竴椤圭己澶卞潎涓?`FAIL_CLOSED`銆?
## C. 鑾峰彇鏂瑰紡

渚涘簲鏂瑰彲閫氳繃浠ヤ笅浠讳竴鏂瑰紡浜や粯锛?
1. 涓€涓彧璇诲帇缂╁寘锛岃В鍘嬪埌 `sources/incumbent_model_textcu_<delivery_version>/`锛屾牴鐩綍鍖呭惈 `DELIVERY_RECEIPT.json`銆乣SHA256_MANIFEST.csv`銆?2. 涓€涓増鏈寲鐩綍鎴?Git 浠撳簱瀵煎嚭鍖咃紱蹇呴』鍚屾椂鎻愪緵 commit/version銆佸畬鏁翠緷璧栭攣鍜屼笉鍙彉 SHA manifest銆?
涓嶅緱閫氳繃鑱婂ぉ绮樿创閮ㄥ垎鏂囦欢銆佷复鏃堕摼鎺ユ垨鍙粰妯″瀷鏉冮噸锛涗氦浠樺繀椤昏兘鍦ㄥ崟鏈虹幆澧冧腑閲嶆斁銆?
## D. 鑷姩楠屾敹闂ㄦ

璺緞瀛樺湪銆佸搱甯屽尮閰嶃€侀敭鏁?褰㈢姸/椤哄簭姝ｇ‘銆佸叚鎶樿竟鐣屾纭€丳IT 瑙勫垯鍙璁°€丱OF 瑕嗙洊 101100 閿€佸姞杞藉鐜版祴璇曢€氳繃锛屼笖鏃?Test/FRESH 璇诲彇鐥曡抗鍚庯紝鎵嶅厑璁哥敓鎴愪竴娆℃€ф巿鏉冦€傚惁鍒欏彧鐢熸垚缂哄彛鎶ュ憡銆?

