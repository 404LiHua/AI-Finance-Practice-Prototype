# AA-GFMNet 宸ヤ綔鍖?02锛氭寮忛潰鏉夸笌鍘?incumbent 鑾峰彇/楠屾敹

## 宸插畬鎴?
- [浜や粯瑙勬牸](governance/AA_GFMNET_FORMAL_DATA_AND_INCUMBENT_DELIVERY_SPEC_V1.md)
- [鍒拌揣鎺㈡祴鑴氭湰](scripts/probe_formal_delivery_v1.py)
- [褰撳墠鎺㈡祴鏀舵嵁](audits/formal_delivery_probe_v1/DELIVERY_PROBE_DECISION.json)

## 褰撳墠浜嬪疄

- M05 姝ｅ紡 300 鑲＄エ姹犲凡鎵惧埌骞舵牳楠岋細SHA-256 `87922e5c556234048b8b90523b657a0fef59931dae1e6fe2cedbbe153ba93a21`銆?- 鍘?incumbent 渚涘簲鏂逛氦浠樼洰褰曞皻涓嶅瓨鍦ㄣ€?- `DELIVERY_RECEIPT.json`銆乣SHA256_MANIFEST.csv` 鍙?M01鈥揗14 瀹屾暣浜や粯灏氭湭鍒颁綅銆?- 褰撳墠鐘舵€侊細`WAITING_FOR_SUPPLIER_COMPLETE_DELIVERY`銆?
## 鑾峰彇鍔ㄤ綔

灏嗘湰瑙勬牸鏂囦欢浜ょ粰鏁版嵁/妯″瀷渚涘簲鏂癸紝瑕佹眰鍏舵寜 A鈥揇 鑺備氦浠樹竴涓笉鍙彉鐩綍鎴栧帇缂╁寘銆傛敹鍒板悗杩愯锛?
```powershell
& 'C:\Users\27793\Documents\deep\cuda-venv\Scripts\python.exe' `
  '...\scripts\probe_formal_delivery_v1.py' `
  --project-root 'C:\Users\27793\Documents\project1\AI_Finance_Prototype' `
  --delivery-root '...\sources\incumbent_model_textcu_<delivery_version>' `
  --output-root '...\audits\formal_delivery_probe_v1'
```

鎺㈡祴閫氳繃鍙唬琛ㄥ彲浠ヨ繘鍏ユ繁搴﹀璁★紝涓嶄唬琛ㄦ巿鏉冭缁冩垨鏅嬬骇銆傛繁搴﹀璁￠€氳繃鍚庯紝鎵嶇敓鎴愪竴娆℃€ф巿鏉冨苟鍚姩姝ｅ紡 H4/T2 鍩虹嚎銆?

