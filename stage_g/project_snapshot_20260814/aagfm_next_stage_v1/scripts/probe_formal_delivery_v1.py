import argparse,csv,hashlib,json,pathlib

REQUIRED=['M01','M02','M03','M04','M05','M06','M07','M08','M09','M10','M11','M12','M13','M14']
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--project-root',required=True); ap.add_argument('--delivery-root',required=True); ap.add_argument('--output-root',required=True); a=ap.parse_args()
 root=pathlib.Path(a.project_root); d=pathlib.Path(a.delivery_root); o=pathlib.Path(a.output_root); o.mkdir(parents=True,exist_ok=True)
 gaps=[]; checks=[]
 # Formal universe is independently bindable.
 candidates=[root/'data'/'M5_FORMAL_STOCK_UNIVERSE_300.csv', root/'research_tracks'/'pit_information_incremental_v1'/'frozen_inputs'/'m5_frozen_tensor_300stocks_2023_2025_v3'/'M5_FORMAL_STOCK_UNIVERSE_300.csv']
 uni=next((p for p in candidates if p.exists()), candidates[0])
 checks.append({'id':'M05','path':str(uni),'exists':uni.exists(),'sha256':sha(uni) if uni.exists() else None})
 if not uni.exists(): gaps.append('M05 formal universe missing')
 # Supplier delivery hard gates.
 receipt=d/'DELIVERY_RECEIPT.json'; manifest=d/'SHA256_MANIFEST.csv'
 for item,p in [('M14_RECEIPT',receipt),('M14_MANIFEST',manifest)]:
  checks.append({'id':item,'path':str(p),'exists':p.exists()})
  if not p.exists(): gaps.append(f'{item} missing')
 if d.exists():
  for mid in REQUIRED:
   marker=list(d.rglob(f'*{mid}*'))
   if not marker: gaps.append(f'{mid} no identifiable artifact')
   checks.append({'id':mid,'candidate_count':len(marker)})
 else: gaps.append('delivery root missing')
 status='FORMAL_DELIVERY_READY_FOR_DEEP_AUDIT' if not gaps else 'WAITING_FOR_SUPPLIER_COMPLETE_DELIVERY'
 decision={'status':status,'formal_authorization_allowed':False,'joint_training_allowed':False,'gaps':gaps,'checks':checks,'delivery_root':str(d)}
 (o/'DELIVERY_PROBE_DECISION.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
 (o/'NEGATIVE_EVIDENCE.md').write_text('# Delivery probe negative evidence\n\n'+''.join(f'- {g}\n' for g in gaps) if gaps else '# No negative evidence; proceed to deep audit.\n',encoding='utf-8')
 print(json.dumps(decision,ensure_ascii=False))
if __name__=='__main__': main()


