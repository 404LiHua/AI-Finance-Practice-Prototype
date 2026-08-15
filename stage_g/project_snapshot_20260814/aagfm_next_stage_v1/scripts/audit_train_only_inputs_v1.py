#!/usr/bin/env python
import argparse, csv, hashlib, json, pathlib, sys
import numpy as np

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',required=True); ap.add_argument('--contract',required=True); ap.add_argument('--output-root',required=True); a=ap.parse_args()
    root=pathlib.Path(a.project_root); out=pathlib.Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    c=json.loads(pathlib.Path(a.contract).read_text(encoding='utf-8')); findings=[]; manifest=[]
    assets=[]
    for spec in c['accepted_train_only_assets']:
        p=root / pathlib.Path(spec['relative_path'].replace('/', '\\')); rec={'id':spec['id'],'path':str(p),'exists':p.exists()}
        if not p.exists(): findings.append(f"MISSING:{spec['id']}"); assets.append(rec); continue
        rec['sha256']=sha256(p); z=np.load(p,allow_pickle=True); rec['keys']=list(z.files)
        rec['shapes']={k:list(z[k].shape) for k in z.files}; rec['dtypes']={k:str(z[k].dtype) for k in z.files}
        rec['nan_counts']={k:int(np.isnan(z[k]).sum()) for k in z.files if np.issubdtype(z[k].dtype,np.floating)}
        rec['observed_false']=int((~z['observed']).sum()) if 'observed' in z else None
        if rec['keys'] != spec['expected_keys']: findings.append(f"KEY_MISMATCH:{spec['id']}")
        if spec['id']=='sequence_cube' and rec['shapes'].get('dense_history_source') != [299,5796,14]: findings.append('SHAPE_MISMATCH:sequence_cube')
        if spec['id']=='multiview_cube' and rec['shapes'].get('dense') != [299,5796,49]: findings.append('SHAPE_MISMATCH:multiview_cube')
        if len(z['origin_index']) != len(np.unique(z['origin_index'])): findings.append(f"DUPLICATE_ORIGINS:{spec['id']}")
        assets.append(rec); manifest.append(rec)
    formal=c['expected_formal_panel']; findings += [f"FORMAL_PANEL_NOT_PRESENT:origins={formal['origins']},stocks={formal['stocks']}"]
    decision='TRAIN_ONLY_ASSETS_ACCEPTED_WITH_FORMAL_PANEL_BLOCK' if assets and all(x.get('exists') for x in assets) else 'INPUT_ASSET_AUDIT_FAIL'
    (out/'INPUT_AUDIT.json').write_text(json.dumps({'contract_id':c['contract_id'],'decision':decision,'findings':findings,'assets':assets},ensure_ascii=False,indent=2),encoding='utf-8')
    with open(out/'INPUT_MANIFEST_SHA256.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['id','path','sha256']); w.writeheader(); [w.writerow({k:x.get(k,'') for k in ['id','path','sha256']}) for x in manifest]
    (out/'DECISION.json').write_text(json.dumps({'status':decision,'formal_training_authorized':False,'joint_model_training_authorized':False,'findings':findings},ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'NEGATIVE_EVIDENCE.md').write_text('# Negative evidence\n\n- Formal 337×300 panel and original incumbent delivery are not present in the accepted train-only cubes.\n- No joint model training is authorized by this audit.\n',encoding='utf-8')
    print(json.dumps({'decision':decision,'findings':findings,'output_root':str(out)},ensure_ascii=False))
if __name__=='__main__': main()


