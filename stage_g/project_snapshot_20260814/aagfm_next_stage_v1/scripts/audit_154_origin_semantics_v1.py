import argparse,json,pathlib,hashlib
import numpy as np,pandas as pd
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--tensor',required=True); ap.add_argument('--technical',required=True); ap.add_argument('--h4',required=True); ap.add_argument('--t2',required=True); ap.add_argument('--output-root',required=True); a=ap.parse_args()
 out=pathlib.Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
 z=np.load(a.tensor,allow_pickle=True); x=z['x']; names=[str(v) for v in z['feature_names']]
 tech=pd.read_parquet(a.technical); h4=pd.read_parquet(a.h4); t2=pd.read_parquet(a.t2)
 eq=int((tech.source_trade_date==tech.origin_date).sum()); future=int((tech.source_trade_date>tech.origin_date).sum())
 stats=[]
 for i,n in enumerate(names):
  v=x[:,:,i].astype(float); finite=np.isfinite(v); q=v[finite]; stats.append({'feature':n,'finite_rate':float(finite.mean()),'nan_count':int((~finite).sum()),'mean':float(np.mean(q)) if len(q) else None,'std':float(np.std(q)) if len(q) else None,'q01':float(np.quantile(q,.01)) if len(q) else None,'q99':float(np.quantile(q,.99)) if len(q) else None})
 result={'status':'AUDIT_COMPLETE_WITH_PIT_BOUNDARY_FLAG' if eq or future else 'AUDIT_COMPLETE','tensor_shape':list(x.shape),'tensor_sha256':sha(a.tensor),'technical_rows':len(tech),'h4_rows':len(h4),'t2_rows':len(t2),'technical_source_equal_origin':eq,'technical_source_after_origin':future,'strict_pit_rule_pass':bool(eq==0 and future==0),'h4_realized_after_origin':bool((h4.label_realized_at>h4.origin_date).all()),'t2_realized_after_origin':bool((t2.label_realized_at>t2.origin_date).all()),'feature_stats':stats}
 (out/'SEMANTICS_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({k:result[k] for k in ['status','technical_source_equal_origin','technical_source_after_origin','strict_pit_rule_pass']},ensure_ascii=False))
if __name__=='__main__': main()


