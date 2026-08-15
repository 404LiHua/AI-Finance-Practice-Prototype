import argparse,json,pathlib
import numpy as np
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--original',required=True); ap.add_argument('--lagged',required=True); ap.add_argument('--receipt',required=True); ap.add_argument('--output-root',required=True); a=ap.parse_args()
 o=np.load(a.original,allow_pickle=True); l=np.load(a.lagged,allow_pickle=True); findings=[]
 for k in ['y','origin_dates','stock_codes','feature_names']:
  if not np.array_equal(o[k],l[k]): findings.append('METADATA_MISMATCH:'+k)
 if not np.isnan(l['x'][0]).all(): findings.append('FIRST_ORIGIN_NOT_MISSING')
 if not np.array_equal(l['x'][1:],o['x'][:-1],equal_nan=True): findings.append('SHIFT_MISMATCH')
 dates=l['origin_dates']; increasing=all(str(dates[i])<str(dates[i+1]) for i in range(len(dates)-1))
 if not increasing: findings.append('DATES_NOT_INCREASING')
 status='PASS_STRICT_LAG_PROXY' if not findings else 'FAIL_STRICT_LAG_PROXY'
 r={'status':status,'findings':findings,'strict_lag_proxy_only':True,'source_timestamp_proof':False,'promotion_allowed':False}
 out=pathlib.Path(a.output_root); out.mkdir(parents=True,exist_ok=True); (out/'DECISION.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__': main()


