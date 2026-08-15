import argparse, json, pathlib, hashlib
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cube',required=True); ap.add_argument('--output-root',required=True); a=ap.parse_args()
    out=pathlib.Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    z=np.load(a.cube,allow_pickle=True); rr=z['relative_return'].astype(np.float64); obs=z['observed'].astype(bool)
    vals=rr[obs]; finite=np.isfinite(vals); vals=vals[finite]
    # Descriptive only: no split, no tuning, no promotion.
    result={'status':'DIAGNOSTIC_ONLY_NOT_FORMAL_BASELINE','cube_sha256':hashlib.sha256(pathlib.Path(a.cube).read_bytes()).hexdigest(),
      'shape':list(rr.shape),'observed_count':int(obs.sum()),'finite_count':int(finite.sum()),
      'return_mean':float(np.mean(vals)),'return_std':float(np.std(vals)),'return_median':float(np.median(vals)),
      'return_q01':float(np.quantile(vals,.01)),'return_q99':float(np.quantile(vals,.99)),
      'up_rate':float(np.mean(vals>0.01)),'down_rate':float(np.mean(vals< -0.01)),'neutral_rate':float(np.mean((vals>=-.01)&(vals<=.01))),
      'formal_panel_used':False,'test_or_fresh_read':False,'promotion_allowed':False}
    (out/'DIAGNOSTIC_BASELINE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'EXECUTION_RECEIPT.json').write_text(json.dumps({'status':'EXECUTED_READ_ONLY','inputs':[a.cube],'outputs':['DIAGNOSTIC_BASELINE.json']},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()


