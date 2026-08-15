import argparse,json,pathlib,hashlib
import numpy as np
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--protocol',required=True); ap.add_argument('--output',required=True); ap.add_argument('--receipt',required=True); a=ap.parse_args()
 z=np.load(a.input,allow_pickle=True); x=z['x'].astype(np.float32); dates=z['origin_dates']; out=pathlib.Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
 if len(dates)<2 or not all(str(dates[i])<str(dates[i+1]) for i in range(len(dates)-1)): raise SystemExit('origin dates not strictly increasing')
 y=np.full_like(x,np.nan,dtype=np.float32); y[1:]=x[:-1]
 np.savez_compressed(out,x=y,y=z['y'],origin_dates=dates,stock_codes=z['stock_codes'],feature_names=z['feature_names'],source_input_sha256=sha(a.input))
 r={'protocol_id':json.loads(pathlib.Path(a.protocol).read_text())['protocol_id'],'status':'COMPLETE_LAGGED_PROXY','input_sha256':sha(a.input),'output_sha256':sha(out),'shape':list(y.shape),'first_origin_all_missing':bool(np.isnan(y[0]).all()),'lag_exact_match':bool(np.array_equal(y[1:],x[:-1],equal_nan=True)),'promotion_allowed':False}
 pathlib.Path(a.receipt).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(a.receipt).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__': main()


