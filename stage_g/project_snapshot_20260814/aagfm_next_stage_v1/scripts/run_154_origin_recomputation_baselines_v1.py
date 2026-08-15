import argparse,json,pathlib,hashlib,os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error,matthews_corrcoef,brier_score_loss
from scipy.stats import spearmanr
from xgboost import XGBRegressor

def probs_po(x,lo,hi):
 p0=lo.predict_proba(x)[:,1]; p_le1=hi.predict_proba(x)[:,1]
 p1=np.maximum(p_le1-p0,0); p2=np.maximum(1-p_le1,0); p=np.c_[p0,p1,p2]; return p/np.maximum(p.sum(1,keepdims=True),1e-12)
def metrics(y,p,cls,pr):
 valid=np.isfinite(y)&np.isfinite(p); y=y[valid]; p=p[valid]; cls=cls[valid]; pr=pr[valid]
 ic=spearmanr(y,p).statistic if len(y)>2 else np.nan; predc=pr.argmax(1)
 return {'n':int(len(y)),'mae':float(mean_absolute_error(y,p)),'rmse':float(mean_squared_error(y,p)**.5),'rank_ic':float(ic),'t2_mcc':float(matthews_corrcoef(cls,predc)),'t2_brier':float(np.mean(np.sum((pr-np.eye(3)[cls])**2,axis=1)))}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--tensor',required=True); ap.add_argument('--labels',required=True); ap.add_argument('--protocol',required=True); ap.add_argument('--output-root',required=True); a=ap.parse_args()
 os.environ.setdefault('OMP_NUM_THREADS','4'); os.environ.setdefault('MKL_NUM_THREADS','4')
 out=pathlib.Path(a.output_root); out.mkdir(parents=True,exist_ok=True); cfg=json.loads(pathlib.Path(a.protocol).read_text())
 z=np.load(a.tensor,allow_pickle=True); x=z['x'].astype(np.float32); dates=z['origin_dates']; stocks=z['stock_codes']
 import pandas as pd
 lab=pd.read_parquet(a.labels); lab['origin_date']=lab.origin_date.dt.strftime('%Y-%m-%d'); mp={(r.origin_date,r.stock_code):(r.relative_return,int(r.t2r_class)) for r in lab.itertuples()}
 y=np.full(x.shape[:2],np.nan); c=np.full(x.shape[:2],-1,dtype=int)
 for i,d in enumerate(dates):
  for j,s in enumerate(stocks):
   if (str(d),str(s)) in mp: y[i,j],c[i,j]=mp[(str(d),str(s))]
 folds=[]; all_y=[]; all_pred=[]; all_c=[]; all_pr=[]
 for start in cfg['fold_validation_starts']:
  tr_end=start-cfg['purge_weeks']; va_end=min(start+cfg['validation_length'],len(dates)); tx=x[:tr_end].reshape(-1,x.shape[-1]); ty=y[:tr_end].ravel(); tc=c[:tr_end].ravel(); vx=x[start:va_end].reshape(-1,x.shape[-1]); vy=y[start:va_end].ravel(); vc=c[start:va_end].ravel()
  ok=np.isfinite(ty)&(tc>=0); vok=np.isfinite(vy)&(vc>=0); tx,ty,tc=tx[ok],ty[ok],tc[ok]; vx,vy,vc=vx[vok],vy[vok],vc[vok]
  ridge=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=10.0)); ridge.fit(tx,ty); rp=ridge.predict(vx)
  xgb=XGBRegressor(n_estimators=250,max_depth=3,learning_rate=.03,subsample=.8,colsample_bytree=.8,reg_lambda=10,n_jobs=4,tree_method='hist',random_state=20260813); xgb.fit(tx,ty,verbose=False); xp=xgb.predict(vx)
  lo=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.2,max_iter=500,n_jobs=4)); hi=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.2,max_iter=500,n_jobs=4)); lo.fit(tx,(tc<=0).astype(int)); hi.fit(tx,(tc<=1).astype(int)); pp=probs_po(vx,lo,hi)
  naive=np.zeros_like(vy); basepr=np.tile(np.bincount(tc,minlength=3)/len(tc),(len(vy),1))
  folds.append({'start':start,'train_origins':tr_end,'validation_origins':va_end-start,'naive':metrics(vy,naive,vc,basepr),'ridge':metrics(vy,rp,vc,pp),'xgboost':metrics(vy,xp,vc,pp)})
  all_y.append(vy); all_pred.append(np.c_[rp,xp]); all_c.append(vc); all_pr.append(pp)
 Y=np.concatenate(all_y); P=np.concatenate(all_pred); C=np.concatenate(all_c); PR=np.concatenate(all_pr)
 summary={'status':'COMPLETE_DEVELOPMENT_ONLY','identity':cfg['identity'],'folds':folds,'pooled':{'ridge':metrics(Y,P[:,0],C,PR),'xgboost':metrics(Y,P[:,1],C,PR)},'test_fresh_read':False,'promotion_allowed':False,'gpu_used':False,'cpu_threads':4}
 (out/'BASELINE_RESULTS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary['pooled'],ensure_ascii=False))
if __name__=='__main__': main()


