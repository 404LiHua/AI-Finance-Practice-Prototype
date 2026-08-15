from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

FOLDS = tuple(f"REV2_RO_{i:02d}" for i in range(1, 7))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("wp25_runner_for_cpu_audit", path)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load WP25 runner")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--source-root", type=Path, required=True); p.add_argument("--rg3-input", type=Path, required=True); p.add_argument("--rg2-input", type=Path, required=True); p.add_argument("--candidate-root", type=Path, required=True); p.add_argument("--runner", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True); a=p.parse_args()
    source=a.source_root.resolve(); rg3_path=a.rg3_input.resolve(); rg2_path=a.rg2_input.resolve(); candidate=a.candidate_root.resolve(); runner_path=a.runner.resolve(); out=a.output_root.resolve()
    if out.exists(): raise RuntimeError(f"refusing to overwrite output: {out}")
    runner=load_runner(runner_path); features=runner.FEATURES; daily=runner.RG3_DAILY; structural=runner.RG3_STRUCTURAL; state_features=runner.RG2_FEATURES; Model=runner.AnchorResidual
    samples_path=source/"data/rg1_4_materialized/samples.csv.gz"; samples=pd.read_csv(samples_path,usecols=["fold_id","split_role","trade_date","stock_code","sample_key_sha256"],dtype={"fold_id":str,"split_role":str,"stock_code":str,"sample_key_sha256":str}); samples.trade_date=pd.to_datetime(samples.trade_date,errors="raise").dt.normalize()
    rg3=pd.read_csv(rg3_path,dtype={"stock_code":str}); rg3.trade_date=pd.to_datetime(rg3.trade_date,errors="raise").dt.normalize(); rg2=pd.read_csv(rg2_path,dtype={"sample_key_sha256":str}); joined=samples.merge(rg3[["trade_date","stock_code",*daily,*structural]],on=["trade_date","stock_code"],how="left",validate="many_to_one").merge(rg2[["sample_key_sha256",*state_features]],on="sample_key_sha256",how="left",validate="one_to_one")
    if joined[features].isna().any().any(): raise RuntimeError("CPU replay feature join incomplete")
    out.mkdir(parents=True); rows=[]
    for fold in FOLDS:
        ckpt_path=candidate/"checkpoints"/f"{fold}.pt"; ckpt=torch.load(ckpt_path,map_location="cpu",weights_only=False); model=Model(len(features)); model.load_state_dict(ckpt["state_dict"]); model.eval()
        anchor=json.loads((candidate/"anchors"/f"{fold}.json").read_text(encoding="utf-8")); val=joined[(joined.fold_id==fold)&(joined.split_role=="VALIDATION")].copy(); x=np.clip((val[features].to_numpy(float)-np.asarray(ckpt["median"])) / np.asarray(ckpt["iqr"]),-8,8).astype(np.float32); ax=np.clip((val[daily].to_numpy(float)-np.asarray(anchor["scaler_median"])) / np.asarray(anchor["scaler_scale"]),-8,8); eta=ax@np.asarray(anchor["beta"]); t=np.asarray(anchor["thresholds"]); c0=1/(1+np.exp(-(t[0]-eta))); c1=1/(1+np.exp(-(t[1]-eta))); ap=np.column_stack([c0,c1-c0,1-c1]); score=0.10*(ap[:,2]-ap[:,0])
        with torch.no_grad(): raw,delta=model(torch.from_numpy(x)); predicted=score+0.10*np.tanh(raw.numpy()); probability=torch.softmax(torch.log(torch.from_numpy(ap.astype(np.float32)).clamp_min(1e-6))+0.20*delta,dim=1).numpy()
        sealed=pd.read_parquet(candidate/"predictions_sealed"/f"{fold}_WP25.parquet"); sealed=sealed.sort_values("sample_key_sha256").reset_index(drop=True); order=np.argsort(val.sample_key_sha256.to_numpy()); predicted=predicted[order]; probability=probability[order]
        max_return=float(np.max(np.abs(predicted-sealed.predicted_h4_relative.to_numpy(float)))); max_probability=float(np.max(np.abs(probability-sealed[["prob_down","prob_neutral","prob_up"]].to_numpy(float)))); rows.append({"fold_id":fold,"rows":int(len(val)),"max_abs_return_difference":max_return,"max_abs_probability_difference":max_probability,"pass":max_return<=1e-5 and max_probability<=1e-5})
    decision={"node_id":"WP25_CPU_REPLAY_AUDIT_V1","status":"PASS_CPU_CHECKPOINT_REPLAY" if all(row["pass"] for row in rows) else "FAIL_CPU_CHECKPOINT_REPLAY","folds":rows,"candidate_manifest_sha256":sha256(candidate/"PREDICTION_SEAL_MANIFEST.json"),"gpu_used":False,"targets_read":False,"fresh_payloads_opened":False,"production_registry_modified":False,"created_at_utc":datetime.now(timezone.utc).isoformat()}; (out/"WP25_CPU_REPLAY_DECISION.json").write_text(json.dumps(decision,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(decision,ensure_ascii=False))


if __name__ == "__main__": main()
