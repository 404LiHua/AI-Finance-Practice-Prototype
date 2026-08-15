from __future__ import annotations

"""Fit the passed WP25 design on the complete development window only."""

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def load(name: str, path: Path, root: Path):
    sys.dont_write_bytecode = True
    if str(root) not in sys.path: sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--source-root",type=Path,required=True); p.add_argument("--rg3-input",type=Path,required=True); p.add_argument("--rg2-input",type=Path,required=True); p.add_argument("--protocol",type=Path,required=True); p.add_argument("--runner",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args()
    source=a.source_root.resolve(); rg3_path=a.rg3_input.resolve(); rg2_path=a.rg2_input.resolve(); protocol_path=a.protocol.resolve(); runner_path=a.runner.resolve(); out=a.output_root.resolve()
    if out.exists(): raise RuntimeError(f"refusing to overwrite output: {out}")
    protocol=json.loads(protocol_path.read_text(encoding="utf-8")); expected=protocol["immutable_inputs"]
    samples_path=source/"data/rg1_4_materialized/samples.csv.gz"; split_path=source/"governance/rev7_1_freeze/SPLIT_PURGE_EMBARGO_AND_SAMPLE_KEY_CONTRACT.json"
    for label,path,digest in (("samples",samples_path,expected["samples_sha256"]),("rg3",rg3_path,expected["rg3_features_sha256"]),("rg2",rg2_path,expected["rg2_state_features_sha256"]),("split",split_path,expected["split_contract_sha256"])):
        if not path.is_file() or sha256(path)!=digest: raise RuntimeError(f"{label} hash mismatch")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    runner=load("wp25_runner_full",runner_path,runner_path.parent.parent); target=load("wp25_target_full",source/"src/rev8_targets.py",source); ordinal=load("wp25_ordinal_full",source/"src/rg2_calibrated_ordinal.py",source)
    features=runner.FEATURES; daily=runner.RG3_DAILY; structural=runner.RG3_STRUCTURAL; state_features=runner.RG2_FEATURES
    samples=pd.read_csv(samples_path,usecols=["fold_id","split_role","trade_date","stock_code","sample_key_sha256","target_return_h4","target_valid","realized_volatility_8w"],dtype={"fold_id":str,"split_role":str,"stock_code":str,"sample_key_sha256":str}); samples.trade_date=pd.to_datetime(samples.trade_date,errors="raise").dt.normalize()
    canonical=samples[["trade_date","stock_code","target_return_h4","target_valid","realized_volatility_8w"]].drop_duplicates(["trade_date","stock_code"],keep="first"); variants=target.build_target_variants(canonical); labels=variants[["trade_date","stock_code","T2_return","T2_valid","T2_label"]].rename(columns={"T2_return":"relative_return","T2_valid":"derived_valid","T2_label":"ordinal_target"})
    identity=samples.sort_values(["trade_date","stock_code","fold_id"]).drop_duplicates(["trade_date","stock_code"],keep="first").merge(labels,on=["trade_date","stock_code"],how="left",validate="one_to_one")
    rg3=pd.read_csv(rg3_path,dtype={"stock_code":str}); rg3.trade_date=pd.to_datetime(rg3.trade_date,errors="raise").dt.normalize(); rg2=pd.read_csv(rg2_path,dtype={"sample_key_sha256":str}); joined=identity.merge(rg3[["trade_date","stock_code",*daily,*structural]],on=["trade_date","stock_code"],how="left",validate="one_to_one").merge(rg2[["sample_key_sha256",*state_features]],on="sample_key_sha256",how="left",validate="one_to_one")
    train=joined[joined.derived_valid.astype(bool)].copy(); x_raw=train[features].to_numpy(float); center,scale=runner.robust_fit(x_raw); x=runner.transform(x_raw,center,scale); anchor=ordinal.fit_proportional_odds(train[daily].to_numpy(float),train.ordinal_target.astype(int).to_numpy(),l2=0.001,max_iter=200); ap=anchor.predict_proba(train[daily].to_numpy(float)); score=0.10*(ap[:,2]-ap[:,0]); model=runner.train_fold(x,train.relative_return.to_numpy(float),train.ordinal_target.astype(int).to_numpy(),ap,score,2026082599,torch.device("cuda")); model.eval()
    out.mkdir(parents=True); model_path=out/"WP25_ANCHORED_RESIDUAL_FULL_DEVELOPMENT.pt"; torch.save({"candidate_id":protocol["model"]["id"],"state_dict":model.state_dict(),"feature_names":features,"median":center,"iqr":scale,"anchor_beta":anchor.beta,"anchor_thresholds":anchor.thresholds},model_path); anchor_path=out/"ANCHOR.json"; anchor_path.write_text(json.dumps({"beta":anchor.beta.tolist(),"thresholds":list(anchor.thresholds),"scaler_median":anchor.scaler.median.tolist(),"scaler_scale":anchor.scaler.scale.tolist()},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    receipt={"artifact_id":"WP25_ANCHORED_RESIDUAL_FULL_DEVELOPMENT_V1","status":"NON_PRODUCTION_SHADOW_ARTIFACT","candidate_id":protocol["model"]["id"],"training_rows":int(len(train)),"development_origins":int(train.trade_date.nunique()),"model_sha256":sha256(model_path),"protocol_sha256":sha256(protocol_path),"input_sha256":{"samples":sha256(samples_path),"rg3":sha256(rg3_path),"rg2":sha256(rg2_path),"split":sha256(split_path)},"device":torch.cuda.get_device_name(0),"future_labels_read":False,"production_replacement_allowed":False,"created_at_utc":datetime.now(timezone.utc).isoformat()}; (out/"WP25_FULL_DEVELOPMENT_ARTIFACT.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(receipt,ensure_ascii=False))


if __name__ == "__main__": main()
