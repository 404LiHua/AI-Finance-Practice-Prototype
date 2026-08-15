from __future__ import annotations

"""Metric-blind WP25 anchored-residual training."""

import argparse
import hashlib
import importlib.util
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FOLDS = tuple(f"REV2_RO_{i:02d}" for i in range(1, 7)); BAN = ("fresh", "screening", "final", "sealed_holdout")
RG3_DAILY = ["momentum_20d", "momentum_60d", "momentum_120d", "realized_volatility_20d", "realized_volatility_60d", "downside_volatility_60d", "current_drawdown_60d", "rsi_14", "macd_scaled", "bollinger_position_20", "amihud_20d", "zero_volume_fraction_20d", "volume_ratio_20d_60d", "intraday_range_mean_20d"]
RG3_STRUCTURAL = ["log_market_cap_total", "log_market_cap_float", "float_share_ratio", "listing_age_weeks", "is_special_treatment", "is_suspended", "is_delisted_asof", "is_suspended_listing_asof", "history_weeks_scaled", "market_cap_small", "market_cap_medium", "market_cap_large", "structural_state_missing"]
RG2_FEATURES = ["capital_event_this_week", "capital_event_increase_flag", "capital_event_decrease_flag", "log_total_shares_change_at_event", "log_tradable_shares_change_at_event", "tradable_share_ratio_change_at_event", "capital_event_age_260_scaled", "capital_history_missing_flag", "market_tradable_fraction", "market_eligible_fraction", "market_small_cap_fraction", "industry_tradable_fraction", "industry_eligible_fraction", "log1p_industry_member_count", "graph_mean_absolute_change", "graph_intra_industry_weight_fraction", "graph_mean_nonself_out_degree_scaled", "graph_max_nonself_out_degree_scaled"]
FEATURES = [*RG3_DAILY, *RG3_STRUCTURAL, *RG2_FEATURES]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path, root: Path):
    sys.dont_write_bytecode = True
    if str(root) not in sys.path: sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class AnchorResidual(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.05), nn.Linear(64, 32), nn.SiLU())
        self.return_residual = nn.Linear(32, 1)
        self.logit_residual = nn.Linear(32, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x); return self.return_residual(h).squeeze(1), self.logit_residual(h)


def robust_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(values, axis=0); scale = np.nanpercentile(values, 75, axis=0) - np.nanpercentile(values, 25, axis=0)
    center[~np.isfinite(center)] = 0.0; scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return center.astype(float), scale.astype(float)


def transform(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((values - center) / scale, -8.0, 8.0).astype(np.float32)


def train_fold(x: np.ndarray, y_return: np.ndarray, y_label: np.ndarray, anchor_p: np.ndarray, anchor_score: np.ndarray, seed: int, device: torch.device) -> AnchorResidual:
    seed_everything(seed); model = AnchorResidual(x.shape[1]).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y_return.astype(np.float32)), torch.from_numpy(y_label.astype(np.int64)), torch.from_numpy(anchor_p.astype(np.float32)), torch.from_numpy(anchor_score.astype(np.float32)))
    loader = DataLoader(data, batch_size=8192, shuffle=True, num_workers=0, pin_memory=True, drop_last=False); model.train()
    for _ in range(40):
        for xb, yb, lb, pb, sb in loader:
            xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True); lb=lb.to(device,non_blocking=True); pb=pb.to(device,non_blocking=True); sb=sb.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True); raw_return, delta = model(xb)
            predicted_return = sb + 0.10 * torch.tanh(raw_return)
            log_probability = torch.log(pb.clamp_min(1e-6)) + 0.20 * delta
            probability = torch.softmax(log_probability, dim=1)
            loss = torch.nn.functional.smooth_l1_loss(predicted_return, yb, beta=0.05) + torch.nn.functional.nll_loss(torch.log(probability.clamp_min(1e-7)), lb) + 0.05 * delta.square().mean()
            if not torch.isfinite(loss): raise RuntimeError("WP25 non-finite loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    return model


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--source-root", type=Path, required=True); p.add_argument("--rg3-input", type=Path, required=True); p.add_argument("--rg2-input", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True); a=p.parse_args()
    source=a.source_root.resolve(); rg3_path=a.rg3_input.resolve(); rg2_path=a.rg2_input.resolve(); protocol_path=a.protocol.resolve(); out=a.output_root.resolve()
    if out.exists(): raise RuntimeError(f"refusing to overwrite output: {out}")
    if any(token in str(path).lower() for path in (source,rg3_path,rg2_path,protocol_path,out) for token in BAN): raise RuntimeError("prohibited path token")
    protocol=json.loads(protocol_path.read_text(encoding="utf-8")); expected=protocol["immutable_inputs"]
    if protocol.get("status") != "FROZEN_BEFORE_CANDIDATE_TRAINING_OR_METRIC_READ": raise RuntimeError("WP25 protocol not frozen")
    samples_path=source/"data/rg1_4_materialized/samples.csv.gz"; split_path=source/"governance/rev7_1_freeze/SPLIT_PURGE_EMBARGO_AND_SAMPLE_KEY_CONTRACT.json"
    for label,path,digest in (("samples",samples_path,expected["samples_sha256"]),("rg3",rg3_path,expected["rg3_features_sha256"]),("rg2",rg2_path,expected["rg2_state_features_sha256"]),("split",split_path,expected["split_contract_sha256"])):
        if not path.is_file() or sha256(path)!=digest: raise RuntimeError(f"{label} hash mismatch")
    if not torch.cuda.is_available(): raise RuntimeError("WP25 requires CUDA")
    torch.set_num_threads(2); torch.set_num_interop_threads(1); device=torch.device("cuda")
    target_module=load_module("wp25_targets",source/"src/rev8_targets.py",source); ordinal_module=load_module("wp25_ordinal",source/"src/rg2_calibrated_ordinal.py",source)
    samples=pd.read_csv(samples_path,usecols=["fold_id","split_role","trade_date","stock_code","sample_key_sha256","target_return_h4","target_valid"],dtype={"fold_id":str,"split_role":str,"stock_code":str,"sample_key_sha256":str}); samples.trade_date=pd.to_datetime(samples.trade_date,errors="raise").dt.normalize()
    canonical=samples[["trade_date","stock_code","target_return_h4","target_valid","realized_volatility_8w"]] if "realized_volatility_8w" in samples.columns else None
    # Re-read the one auxiliary target column only after the fixed input hash has been checked.
    samples_full=pd.read_csv(samples_path,usecols=["fold_id","split_role","trade_date","stock_code","sample_key_sha256","target_return_h4","target_valid","realized_volatility_8w"],dtype={"fold_id":str,"split_role":str,"stock_code":str,"sample_key_sha256":str}); samples_full.trade_date=pd.to_datetime(samples_full.trade_date,errors="raise").dt.normalize()
    canonical=samples_full[["trade_date","stock_code","target_return_h4","target_valid","realized_volatility_8w"]].drop_duplicates(["trade_date","stock_code"],keep="first"); variants=target_module.build_target_variants(canonical); labels=variants[["trade_date","stock_code","T2_return","T2_valid","T2_label"]].rename(columns={"T2_return":"relative_return","T2_valid":"derived_valid","T2_label":"ordinal_target"})
    samples=samples.merge(labels,on=["trade_date","stock_code"],how="left",validate="many_to_one")
    rg3=pd.read_csv(rg3_path,dtype={"stock_code":str}); rg3.trade_date=pd.to_datetime(rg3.trade_date,errors="raise").dt.normalize(); rg2=pd.read_csv(rg2_path,dtype={"sample_key_sha256":str}); rg2.trade_date=pd.to_datetime(rg2.trade_date,errors="raise").dt.normalize()
    joined=samples.merge(rg3[["trade_date","stock_code",*RG3_DAILY,*RG3_STRUCTURAL]],on=["trade_date","stock_code"],how="left",validate="many_to_one").merge(rg2[["sample_key_sha256",*RG2_FEATURES]],on="sample_key_sha256",how="left",validate="one_to_one")
    if joined[FEATURES].isna().any().any() or not np.isfinite(joined[FEATURES].to_numpy(float)).all(): raise RuntimeError("WP25 feature join incomplete")
    out.mkdir(parents=True); (out/"predictions_sealed").mkdir(); (out/"checkpoints").mkdir(); (out/"normalization").mkdir(); (out/"anchors").mkdir()
    prediction_hashes={}; checkpoint_hashes={}
    for index,fold in enumerate(FOLDS,1):
        train=joined[(joined.fold_id==fold)&(joined.split_role=="TRAIN")&joined.derived_valid.astype(bool)].copy(); val=joined[(joined.fold_id==fold)&(joined.split_role=="VALIDATION")].copy(); train_x_raw=train[FEATURES].to_numpy(float); val_x_raw=val[FEATURES].to_numpy(float); center,scale=robust_fit(train_x_raw); train_x=transform(train_x_raw,center,scale); val_x=transform(val_x_raw,center,scale)
        anchor=ordinal_module.fit_proportional_odds(train[RG3_DAILY].to_numpy(float),train.ordinal_target.astype(int).to_numpy(),l2=0.001,max_iter=200); anchor_train=anchor.predict_proba(train[RG3_DAILY].to_numpy(float)); anchor_val=anchor.predict_proba(val[RG3_DAILY].to_numpy(float)); train_score=0.10*(anchor_train[:,2]-anchor_train[:,0]); val_score=0.10*(anchor_val[:,2]-anchor_val[:,0]); model=train_fold(train_x,train.relative_return.to_numpy(float),train.ordinal_target.astype(int).to_numpy(),anchor_train,train_score,2026081525+index,device); model.eval()
        with torch.no_grad():
            raw,delta=model(torch.from_numpy(val_x).to(device)); predicted=val_score+0.10*np.tanh(raw.cpu().numpy()); prob=torch.softmax(torch.log(torch.from_numpy(anchor_val.astype(np.float32)).to(device).clamp_min(1e-6))+0.20*delta,dim=1).cpu().numpy()
        if not np.isfinite(prob).all() or not np.allclose(prob.sum(1),1.0,atol=1e-6): raise RuntimeError(f"WP25 probability failure {fold}")
        pred=val[["fold_id","split_role","trade_date","stock_code","sample_key_sha256"]].copy(); pred["candidate_id"]=protocol["model"]["id"]; pred["predicted_h4_relative"]=predicted; pred["predicted_h4_scale"]=0.10; pred["prob_down"]=prob[:,0]; pred["prob_neutral"]=prob[:,1]; pred["prob_up"]=prob[:,2]; pred["predicted_ordinal"]=prob.argmax(1).astype(np.int8); pred=pred.sort_values("sample_key_sha256",kind="mergesort"); pred_path=out/"predictions_sealed"/f"{fold}_WP25.parquet"; pred.to_parquet(pred_path,index=False,engine="pyarrow",compression="zstd")
        norm_path=out/"normalization"/f"{fold}.json"; norm_path.write_text(json.dumps({"fold_id":fold,"features":FEATURES,"median":center.tolist(),"iqr":scale.tolist()},ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); checkpoint=out/"checkpoints"/f"{fold}.pt"; torch.save({"candidate_id":protocol["model"]["id"],"fold_id":fold,"state_dict":model.state_dict(),"median":center,"iqr":scale,"anchor_beta":anchor.beta,"anchor_thresholds":anchor.thresholds},checkpoint); anchor_path=out/"anchors"/f"{fold}.json"; anchor_path.write_text(json.dumps({"fold_id":fold,"beta":anchor.beta.tolist(),"thresholds":list(anchor.thresholds),"scaler_median":anchor.scaler.median.tolist(),"scaler_scale":anchor.scaler.scale.tolist()},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        prediction_hashes[str(pred_path.relative_to(out)).replace("\\","/")]=sha256(pred_path); checkpoint_hashes[str(checkpoint.relative_to(out)).replace("\\","/")]=sha256(checkpoint); del model; torch.cuda.empty_cache()
    manifest={"node_id":"WP25_ANCHORED_RESIDUAL_PREDICTION_SEAL","status":"SEALED_PENDING_INDEPENDENT_METRIC_READ","candidate_id":protocol["model"]["id"],"protocol_sha256":sha256(protocol_path),"input_sha256":{"samples":sha256(samples_path),"rg3":sha256(rg3_path),"rg2":sha256(rg2_path),"split":sha256(split_path)},"prediction_sha256":prediction_hashes,"checkpoint_sha256":checkpoint_hashes,"folds":list(FOLDS),"metrics_read":False,"targets_written":False,"fresh_payloads_opened":False,"production_replacement_allowed":False,"gpu_used":True}; (out/"PREDICTION_SEAL_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); (out/"EXECUTION_RECEIPT.json").write_text(json.dumps({"node_id":"WP25_ANCHORED_RESIDUAL_GPU_V1","status":"PASS_PREDICTIONS_SEALED_PENDING_INDEPENDENT_METRIC_READ","candidate_id":protocol["model"]["id"],"metrics_read":False,"fresh_payloads_opened":False,"production_replacement_allowed":False,"created_at_utc":datetime.now(timezone.utc).isoformat()},ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps({"status":manifest["status"],"output_root":str(out)},ensure_ascii=False))


if __name__ == "__main__": main()
