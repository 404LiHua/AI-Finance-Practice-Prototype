from __future__ import annotations

"""Exercise the active T2 HTTP app over a loopback-only ephemeral listener."""

import argparse
import hashlib
import http.client
import importlib.util
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_server(package_root: Path):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    path = package_root / "apps" / "stock_predictor" / "server.py"
    spec = importlib.util.spec_from_file_location("wp12_local_server", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import local server")
    module = importlib.util.module_from_spec(spec); sys.modules["wp12_local_server"] = module
    spec.loader.exec_module(module)
    return module


def request(port: int, method: str, path: str, body: bytes | None = None) -> tuple[int, dict[str, Any], float]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    started = time.perf_counter()
    headers = {"Content-Type": "application/json; charset=utf-8"} if body is not None else {}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse(); content = response.read(); elapsed = time.perf_counter() - started
    connection.close()
    return int(response.status), json.loads(content.decode("utf-8")), elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--daily-root", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    package, registry_path, daily_root, policy_path, output = (args.package_root.resolve(), args.registry.resolve(), args.daily_root.resolve(), args.policy.resolve(), args.output_root.resolve())
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    policy = json.loads(policy_path.read_text(encoding="utf-8")); identity = policy["identity"]
    if policy.get("status") != "FROZEN_BEFORE_LOCAL_LOOPBACK_OPERATIONAL_AUDIT":
        raise RuntimeError("operational acceptance policy is not frozen")
    if sha256(registry_path) != identity["registry_sha256"]:
        raise RuntimeError("local registry identity mismatch")
    registry = json.loads(registry_path.read_text(encoding="utf-8")); active = registry["active_model"]
    model_path = Path(active["path"])
    source_paths = {"server_source_sha256": package / "apps" / "stock_predictor" / "server.py", "stock_service_source_sha256": package / "src" / "stock_prediction_service.py", "confirmed_model_source_sha256": package / "src" / "confirmed_safe_model.py"}
    if active["model_id"] != identity["active_model_id"] or active["sha256"] != identity["active_model_sha256"] or sha256(model_path) != active["sha256"]:
        raise RuntimeError("active production anchor identity mismatch")
    for field, path in source_paths.items():
        if sha256(path) != identity[field]:
            raise RuntimeError(f"local service source mismatch: {field}")
    if policy["runtime_policy"]["host"] != "127.0.0.1" or not daily_root.is_dir():
        raise RuntimeError("loopback/daily-source contract failure")
    os.environ["AAGFMNET_GPU_POLICY"] = "disabled"; os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
    server_module = load_server(package)
    from src.stock_prediction_service import StockPredictionService
    service = StockPredictionService(model_path, daily_root)
    server = server_module.create_server(service, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, name="local-t2-loopback-audit", daemon=True)
    thread.start()
    checks: dict[str, Any] = {}; failure: str | None = None
    try:
        port = int(server.server_port)
        status, health, elapsed = request(port, "GET", "/api/health")
        checks["health"] = {"status": status, "elapsed_seconds": elapsed, "payload": health}
        if status != 200 or health.get("status") != "ok" or health.get("model_id") != active["model_id"] or health.get("execution_policy", {}).get("gpu_policy") != "disabled" or health.get("graph_weight") != 0.0 or health.get("event_weight") != 0.0:
            raise RuntimeError("health endpoint contract failure")
        status, model, elapsed = request(port, "GET", "/api/model")
        checks["model"] = {"status": status, "elapsed_seconds": elapsed, "payload": model}
        if status != 200 or model.get("model_sha256") != active["sha256"] or model.get("feature_count") != 14:
            raise RuntimeError("model endpoint contract failure")
        status, first, latency_one = request(port, "GET", "/api/predict?stock_code=000001.SZ")
        status_two, second, latency_two = request(port, "GET", "/api/predict?stock_code=000001.SZ")
        checks["prediction"] = {"first_status": status, "second_status": status_two, "latency_seconds": [latency_one, latency_two], "first_cache": first.get("input_quality", {}).get("source_cache"), "second_cache": second.get("input_quality", {}).get("source_cache"), "requested_decision_date": first.get("requested_decision_date"), "review_route": first.get("prediction", {}).get("review_route")}
        prob_one = np.asarray([first.get("prediction", {}).get(key) for key in ("p_down", "p_neutral", "p_up")], dtype=float)
        prob_two = np.asarray([second.get("prediction", {}).get(key) for key in ("p_down", "p_neutral", "p_up")], dtype=float)
        if status != 200 or status_two != 200 or not np.isfinite(prob_one).all() or not np.isclose(prob_one.sum(), 1.0, atol=float(policy["checks"]["probability_sum_tolerance"])) or not np.array_equal(prob_one, prob_two) or checks["prediction"]["first_cache"] != "MISS" or checks["prediction"]["second_cache"] != "HIT" or max(latency_one, latency_two) > float(policy["checks"]["prediction_latency_max_seconds"]):
            raise RuntimeError("valid/replay prediction contract failure")
        status, invalid, elapsed = request(port, "GET", "/api/predict?stock_code=invalid")
        checks["invalid_stock"] = {"status": status, "elapsed_seconds": elapsed, "type": invalid.get("type")}
        if status != 400 or invalid.get("type") != "input_error":
            raise RuntimeError("invalid-stock fail-closed contract failure")
        status, future, elapsed = request(port, "GET", "/api/predict?stock_code=000001.SZ&date=2099-01-01")
        checks["future_date"] = {"status": status, "elapsed_seconds": elapsed, "type": future.get("type")}
        if status != 400 or future.get("type") != "input_error":
            raise RuntimeError("future-date fail-closed contract failure")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=10)
    if thread.is_alive():
        failure = failure or "server thread did not terminate"
    output.mkdir(parents=True)
    decision = {"node_id": "AA_GFMNET_LOCAL_T2_LOOPBACK_OPERATIONAL_AUDIT_V1", "status": "PASS_LOCAL_LOOPBACK_OPERATIONAL_ACCEPTANCE" if failure is None else "FAIL_CLOSED_LOCAL_LOOPBACK_OPERATIONAL_ACCEPTANCE", "failure": failure, "policy_sha256": sha256(policy_path), "registry_sha256": sha256(registry_path), "model_sha256": sha256(model_path), "daily_root": str(daily_root), "source_hashes": {field: sha256(path) for field, path in source_paths.items()}, "checks": checks, "target_labels_read": False, "fresh_labels_read": False, "wp10_outputs_read": False, "metrics_read": False, "model_trained": False, "gpu_used": False, "automatic_trading": False, "production_assets_modified": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    decision_path = output / "LOCAL_LOOPBACK_OPERATIONAL_DECISION.json"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt = {"node_id": decision["node_id"], "status": decision["status"], "decision_sha256": sha256(decision_path), "target_labels_read": False, "gpu_used": False}
    (output / "LOCAL_LOOPBACK_OPERATIONAL_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision["status"], "failure": failure}, ensure_ascii=False))
    if failure is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


