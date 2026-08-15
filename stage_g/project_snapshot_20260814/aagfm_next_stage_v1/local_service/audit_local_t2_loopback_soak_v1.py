from __future__ import annotations

"""Bounded loopback soak test for the active CPU-only T2 service."""

import argparse
import hashlib
import http.client
import importlib.util
import json
import os
import sys
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_server(package: Path):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    path = package / "apps" / "stock_predictor" / "server.py"
    spec = importlib.util.spec_from_file_location("local_t2_soak_server", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import local server")
    module = importlib.util.module_from_spec(spec); sys.modules["local_t2_soak_server"] = module; spec.loader.exec_module(module)
    return module


def predict(port: int, code: str) -> tuple[np.ndarray, float, str]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    started = time.perf_counter(); connection.request("GET", f"/api/predict?stock_code={code}")
    response = connection.getresponse(); payload = json.loads(response.read().decode("utf-8")); elapsed = time.perf_counter() - started; connection.close()
    if response.status != 200:
        raise RuntimeError(f"unexpected response status {response.status}: {payload}")
    probability = np.asarray([payload["prediction"][key] for key in ("p_down", "p_neutral", "p_up")], dtype=float)
    if not np.isfinite(probability).all() or not np.isclose(probability.sum(), 1.0, atol=1e-12):
        raise RuntimeError("probability contract failure")
    return probability, elapsed, str(payload["input_quality"]["source_cache"])


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
    policy = json.loads(policy_path.read_text(encoding="utf-8")); registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if policy.get("status") != "FROZEN_BEFORE_LOCAL_LOOPBACK_SOAK_TEST" or sha256(registry_path) != policy["registry_sha256"]:
        raise RuntimeError("soak policy/registry contract failure")
    active = registry["active_model"]; model_path = Path(active["path"])
    if active["sha256"] != policy["model_sha256"] or sha256(model_path) != policy["model_sha256"] or not daily_root.is_dir():
        raise RuntimeError("soak active-model/daily-source contract failure")
    os.environ["AAGFMNET_GPU_POLICY"] = "disabled"; os.environ["OMP_NUM_THREADS"] = str(policy["cpu_threads"]); os.environ["MKL_NUM_THREADS"] = str(policy["cpu_threads"]); os.environ["OPENBLAS_NUM_THREADS"] = str(policy["cpu_threads"])
    server_module = load_server(package)
    from src.stock_prediction_service import StockPredictionService
    service = StockPredictionService(model_path, daily_root)
    server = server_module.create_server(service, str(policy["host"]), 0); thread = threading.Thread(target=server.serve_forever, name="local-t2-soak", daemon=True); thread.start()
    failure: str | None = None; latencies: list[float] = []; cache_states: list[str] = []
    try:
        port = int(server.server_port); code = str(policy["stock_code"])
        baseline, latency, state = predict(port, code); latencies.append(latency); cache_states.append(state)
        tracemalloc.start(); before_current, _ = tracemalloc.get_traced_memory()
        for _ in range(int(policy["sequential_requests"])):
            probability, latency, state = predict(port, code); latencies.append(latency); cache_states.append(state)
            if not np.array_equal(probability, baseline):
                raise RuntimeError("sequential prediction changed")
        with ThreadPoolExecutor(max_workers=int(policy["max_client_workers"])) as executor:
            futures = [executor.submit(predict, port, code) for _ in range(int(policy["concurrent_requests"]))]
            for future in futures:
                probability, latency, state = future.result(); latencies.append(latency); cache_states.append(state)
                if not np.array_equal(probability, baseline):
                    raise RuntimeError("concurrent prediction changed")
        after_current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        growth = max(0, int(after_current - before_current))
        p99 = float(np.quantile(np.asarray(latencies, dtype=float), 0.99))
        if p99 > float(policy["p99_latency_max_seconds"]) or growth > int(policy["max_tracemalloc_current_growth_bytes"]) or any(state != "HIT" for state in cache_states[1:]):
            raise RuntimeError("latency, memory, or cache acceptance threshold failed")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"; p99 = None; growth = None; peak = None
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=10)
    if thread.is_alive():
        failure = failure or "server thread did not terminate"
    output.mkdir(parents=True)
    decision = {"node_id": "AA_GFMNET_LOCAL_T2_LOOPBACK_SOAK_V1", "status": "PASS_LOCAL_LOOPBACK_SOAK" if failure is None else "FAIL_CLOSED_LOCAL_LOOPBACK_SOAK", "failure": failure, "policy_sha256": sha256(policy_path), "registry_sha256": sha256(registry_path), "model_sha256": sha256(model_path), "requests": int(len(latencies)), "latency_seconds": {"min": float(np.min(latencies)) if latencies else None, "median": float(np.median(latencies)) if latencies else None, "p95": float(np.quantile(latencies, 0.95)) if latencies else None, "p99": p99}, "tracemalloc_current_growth_bytes": growth, "tracemalloc_peak_bytes": int(peak) if peak is not None else None, "cache_states": {"miss": int(cache_states.count("MISS")), "hit": int(cache_states.count("HIT"))}, "target_labels_read": False, "fresh_labels_read": False, "wp10_outputs_read": False, "metrics_read": False, "model_trained": False, "gpu_used": False, "automatic_trading": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    path = output / "LOCAL_LOOPBACK_SOAK_DECISION.json"; path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision["status"], "requests": decision["requests"], "failure": failure}, ensure_ascii=False))
    if failure is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


