from __future__ import annotations

"""Start the active T2 anchor only on 127.0.0.1 with GPU disabled."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the active T2 anchor on a local loopback listener")
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--daily-root", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    package, registry_path, daily_root = args.package_root.resolve(), args.registry.resolve(), args.daily_root.resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8")); active = registry["active_model"]
    model_path = Path(active["path"])
    if active["model_id"] != "RG_OBGNET_CONFIRMED_SAFE_V1_1" or active["target_id"] != "T2_MARKET_RELATIVE_FIXED" or sha256(model_path) != active["sha256"]:
        raise RuntimeError("active anchor identity contract failure")
    if not daily_root.is_dir() or not (1 <= int(args.port) <= 65535):
        raise RuntimeError("daily-root or local port contract failure")
    os.environ["AAGFMNET_GPU_POLICY"] = "disabled"; os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    server_path = package / "apps" / "stock_predictor" / "server.py"
    spec = importlib.util.spec_from_file_location("local_t2_server", server_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import local server")
    module = importlib.util.module_from_spec(spec); sys.modules["local_t2_server"] = module; spec.loader.exec_module(module)
    from src.stock_prediction_service import StockPredictionService
    service = StockPredictionService(model_path, daily_root)
    server = module.create_server(service, "127.0.0.1", int(args.port))
    print(json.dumps({"status": "LOCAL_LOOPBACK_SERVER_STARTED", "url": f"http://127.0.0.1:{server.server_port}", "model_id": active["model_id"], "gpu_policy": "disabled", "automatic_trading": False}, ensure_ascii=False))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()


