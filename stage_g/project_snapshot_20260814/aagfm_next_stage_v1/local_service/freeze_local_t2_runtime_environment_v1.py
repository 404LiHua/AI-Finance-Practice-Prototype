from __future__ import annotations

"""Record the exact personal-machine environment used by the local T2 runtime."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = ("numpy", "pandas", "pyarrow", "torch", "xgboost", "cupy-cuda12x")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    registry_path, policy_path, output = args.registry.resolve(), args.policy.resolve(), args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    policy = json.loads(policy_path.read_text(encoding="utf-8")); registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if policy.get("status") != "FROZEN_BEFORE_LOCAL_LOOPBACK_OPERATIONAL_AUDIT" or sha256(registry_path) != policy["identity"]["registry_sha256"]:
        raise RuntimeError("environment lock policy/registry contract failure")
    import torch
    packages: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    if packages["numpy"] is None or packages["pandas"] is None or packages["pyarrow"] is None:
        raise RuntimeError("required anchor/batch runtime package is missing")
    active = registry["active_model"]; model_path = Path(active["path"])
    payload = {"node_id": "AA_GFMNET_LOCAL_T2_RUNTIME_ENVIRONMENT_LOCK_V1", "status": "PASS_LOCAL_RUNTIME_ENVIRONMENT_LOCKED", "registry_sha256": sha256(registry_path), "model_sha256": sha256(model_path), "python": {"implementation": platform.python_implementation(), "version": platform.python_version(), "executable": sys.executable, "platform": platform.platform()}, "packages": packages, "cuda_research_environment": {"torch_cuda_build": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available()), "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "anchor_runtime_gpu_policy": "disabled"}, "roles": {"anchor_http_and_batch": ["numpy", "pandas", "pyarrow", "CPU only"], "research_only": ["torch", "xgboost", "cupy-cuda12x", "CUDA only after a separate protocol"]}, "secrets_recorded": False, "target_labels_read": False, "fresh_labels_read": False, "gpu_used": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True)
    lock_path = output / "LOCAL_T2_RUNTIME_ENVIRONMENT_LOCK.json"
    lock_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "lock_sha256": sha256(lock_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


