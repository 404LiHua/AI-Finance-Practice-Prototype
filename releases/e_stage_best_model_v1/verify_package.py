"""Verify release hashes and independently load all nine checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from inference import load_one


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    hashes_ok = all(sha256_file(root / item["path"]) == item["sha256"] for item in manifest["artifacts"])
    checkpoints = sorted((root / "checkpoints").rglob("*.pt"))
    loaded = []
    for checkpoint in checkpoints:
        model, payload = load_one(checkpoint)
        dummy = np.zeros((1, 8, 100, int(payload["input_size"])), dtype=np.float32)
        prediction = model(torch.from_numpy(dummy)).detach().numpy()
        loaded.append(bool(prediction.shape == (1, 100) and np.isfinite(prediction).all()))
    result = {"status": "PASS" if hashes_ok and len(checkpoints) == 9 and all(loaded) else "FAIL", "hashes_ok": hashes_ok, "checkpoint_count": len(checkpoints), "all_checkpoints_load": all(loaded)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
