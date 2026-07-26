"""Register a local pretrained model and its auditable license evidence for E-2."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.hashing import sha256_file, stable_json_sha256

MODEL_LICENSE_ID = "MODEL_BGE_SMALL_ZH_V15_MIT"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--license-source", type=Path, required=True)
    parser.add_argument("--batch-roots", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    model_path = args.model_path.resolve()
    model_card = model_path / "README.md"
    license_source = args.license_source.resolve()
    if not model_card.exists() or not license_source.exists():
        raise FileNotFoundError("model card or official license source is missing")
    model_files = [
        {"path": str(path.relative_to(model_path)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(model_path.rglob("*")) if path.is_file() and ".git" not in path.parts
    ]
    model_manifest = {
        "model_id": "BAAI/bge-small-zh-v1.5",
        "distribution_source": "https://www.modelscope.cn/AI-ModelScope/bge-small-zh-v1.5",
        "upstream_project": "https://github.com/FlagOpen/FlagEmbedding",
        "license": "MIT",
        "model_card_sha256": sha256_file(model_card),
        "official_license_sha256": sha256_file(license_source),
        "files": model_files,
    }
    model_manifest["model_sha256"] = stable_json_sha256(model_files)
    (model_path / "E2_MODEL_MANIFEST.json").write_text(
        json.dumps(model_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for batch_root in args.batch_roots:
        batch_root = batch_root.resolve()
        licenses = batch_root / "licenses"
        licenses.mkdir(parents=True, exist_ok=True)
        copied_license = licenses / "BGE_FLAGEMBEDDING_MIT_LICENSE.txt"
        copied_card = licenses / "BGE_SMALL_ZH_V15_MODEL_CARD.md"
        shutil.copyfile(license_source, copied_license)
        shutil.copyfile(model_card, copied_card)
        evidence = licenses / "BGE_SMALL_ZH_V15_LICENSE_EVIDENCE.md"
        evidence.write_text(
            "# BGE small zh v1.5 model license evidence\n\n"
            "- Model: `BAAI/bge-small-zh-v1.5`\n"
            "- Distribution: https://www.modelscope.cn/AI-ModelScope/bge-small-zh-v1.5\n"
            "- Upstream: https://github.com/FlagOpen/FlagEmbedding\n"
            "- Declared license: MIT\n"
            f"- Model card SHA-256: `{sha256_file(copied_card)}`\n"
            f"- Official MIT license SHA-256: `{sha256_file(copied_license)}`\n"
            f"- Model file-set SHA-256: `{model_manifest['model_sha256']}`\n",
            encoding="utf-8",
        )
        registry_path = batch_root / "license_registry.csv"
        registry = pd.read_csv(registry_path, dtype=str).fillna("")
        registry = registry.loc[~registry["license_id"].eq(MODEL_LICENSE_ID)].copy()
        row = pd.DataFrame([{
            "license_id": MODEL_LICENSE_ID, "asset_type": "model", "provider": "BAAI / FlagEmbedding",
            "effective_from": "2023-09-01", "effective_to": "", "research_use_allowed": True,
            "commercial_use_allowed": True, "derivative_features_allowed": True,
            "raw_text_storage_allowed": True, "redistribution_allowed": True,
            "evidence_path": "licenses/BGE_SMALL_ZH_V15_LICENSE_EVIDENCE.md",
            "evidence_sha256": sha256_file(evidence),
            "license_notes": "MIT model license; upstream text licenses remain separate",
        }])
        pd.concat([registry, row], ignore_index=True).to_csv(registry_path, index=False, encoding="utf-8-sig")
        manifest_path = batch_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"] = [
            {"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(batch_root.glob("*.csv"))
        ]
        manifest["semantic_model_license_id"] = MODEL_LICENSE_ID
        manifest["semantic_model_sha256"] = model_manifest["model_sha256"]
        manifest.pop("batch_sha256", None)
        manifest["batch_sha256"] = stable_json_sha256(manifest)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"registered {MODEL_LICENSE_ID}: {batch_root}")


if __name__ == "__main__":
    main()
