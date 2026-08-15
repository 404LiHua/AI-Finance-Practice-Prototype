"""Build the local Stage-F importance archive after the Git commit is created."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def run(output_root: Path, version: str, commit: str) -> dict:
    release = REPO_ROOT / "releases/stage_f_closure_v1"
    target = output_root / f"AI_Finance_Prototype_{version}_{commit[:7]}"
    zip_path = Path(f"{target}.zip")
    if target.exists() or zip_path.exists():
        raise FileExistsError(f"archive target exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(release, target / "stage_f_closure_v1")
    version_info = {
        "version": version, "git_commit": commit, "git_tag": version,
        "stage_f_conclusion": "FORMAL_NO_ROBUST_PROMOTABLE_CANDIDATE_RETAIN_STAGE_E_INCUMBENT",
        "retained_model": "stock_node_gwnet_fixed_industry_l8",
        "row_level_predictions_included": False, "checkpoints_included": False,
    }
    (target / "VERSION.json").write_text(
        json.dumps(version_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    shutil.make_archive(str(target), "zip", root_dir=target)
    result = {
        "version": version, "git_commit": commit, "archive_directory": str(target),
        "archive_zip": str(zip_path), "zip_sha256": sha256_file(zip_path),
        "release_manifest_sha256": sha256_file(target / "stage_f_closure_v1/SHA256_MANIFEST.json"),
    }
    (target / "ARCHIVE_RECEIPT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("D:/项目/源文件/importance"))
    parser.add_argument("--version", default="v0.6.0-stage-f")
    parser.add_argument("--commit", default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root, args.version, args.commit or current_commit()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
