from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION = "v0.4.0-stage-d"
IMPORTANCE_ROOT = Path("D:/项目/源文件/importance")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, target_root: Path) -> None:
    relative = source.resolve().relative_to(REPO_ROOT.resolve())
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def selected_files() -> list[Path]:
    files: set[Path] = set()
    for name in ("README.md", "CHANGELOG.md"):
        files.add(REPO_ROOT / name)
    for root in (REPO_ROOT / "stage_d", REPO_ROOT / "experiments"):
        for path in root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".log"}:
                files.add(path)
    for path in (REPO_ROOT / "reports").glob("STAGE_D_*"):
        if path.is_file():
            files.add(path)
    files.add(REPO_ROOT / "plans/STAGE_D_IMPLEMENTATION_PLAN.md")
    for relative_root in (
        "outputs/stage_d/d1_rolling_origin_v1",
        "outputs/stage_d/d3_robust_diagnostics_v1",
    ):
        for path in (REPO_ROOT / relative_root).rglob("*"):
            if path.is_file():
                files.add(path)
    for name in (
        "preregistered_config.json", "metrics_by_fold_seed.csv", "per_fold_summary.csv",
        "cross_fold_model_summary.csv", "completed_base_runs.json", "evidence_manifest.json",
        "sha256_manifest.json",
    ):
        files.add(REPO_ROOT / "outputs/stage_d/d2_bounded_baselines_v1" / name)
    for name in ("independent_recalc_result.json", "INDEPENDENT_RECALC_RECEIPT.json"):
        files.add(REPO_ROOT / "outputs/stage_d/d4_independent_recalc_v1" / name)
    for name in ("screening_decision.json", "SCREENING_EVIDENCE.json", "D5_EXECUTION_RECEIPT.json"):
        files.add(REPO_ROOT / "outputs/stage_d/d5_screening_20240614_20250613" / name)
    for path in (
        REPO_ROOT / "data/processed/weekly_30stocks_stage_a_v1/metadata.json",
        REPO_ROOT / "data/processed/weekly_30stocks_stage_a_v1/selected_stocks.txt",
        REPO_ROOT / "data/screening/stage_d_d5_20240614_20250613/SOURCE_MANIFEST.json",
    ):
        files.add(path)
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"archive inputs missing: {missing}")
    return sorted(files)


def current_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def build_archive(commit: str) -> dict[str, object]:
    short = commit[:7]
    target = IMPORTANCE_ROOT / f"AI_Finance_Prototype_{VERSION}_{short}"
    zip_path = Path(f"{target}.zip")
    if target.exists() or zip_path.exists():
        raise FileExistsError(f"release archive target already exists: {target}")
    target.mkdir(parents=True)
    files = selected_files()
    for path in files:
        copy_file(path, target)
    archive_readme = target / "ARCHIVE_README.md"
    archive_readme.write_text(
        "# Stage D local importance archive\n\n"
        f"- Version: {VERSION}\n"
        f"- Git commit: {commit}\n"
        "- Stage conclusion: engineering complete; independent D-SCREENING PASS\n"
        "- Includes: Stage D source, frozen checkpoints, reports, tests, aggregate evidence and manifests\n"
        "- Excludes: D-5 raw weekly data and row-level screening predictions\n"
        "- Restriction: the consumed D-SCREENING interval must not be reused for tuning or reselection\n",
        encoding="utf-8",
    )
    version_path = target / "VERSION.json"
    version_path.write_text(json.dumps({
        "version": VERSION,
        "git_commit": commit,
        "git_tag": VERSION,
        "stage_d_outcome": "PASS",
        "freeze_root_sha256": "e3303c3b4818e547476b0bded9860c6fd0334cf1db4f5b4e5677249955d208f6",
        "row_level_d5_data_included": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = target / "SHA256_MANIFEST.csv"
    manifest_files = sorted(
        path for path in target.rglob("*") if path.is_file() and path != manifest_path
    )
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        for path in manifest_files:
            writer.writerow({
                "path": path.relative_to(target).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    shutil.make_archive(str(target), "zip", root_dir=target)
    return {
        "version": VERSION,
        "git_commit": commit,
        "archive_directory": str(target),
        "archive_zip": str(zip_path),
        "file_count": len(manifest_files),
        "manifest_sha256": sha256_file(manifest_path),
        "zip_sha256": sha256_file(zip_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local Stage D importance release archive.")
    parser.add_argument("--commit", default=None)
    args = parser.parse_args()
    commit = args.commit or current_commit()
    print(json.dumps(build_archive(commit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
