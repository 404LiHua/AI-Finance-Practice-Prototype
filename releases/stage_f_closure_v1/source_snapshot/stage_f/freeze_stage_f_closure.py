"""Freeze the Stage-F negative conclusion and build its auditable source/evidence release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from stage_e.hashing import sha256_file, stable_json_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def selected_source_files() -> list[Path]:
    files: set[Path] = set()
    for path in (REPO_ROOT / "stage_f").rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or "frozen" in path.parts:
            continue
        if path.suffix not in {".py", ".json", ".md"}:
            continue
        files.add(path)
    for path in (REPO_ROOT / "reports").glob("STAGE_F_*.md"):
        files.add(path)
    for relative in (
        "plans/STAGE_F_IMPLEMENTATION_PLAN.md", "plans/PROJECT_MASTER_PLAN_V3.md",
        "plans/project_master_plan_v3.json", "PROJECT_GAP_ANALYSIS_AND_PLAN.md",
        "plans/COMMERCIALIZATION_ROADMAP.md", "README.md", "CHANGELOG.md",
    ):
        files.add(REPO_ROOT / relative)
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage-F source archive inputs missing: {missing}")
    return sorted(files)


def selected_evidence_files() -> list[Path]:
    root = REPO_ROOT / "outputs/stage_f"
    files = [
        path for path in root.rglob("*")
        if path.is_file()
        and path.name != "f3_stage_closure_acceptance_v1.json"
        and path.suffix not in {".pt", ".npz", ".gz"}
    ]
    if not files:
        raise FileNotFoundError("no compact Stage-F evidence found")
    return sorted(files)


def copy_files(files: Iterable[Path], destination: Path, prefix: str) -> int:
    count = 0
    for source in files:
        relative = source.resolve().relative_to(REPO_ROOT.resolve())
        target = destination / prefix / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        count += 1
    return count


def write_final_report(config: dict, metadata: dict, acceptance: dict) -> Path:
    gate_rows = metadata["candidate_gate_summary"]
    table = "\n".join(
        f"| `{row['candidate_id']}` | {row['passed_gate_count']}/20 | {row['overall_mae']:.6f} | "
        f"{row['worst_fold_mae']:.6f} | {row['stocks_below_naive']}/100 | {row['stress_composite_ratio']:.6f} |"
        for row in gate_rows
    )
    gan = metadata["gan_stability_result"]
    report = f"""# 阶段F最终报告

日期：2026-07-28  
版本：`{config['version']}`  
阶段状态：`CLOSED / NO ROBUST PROMOTABLE CANDIDATE`

## 1. 阶段目标与完成情况

阶段F已完成极端行情与鲁棒性协议冻结、三个有界非GAN候选、唯一有界GAN附录、单种子工程回执、三种子复核和统一鲁棒性诊断。全部实验严格使用原三折、三种子、冻结样本键和F-0二十项硬门槛；没有读取新的SCREENING或FINAL。

## 2. 最终候选结论

| 候选 | 通过门槛 | 总体MAE | 最差折MAE | 优于Naive股票 | 压力综合比 |
|---|---:|---:|---:|---:|---:|
{table}

四个候选均未通过全部硬门槛，正式结论为`{config['formal_conclusion']}`。阶段E模型`{config['retained_model']}`继续作为开发期保留模型；这不是生产部署批准。

## 3. GAN负面结论冻结

唯一GAN候选工程9/9运行通过，总耗时244.316秒，但三种子稳定性为：MAE CV `{gan['seed_mae_cv']:.6f}`、最低Pearson `{gan['minimum_pairwise_prediction_pearson']:.6f}`、最低Spearman `{gan['minimum_pairwise_prediction_spearman']:.6f}`、预测标准差均值 `{gan['prediction_seed_std_mean']:.6f}`、P95 `{gan['prediction_seed_std_p95']:.6f}`。除MAE CV外其余四项失败，且不可由正常误差、压力表现或工程成本补偿。

## 4. 证据与工程闭环

- F-2.4正常预测31,500行、压力预测133,812行；
- F-2.4独立验收`{acceptance['passed_checks']}/{acceptance['required_checks']} PASS`；
- 逐股票、行业、市值、收益尾部、最差折、压力场景、稳定性和成本诊断均已生成；
- 所有失败批次、失败种子和负面结论均保留；
- 阶段F必要源码、配置、测试、报告和紧凑证据已归档；大体量检查点、NPZ与逐行gzip预测不提交Git，其SHA证据保留在原始回执中。

## 5. 阶段关闭与下一步

阶段F正式关闭，不新增鲁棒性候选、不降低门槛、不自动申请未来数据。下一阶段为G：先冻结证据化解释契约、事实正确性门槛、拒答规则和人工评价协议，再实现模板/RAG解释控制组。
"""
    path = REPO_ROOT / "reports/STAGE_F_FINAL_REPORT.md"
    path.write_text(report, encoding="utf-8")
    return path


def run(config_path: Path, overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for item in config["upstream"].values():
        if sha256_file(resolve(item["path"])) != item["sha256"]:
            raise RuntimeError(f"F-3 upstream hash mismatch: {item['path']}")
    metadata = json.loads(resolve(config["upstream"]["f2_4_metadata"]["path"]).read_text(encoding="utf-8"))
    acceptance = json.loads(resolve(config["upstream"]["f2_4_acceptance"]["path"]).read_text(encoding="utf-8"))
    if acceptance["status"] != "PASS" or metadata["eligibility_conclusion"]["eligible_candidate_count"] != 0:
        raise RuntimeError("F-3 requires accepted zero-candidate F-2.4 conclusion")
    if metadata["eligibility_conclusion"]["conclusion"] != config["formal_conclusion"]:
        raise RuntimeError("F-3 formal negative conclusion changed")
    write_final_report(config, metadata, acceptance)

    release = resolve(config["paths"]["release_root"])
    if release.exists():
        if not overwrite:
            raise FileExistsError(release)
        shutil.rmtree(release)
    release.mkdir(parents=True)
    source_count = copy_files(selected_source_files(), release, "source_snapshot")
    evidence_count = copy_files(selected_evidence_files(), release, "evidence_snapshot")

    conclusion = {
        "stage": "F", "version": config["version"], "closed_at": config["closed_at"],
        "status": "CLOSED", "formal_conclusion": config["formal_conclusion"],
        "retained_model": config["retained_model"], "promotable_candidate_count": 0,
        "candidate_gate_counts": config["candidate_gate_counts"],
        "gan_non_compensable_stability_failures": config["gan_non_compensable_stability_failures"],
        "new_training_performed_during_closure": False, "new_inference_performed_during_closure": False,
        "ranking_performed_during_closure": False, "screening_accessed": False, "final_accessed": False,
    }
    (release / "NEGATIVE_CONCLUSION.json").write_text(
        json.dumps(conclusion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (release / "README.md").write_text(
        "# Stage F closure archive\n\n"
        "This package freezes the formal no-promotable-robust-candidate conclusion.\n"
        f"Retained model: `{config['retained_model']}`.\n\n"
        "`source_snapshot/` contains necessary source/config/test/report files. "
        "`evidence_snapshot/` contains compact JSON/CSV receipts, including failed attempts. "
        "Checkpoints, NPZ arrays and row-level gzip predictions are intentionally excluded; "
        "their hashes remain recorded in receipts.\n",
        encoding="utf-8",
    )
    manifest_files = sorted(
        path for path in release.rglob("*")
        if path.is_file() and path.name not in {"SHA256_MANIFEST.json", "SHA256_MANIFEST.csv", "FREEZE_RECEIPT.json"}
    )
    entries = [{
        "path": path.relative_to(release).as_posix(), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in manifest_files]
    root_sha = stable_json_sha256(entries)
    manifest = {"version": config["version"], "artifact_count": len(entries), "entries": entries, "manifest_root_sha256": root_sha}
    manifest_json = release / "SHA256_MANIFEST.json"
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (release / "SHA256_MANIFEST.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader(); writer.writerows(entries)
    receipt = {
        "stage": "F-3 closure freeze", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS", "version": config["version"], "source_file_count": source_count,
        "compact_evidence_file_count": evidence_count, "manifest_artifact_count": len(entries),
        "manifest_root_sha256": root_sha, "manifest_sha256": sha256_file(manifest_json),
        "config_sha256": sha256_file(config_path), "formal_conclusion": config["formal_conclusion"],
        "retained_model": config["retained_model"], "large_artifacts_intentionally_excluded": True,
        "new_training_performed": False, "new_inference_performed": False,
        "screening_accessed": False, "final_accessed": False,
    }
    receipt["freeze_receipt_sha256"] = stable_json_sha256(receipt)
    (release / "FREEZE_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(run(path, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
