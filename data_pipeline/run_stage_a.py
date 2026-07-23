"""Run and verify the complete 30-stock Stage A data workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def write_report(project_root: Path, config: dict, metadata: dict, metrics: dict, text_meta: dict) -> Path:
    report_dir = project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "STAGE_A_30_STOCKS_REPORT.md"
    quality = metadata["quality_before_features"]
    split = metadata["eligible_sample_counts"]
    csmar = metadata["csmar_source"]
    lines = [
        "# Stage A: 30-stock data pipeline acceptance report",
        "",
        f"Report date: {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        f"- Stocks: {metadata['selected_stock_count']}",
        f"- Date range: {config['date_range']['start']} to {config['date_range']['end']}",
        "- Raw weekly prices: JiuZhang Quant, unadjusted",
        "- Model prices: BaoStock, forward-adjusted",
        "- Structured events: CSMAR special treatment and capital changes",
        "- Split: chronological train/validation/test with purge weeks",
        "",
        "## Acceptance results",
        "",
        f"- Raw selected rows: {quality['rows']}",
        f"- Duplicate stock/date rows: {quality['duplicate_stock_date']}",
        f"- Bad dates: {quality['bad_dates']}",
        f"- Missing numeric cells: {quality['missing_numeric_cells']}",
        f"- Invalid OHLC rows: {quality['invalid_ohlc_rows']}",
        f"- Previous-close continuity differences (informational): {quality['previous_close_continuity_mismatch_rows']}",
        f"- BaoStock coverage: {metadata['baostock_coverage_selected_rows']:.1%}",
        f"- CSMAR weekly-state coverage: {metadata['csmar_weekly_coverage_selected_rows']:.1%}",
        f"- Text/event records aligned to panel: {metadata['text_event_rows_in_panel']}",
        f"- Training text rows used to fit TF-IDF: {text_meta['training_text_rows']}",
        f"- TF-IDF vocabulary size: {text_meta['tfidf_vocabulary_size']}",
        f"- SVD dimensions: {text_meta['svd_components']}",
        f"- Text clusters: {text_meta['clusters']}",
        f"- Train samples: {split.get('train', 0)}",
        f"- Validation samples: {split.get('validation', 0)}",
        f"- Test samples: {split.get('test', 0)}",
        "",
        "## Baseline verification",
        "",
        f"- Validation MAE: {metrics['validation']['mae']:.6f}",
        f"- Validation RMSE: {metrics['validation']['rmse']:.6f}",
        f"- Validation direction accuracy: {metrics['validation']['direction_accuracy']:.2%}",
        f"- Test MAE: {metrics['test']['mae']:.6f}",
        f"- Test RMSE: {metrics['test']['rmse']:.6f}",
        f"- Test direction accuracy: {metrics['test']['direction_accuracy']:.2%}",
        "",
        "## Traceability",
        "",
        f"- JiuZhang source files indexed: {metadata['raw_file_count']}",
        f"- BaoStock downloaded files: {len(metadata['baostock_source']['files'])}",
        f"- CSMAR source ZIP files: {len(csmar['source_zips'])}",
        "- Every source file is recorded with SHA-256; CSMAR event rows retain source ZIP and row number.",
        "",
        "## Known limitations",
        "",
        "- The exported CSMAR category provides special-treatment/capital events, not broad daily financial news.",
        "- Event text is sparse: most normal stocks have no special-treatment event during one year.",
        "- CSMAR exports are marked for Jiaxing University use only and are excluded from Git.",
        "- The random-forest model is a data-path verification baseline, not the proposed graph-frequency model.",
        "- Stage B should reproduce stronger time-series baselines before expanding beyond 30 stocks.",
        "",
        "## Gate decision",
        "",
        "**PASS.** Fatal quality counts are zero, source coverage is 100%, text features are fitted from training data only, and all splits are non-empty.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "stock_count": metadata["selected_stock_count"],
        "date_range": config["date_range"],
        "quality": quality,
        "coverage": {
            "baostock": metadata["baostock_coverage_selected_rows"],
            "csmar": metadata["csmar_weekly_coverage_selected_rows"],
        },
        "samples": split,
        "event_count": metadata["text_event_rows_in_panel"],
        "text_features": text_meta,
        "baseline": {"validation": metrics["validation"], "test": metrics["test"]},
    }
    (report_dir / "stage_a_30stocks_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report_path


def verify(metadata: dict) -> None:
    quality = metadata["quality_before_features"]
    fatal_fields = [
        "duplicate_stock_date", "bad_dates", "missing_numeric_cells",
        "invalid_ohlc_rows", "nonpositive_price_rows",
    ]
    failures = [field for field in fatal_fields if quality.get(field, 0) != 0]
    if metadata["selected_stock_count"] != 30:
        failures.append("selected_stock_count")
    if metadata["baostock_coverage_selected_rows"] != 1.0:
        failures.append("baostock_coverage")
    if metadata["csmar_weekly_coverage_selected_rows"] != 1.0:
        failures.append("csmar_coverage")
    for split_name in ("train", "validation", "test"):
        if metadata["eligible_sample_counts"].get(split_name, 0) <= 0:
            failures.append(f"empty_{split_name}")
    if failures:
        raise RuntimeError(f"Stage A acceptance failed: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("data_pipeline/configs/weekly_a_share.json"))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run([sys.executable, "-m", "unittest", "discover", "-s", "data_pipeline/tests", "-v"], project_root)
    run([sys.executable, "data_pipeline/prepare_csmar_events.py", "--config", str(config_path)], project_root)
    if not args.skip_download:
        run([sys.executable, "data_pipeline/download_baostock_adjusted.py", "--config", str(config_path)], project_root)
    run([
        sys.executable, "data_pipeline/build_weekly_dataset.py",
        "--config", str(config_path), "--overwrite",
    ], project_root)
    run([sys.executable, "data_pipeline/build_text_features.py", "--config", str(config_path)], project_root)
    run([sys.executable, "data_pipeline/train_fusion_baseline.py", "--config", str(config_path)], project_root)
    data_root = project_root / config["output"]["root"]
    metadata = json.loads((data_root / "metadata.json").read_text(encoding="utf-8"))
    text_meta = json.loads((data_root / "text_features_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (project_root / "outputs/stage_a_30stocks_baseline_v1/metrics.json").read_text(encoding="utf-8")
    )
    verify(metadata)
    print(write_report(project_root, config, metadata, metrics, text_meta))


if __name__ == "__main__":
    main()
