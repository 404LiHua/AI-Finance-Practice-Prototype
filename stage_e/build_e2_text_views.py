"""Build leakage-safe E-2 no-text, TF-IDF/SVD and pretrained semantic views."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.custody import StageEDataCustodyGuard  # noqa: E402
from stage_e.hashing import sha256_file, stable_json_sha256  # noqa: E402


DEFAULT_CUSTODY = REPO_ROOT / "stage_e/configs/data_custody_v1.json"
REQUIRED_EVENT_COLUMNS = {
    "published_at", "source_name", "source_item_id", "source_url", "stock_code",
    "title", "body", "license_id", "retrieved_at", "source_record_sha256",
}
REQUIRED_LICENSE_COLUMNS = {
    "license_id", "asset_type", "provider", "effective_from", "effective_to",
    "research_use_allowed", "commercial_use_allowed", "derivative_features_allowed", "raw_text_storage_allowed",
    "redistribution_allowed", "evidence_path", "evidence_sha256", "license_notes",
}


class E2TextGovernanceError(RuntimeError):
    """Raised when source, timestamp, mapping, custody or use-scope checks fail."""


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def normalize_code(value: Any) -> str:
    text = re.sub(r"\.0$", "", str(value).strip().upper())
    if re.fullmatch(r"\d{6}\.(SZ|SH)", text):
        return text
    if re.fullmatch(r"\d{6}", text):
        return f"{text}.{'SH' if text.startswith(('5', '6', '9')) else 'SZ'}"
    raise E2TextGovernanceError(f"invalid explicit stock_code mapping: {value!r}")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise E2TextGovernanceError(f"unsupported text input format: {path}")


def canonical_source_record_sha256(row: pd.Series | dict[str, Any]) -> str:
    values = row if isinstance(row, dict) else row.to_dict()
    published = pd.Timestamp(values["published_at"])
    if published.tzinfo is None:
        raise E2TextGovernanceError("published_at must contain timezone information")
    published_text = published.tz_convert("UTC").isoformat()
    payload = "|".join([
        published_text,
        str(values["source_name"]).strip(), str(values["source_item_id"]).strip(),
        str(values["source_url"]).strip(), normalize_code(values["stock_code"]),
        str(values["title"]).strip(), str(values["body"]).strip(), str(values["license_id"]).strip(),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_license_registry(path: Path, required_use_scope: str) -> pd.DataFrame:
    registry = pd.read_csv(path, dtype=str).fillna("")
    missing = sorted(REQUIRED_LICENSE_COLUMNS - set(registry.columns))
    if missing:
        raise E2TextGovernanceError(f"license registry missing columns: {missing}")
    if registry["license_id"].duplicated().any():
        raise E2TextGovernanceError("license registry contains duplicate license_id")
    for column in ("research_use_allowed", "commercial_use_allowed", "derivative_features_allowed", "raw_text_storage_allowed", "redistribution_allowed"):
        registry[column] = registry[column].map(parse_bool)
    registry["effective_from"] = pd.to_datetime(registry["effective_from"], errors="coerce")
    registry["effective_to"] = pd.to_datetime(registry["effective_to"], errors="coerce")
    for row in registry.itertuples(index=False):
        evidence = Path(row.evidence_path)
        if not evidence.is_absolute():
            evidence = path.parent / evidence
        if not evidence.exists():
            raise E2TextGovernanceError(f"license evidence file missing for {row.license_id}: {evidence}")
        if sha256_file(evidence).casefold() != str(row.evidence_sha256).casefold():
            raise E2TextGovernanceError(f"license evidence SHA-256 mismatch for {row.license_id}")
        if not row.derivative_features_allowed:
            raise E2TextGovernanceError(f"license {row.license_id} does not allow derivative features")
        if required_use_scope == "commercial" and not row.commercial_use_allowed:
            raise E2TextGovernanceError(f"license {row.license_id} does not allow commercial use")
        if required_use_scope == "academic_research" and not row.research_use_allowed:
            raise E2TextGovernanceError(f"license {row.license_id} does not allow academic research")
    return registry


def discover_event_files(root: Path, patterns: list[str]) -> list[Path]:
    files = sorted({path for pattern in patterns for path in root.glob(pattern) if path.is_file()})
    if not files:
        raise E2TextGovernanceError(f"no licensed text files found in {root}")
    return files


def validate_events(
    files: list[Path],
    registry: pd.DataFrame,
    allowed_codes: set[str],
    development_ceiling: pd.Timestamp,
    timezone_name: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    manifest = []
    for path in files:
        frame = read_table(path)
        missing = sorted(REQUIRED_EVENT_COLUMNS - set(frame.columns))
        if missing:
            raise E2TextGovernanceError(f"{path.name} missing columns: {missing}")
        frame = frame[list(REQUIRED_EVENT_COLUMNS)].copy()
        frame["input_file"] = path.name
        frame["input_file_sha256"] = sha256_file(path)
        frame["input_row_number"] = np.arange(2, len(frame) + 2, dtype=np.int64)
        frames.append(frame)
        manifest.append({"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    events = pd.concat(frames, ignore_index=True)
    events["stock_code"] = events["stock_code"].map(normalize_code)
    events["published_at"] = pd.to_datetime(events["published_at"], errors="coerce", utc=True)
    events["retrieved_at"] = pd.to_datetime(events["retrieved_at"], errors="coerce", utc=True)
    if events[["published_at", "retrieved_at"]].isna().any().any():
        raise E2TextGovernanceError("published_at and retrieved_at must be timezone-aware parseable timestamps")
    if events["published_at"].dt.tz_convert(timezone_name).dt.tz_localize(None).gt(development_ceiling + pd.Timedelta(days=1)).any():
        raise E2TextGovernanceError("licensed text contains post-development-ceiling publication")
    if (~events["stock_code"].isin(allowed_codes)).any():
        bad = sorted(events.loc[~events["stock_code"].isin(allowed_codes), "stock_code"].unique())
        raise E2TextGovernanceError(f"text contains unmapped/out-of-universe stock codes: {bad[:20]}")
    for column in ("source_name", "source_item_id", "source_url", "title", "body", "license_id", "source_record_sha256"):
        if events[column].fillna("").astype(str).str.strip().eq("").any():
            raise E2TextGovernanceError(f"text contains empty required field: {column}")
    if events.duplicated(["source_name", "source_item_id", "stock_code"]).any():
        raise E2TextGovernanceError("duplicate source item and stock mapping found")

    licenses = registry.set_index("license_id")
    unknown = sorted(set(events["license_id"].astype(str)) - set(licenses.index.astype(str)))
    if unknown:
        raise E2TextGovernanceError(f"events reference unknown license IDs: {unknown}")
    for row in events.itertuples(index=False):
        license_row = licenses.loc[str(row.license_id)]
        published = row.published_at.tz_convert(timezone_name).tz_localize(None).normalize()
        if pd.notna(license_row["effective_from"]) and published < license_row["effective_from"]:
            raise E2TextGovernanceError(f"license {row.license_id} not yet effective for {row.source_item_id}")
        if pd.notna(license_row["effective_to"]) and published > license_row["effective_to"]:
            raise E2TextGovernanceError(f"license {row.license_id} expired for {row.source_item_id}")
        if canonical_source_record_sha256(pd.Series(row._asdict())) != str(row.source_record_sha256).casefold():
            raise E2TextGovernanceError(f"source record SHA-256 mismatch: {row.source_item_id}")
    return events, manifest


def align_availability(events: pd.DataFrame, open_weeks: pd.DatetimeIndex, timezone_name: str) -> pd.DataFrame:
    work = events.copy()
    local = work["published_at"].dt.tz_convert(timezone_name)
    local_naive = local.dt.tz_localize(None)
    candidate = local_naive.dt.to_period("W-FRI").dt.end_time.dt.normalize()
    friday_after_close = local.dt.dayofweek.eq(4) & (local.dt.hour * 60 + local.dt.minute).gt(15 * 60)
    candidate = candidate + pd.to_timedelta(friday_after_close.astype(int) * 7, unit="D")
    positions = open_weeks.searchsorted(candidate.to_numpy(), side="left")
    valid = positions < len(open_weeks)
    work = work.loc[valid].copy()
    work["trade_date"] = open_weeks.take(positions[valid]).to_numpy()
    return work


def aggregate_documents(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["document"] = (events["title"].astype(str) + "\n" + events["body"].astype(str)).str.strip()
    return events.groupby(["stock_code", "trade_date"], as_index=False).agg(
        text_count=("source_item_id", "size"),
        document=("document", lambda values: "\n[DOC_SEP]\n".join(values)),
        source_names=("source_name", lambda values: "|".join(sorted(set(map(str, values))))),
        license_ids=("license_id", lambda values: "|".join(sorted(set(map(str, values))))),
    )


def load_assignments(config: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(config["paths"]["fold_assignments"])
    assignments = pd.read_csv(path)
    assignments["trade_date"] = pd.to_datetime(assignments["trade_date"], errors="coerce")
    assignments["target_date"] = pd.to_datetime(assignments["target_date"], errors="coerce")
    required = {"fold_id", "split", "stock_code", "trade_date", "target_date", "sample_row_id"}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise E2TextGovernanceError(f"fold assignments missing columns: {missing}")
    return assignments


def build_no_text(assignments: pd.DataFrame, output_root: Path) -> Path:
    output = assignments[["fold_id", "split", "stock_code", "trade_date", "target_date", "sample_row_id"]].copy()
    output["text_available"] = False
    output["text_count"] = 0
    output["text_feature_dim"] = 0
    path = output_root / "no_text_view.csv.gz"
    output.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    return path


def build_tfidf_svd(
    assignments: pd.DataFrame,
    documents: pd.DataFrame,
    output_root: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise E2TextGovernanceError("scikit-learn is required for TF-IDF/SVD view") from exc
    records = []
    options = config["tfidf_svd"]
    for fold_id, fold in assignments.groupby("fold_id", sort=True):
        joined = fold.merge(documents, on=["stock_code", "trade_date"], how="left", validate="many_to_one")
        joined["document"] = joined["document"].fillna("")
        train_mask = joined["split"].eq("train") & joined["document"].ne("")
        if train_mask.sum() < int(options["minimum_train_documents"]):
            raise E2TextGovernanceError(
                f"{fold_id} has only {int(train_mask.sum())} licensed training documents"
            )
        vectorizer = TfidfVectorizer(
            analyzer=options.get("analyzer", "char"),
            ngram_range=tuple(options.get("ngram_range", [2, 4])),
            max_features=int(options["max_features"]), min_df=int(options.get("min_df", 2)),
        )
        train_matrix = vectorizer.fit_transform(joined.loc[train_mask, "document"])
        components = min(int(options["svd_components"]), train_matrix.shape[0] - 1, train_matrix.shape[1] - 1)
        if components < 1:
            raise E2TextGovernanceError(f"{fold_id} has insufficient TF-IDF rank for SVD")
        svd = TruncatedSVD(n_components=components, random_state=int(config["random_seed"]))
        svd.fit(train_matrix)
        transformed = svd.transform(vectorizer.transform(joined["document"]))
        transformed[joined["document"].eq("").to_numpy()] = 0.0
        output = joined[["fold_id", "split", "stock_code", "trade_date", "target_date", "sample_row_id"]].copy()
        output["text_available"] = joined["document"].ne("")
        output["text_count"] = joined["text_count"].fillna(0).astype(np.int32)
        for index in range(components):
            output[f"text_svd_{index + 1:03d}"] = transformed[:, index]
        fold_root = output_root / "tfidf_svd" / fold_id
        fold_root.mkdir(parents=True, exist_ok=True)
        feature_path = fold_root / "features.csv.gz"
        output.to_csv(feature_path, index=False, compression={"method": "gzip", "mtime": 0})
        model_path = fold_root / "train_fitted_model.pkl"
        with model_path.open("wb") as handle:
            pickle.dump({"vectorizer": vectorizer, "svd": svd}, handle)
        records.append({
            "fold_id": fold_id, "training_documents": int(train_mask.sum()),
            "validation_documents": int((joined["split"].eq("validation") & joined["document"].ne("")).sum()),
            "vocabulary_size": len(vectorizer.vocabulary_), "components": components,
            "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
            "features_sha256": sha256_file(feature_path), "model_sha256": sha256_file(model_path),
        })
    return records


def build_semantic(
    assignments: pd.DataFrame,
    documents: pd.DataFrame,
    registry: pd.DataFrame,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    options = config["semantic_encoder"]
    model_path = resolve_path(options["model_path"])
    if not model_path.exists():
        raise E2TextGovernanceError(f"pretrained semantic model is missing: {model_path}")
    license_id = str(options["model_license_id"])
    license_rows = registry[registry["license_id"].eq(license_id) & registry["asset_type"].eq("model")]
    if len(license_rows) != 1:
        raise E2TextGovernanceError(f"semantic model license is not uniquely registered: {license_id}")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise E2TextGovernanceError("sentence-transformers is required for semantic view") from exc
    model = SentenceTransformer(str(model_path), device=options.get("device", "cpu"))
    unique = documents[["stock_code", "trade_date", "document", "text_count"]].copy()
    vectors = model.encode(
        unique["document"].tolist(), batch_size=int(options.get("batch_size", 32)),
        show_progress_bar=False, normalize_embeddings=bool(options.get("normalize_embeddings", True)),
    )
    vector_columns = [f"text_semantic_{index + 1:03d}" for index in range(vectors.shape[1])]
    vector_frame = pd.DataFrame(vectors, columns=vector_columns, index=unique.index)
    encoded = pd.concat([unique[["stock_code", "trade_date", "text_count"]], vector_frame], axis=1)
    joined = assignments.merge(encoded, on=["stock_code", "trade_date"], how="left", validate="many_to_one")
    joined["text_available"] = joined["text_count"].notna()
    joined["text_count"] = joined["text_count"].fillna(0).astype(np.int32)
    joined[vector_columns] = joined[vector_columns].fillna(0.0)
    path = output_root / "semantic_view.csv.gz"
    joined[["fold_id", "split", "stock_code", "trade_date", "target_date", "sample_row_id", "text_available", "text_count", *vector_columns]].to_csv(
        path, index=False, compression={"method": "gzip", "mtime": 0}
    )
    return {
        "model_path": str(model_path.resolve()), "model_sha256": stable_json_sha256([
            {"path": str(path.relative_to(model_path)), "sha256": sha256_file(path)}
            for path in sorted(model_path.rglob("*")) if path.is_file() and ".git" not in path.parts
        ]),
        "model_license_id": license_id, "embedding_dimension": int(vectors.shape[1]),
        "features_sha256": sha256_file(path),
    }


def build(config_path: Path, requested_views: set[str], overwrite: bool = False) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    guard = StageEDataCustodyGuard.from_config(DEFAULT_CUSTODY, REPO_ROOT)
    panel_path = guard.assert_path_allowed(resolve_path(config["paths"]["panel"]), purpose="E-2 panel")
    assignments_path = guard.assert_path_allowed(resolve_path(config["paths"]["fold_assignments"]), purpose="E-2 assignments")
    output_root = guard.assert_path_allowed(resolve_path(config["paths"]["output_root"]), purpose="E-2 outputs")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(panel_path, usecols=["trade_date", "stock_code", "is_market_open_week"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    guard.assert_development_frame(panel, date_columns=("trade_date",))
    allowed_codes = set(panel["stock_code"].astype(str).unique())
    open_weeks = pd.DatetimeIndex(sorted(panel.loc[panel["is_market_open_week"].astype(bool), "trade_date"].unique()))
    assignments = load_assignments(config)
    artifacts: dict[str, Any] = {}
    if "no_text" in requested_views:
        path = build_no_text(assignments, output_root)
        artifacts["no_text"] = {"path": str(path), "sha256": sha256_file(path)}

    text_views = requested_views & {"tfidf_svd", "semantic"}
    source_manifest: list[dict[str, Any]] = []
    if text_views:
        licensed_root = guard.assert_path_allowed(resolve_path(config["paths"]["licensed_text_root"]), purpose="licensed text")
        license_path = guard.assert_path_allowed(resolve_path(config["paths"]["license_registry"]), purpose="license registry")
        if not licensed_root.exists() or not license_path.exists():
            raise E2TextGovernanceError(
                "research text package is incomplete: licensed_text_root and license_registry are required"
            )
        registry = load_license_registry(license_path, config.get("required_use_scope", "academic_research"))
        files = discover_event_files(licensed_root, config["input_patterns"])
        events, source_manifest = validate_events(
            files, registry, allowed_codes, pd.Timestamp(config["development_date_ceiling"]), config["timezone"]
        )
        aligned = align_availability(events, open_weeks, config["timezone"])
        documents = aggregate_documents(aligned)
        if "tfidf_svd" in requested_views:
            artifacts["tfidf_svd"] = build_tfidf_svd(assignments, documents, output_root, config)
        if "semantic" in requested_views:
            artifacts["semantic"] = build_semantic(assignments, documents, registry, output_root, config)
    metadata = {
        "stage": "E-2", "data_batch_id": config["data_batch_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_date_ceiling": config["development_date_ceiling"],
        "requested_views": sorted(requested_views), "artifacts": artifacts,
        "source_manifest": source_manifest,
        "config_sha256": sha256_file(config_path), "panel_sha256": sha256_file(panel_path),
        "assignments_sha256": sha256_file(assignments_path),
        "sealed_data_read": False, "future_screening_or_final_read": False,
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--views", nargs="+", choices=["no_text", "tfidf_svd", "semantic"], default=["no_text", "tfidf_svd", "semantic"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(build(config_path, set(args.views), overwrite=args.overwrite))


if __name__ == "__main__":
    main()
