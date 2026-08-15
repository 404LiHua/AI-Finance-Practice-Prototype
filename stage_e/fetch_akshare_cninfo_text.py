"""Fetch a bounded, auditable sample of CNInfo announcement PDFs discovered via AKShare."""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_e.build_e2_text_views import canonical_source_record_sha256, normalize_code  # noqa: E402
from stage_e.hashing import sha256_file, stable_json_sha256  # noqa: E402
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"}
QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
STOCK_MAP_URL = "https://www.cninfo.com.cn/new/data/szse_stock.json"
PDF_ROOT = "https://static.cninfo.com.cn/"
LICENSE_ID = "CNINFO_PUBLIC_DISCLOSURE_ACADEMIC_RESEARCH_V1"
PRIORITY = ("风险", "退市", "特别处理", "业绩预告", "日常经营", "重大", "董事会", "股东大会", "年度报告", "半年度报告")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def request_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.request(method, url, headers=HEADERS, timeout=45, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # network boundary
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"request failed after retries: {url}") from last


def fetch_metadata(code: str, org_id: str, start: str, end: str) -> list[dict[str, Any]]:
    payload = {
        "pageNum": "1", "pageSize": "30", "column": "szse", "tabName": "fulltext",
        "plate": "", "stock": f"{code},{org_id}", "searchkey": "", "secid": "",
        "category": "", "trade": "", "seDate": f"{start}~{end}", "sortName": "",
        "sortType": "", "isHLtitle": "true",
    }
    first = request_json("POST", QUERY_URL, data=payload)
    pages = max(1, math.ceil(int(first.get("totalAnnouncement", 0)) / 30))
    rows = list(first.get("announcements") or [])
    for page in range(2, pages + 1):
        payload["pageNum"] = str(page)
        rows.extend(request_json("POST", QUERY_URL, data=payload).get("announcements") or [])
    return rows


def priority_score(title: str) -> tuple[int, int]:
    for index, keyword in enumerate(PRIORITY):
        if keyword in title:
            return index, len(title)
    return len(PRIORITY), len(title)


def select_quarterly(rows: list[dict[str, Any]], max_pdf_kb: int) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        published = pd.to_datetime(row.get("announcementTime"), unit="ms", utc=True, errors="coerce")
        if pd.isna(published) or str(row.get("adjunctType", "")).upper() != "PDF":
            continue
        size = pd.to_numeric(row.get("adjunctSize"), errors="coerce")
        if pd.isna(size) or float(size) > max_pdf_kb:
            continue
        item = dict(row)
        item["published_at"] = published
        item["quarter"] = published.tz_convert("Asia/Shanghai").tz_localize(None).to_period("Q")
        normalized.append(item)
    if not normalized:
        return []
    frame = pd.DataFrame(normalized)
    frame["priority"] = frame["announcementTitle"].fillna("").map(priority_score)
    frame["priority_a"] = frame["priority"].map(lambda value: value[0])
    frame["priority_b"] = frame["priority"].map(lambda value: value[1])
    frame = frame.sort_values(["quarter", "priority_a", "adjunctSize", "priority_b", "announcementTime"])
    return frame.groupby("quarter", as_index=False).head(1).to_dict("records")


def fetch_selected_for_code(code: str, org_id: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = fetch_metadata(code[:6], org_id, config["start_date"], config["end_date"])
    selected = select_quarterly(rows, int(config["max_pdf_kb"]))
    for item in selected:
        item["project_stock_code"] = code
        item["published_at"] = pd.Timestamp(item["published_at"]).isoformat()
        item["quarter"] = str(item["quarter"])
        item.pop("priority", None)
    return selected


def extract_pdf(item: dict[str, Any], max_pages: int, max_chars: int) -> dict[str, Any] | None:
    url = PDF_ROOT + str(item["adjunctUrl"]).lstrip("/")
    last: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(url, headers=HEADERS, timeout=60)
            response.raise_for_status()
            content = response.content
            reader = PdfReader(io.BytesIO(content))
            pages = []
            for page in reader.pages[:max_pages]:
                pages.append(page.extract_text() or "")
                if sum(map(len, pages)) >= max_chars:
                    break
            body = "\n".join(pages).strip()[:max_chars]
            if len(body) < 80:
                return None
            return {"body": body, "pdf_url": url, "pdf_sha256": __import__("hashlib").sha256(content).hexdigest(), "pdf_bytes": len(content)}
        except Exception as exc:  # malformed or unavailable PDF
            last = exc
            time.sleep(1.5 * (attempt + 1))
    return None


def write_license_files(output_root: Path, akshare_license_path: Path) -> None:
    licenses = output_root / "licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    akshare_copy = licenses / "AKSHARE_MIT_LICENSE.txt"
    akshare_copy.write_bytes(akshare_license_path.read_bytes())
    basis = licenses / "CNINFO_RESEARCH_USE_BASIS.md"
    basis.write_text(
        "# CNInfo academic-research use basis\n\n"
        "This batch contains public statutory disclosure documents accessed from CNInfo through "
        "the interface documented by AKShare. The project uses them only for academic research, "
        "stores source URLs and hashes, does not assert commercial rights, and does not redistribute raw PDFs.\n",
        encoding="utf-8",
    )
    registry = pd.DataFrame([
        {
            "license_id": LICENSE_ID, "asset_type": "text", "provider": "CNINFO via AKShare",
            "effective_from": "2018-01-01", "effective_to": "", "research_use_allowed": True,
            "commercial_use_allowed": False, "derivative_features_allowed": True,
            "raw_text_storage_allowed": True, "redistribution_allowed": False,
            "evidence_path": "licenses/CNINFO_RESEARCH_USE_BASIS.md",
            "evidence_sha256": sha256_file(basis),
            "license_notes": "Academic research only; no commercial license claim; no raw PDF redistribution",
        },
        {
            "license_id": "AKSHARE_SOFTWARE_MIT_1_18_78", "asset_type": "software", "provider": "AKShare",
            "effective_from": "2019-01-01", "effective_to": "", "research_use_allowed": True,
            "commercial_use_allowed": True, "derivative_features_allowed": True,
            "raw_text_storage_allowed": True, "redistribution_allowed": True,
            "evidence_path": "licenses/AKSHARE_MIT_LICENSE.txt", "evidence_sha256": sha256_file(akshare_copy),
            "license_notes": "MIT applies to AKShare software, not automatically to upstream content",
        },
    ])
    registry.to_csv(output_root / "license_registry.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()) and not (args.overwrite or args.resume):
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_root / "metadata_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(resolve_path(config["selected_universe_path"]))
    codes = [normalize_code(value) for value in universe["stock_code"]]
    stock_map_json = request_json("GET", STOCK_MAP_URL)
    stock_map = {row["code"]: row["orgId"] for row in stock_map_json["stockList"]}
    missing = [code for code in codes if code[:6] not in stock_map]
    if missing:
        raise ValueError(f"CNInfo stock mapping missing: {missing[:20]}")
    selected_by_code: dict[str, list[dict[str, Any]]] = {}
    pending: list[str] = []
    for code in codes:
        checkpoint = checkpoint_root / f"{code.replace('.', '_')}.json"
        if args.resume and checkpoint.exists():
            selected_by_code[code] = json.loads(checkpoint.read_text(encoding="utf-8"))
        else:
            pending.append(code)
    completed = len(selected_by_code)
    if completed:
        print(f"metadata resumed={completed}/{len(codes)}", flush=True)
    with ThreadPoolExecutor(max_workers=int(config.get("metadata_workers", 4))) as pool:
        futures = {
            pool.submit(fetch_selected_for_code, code, stock_map[code[:6]], config): code
            for code in pending
        }
        for future in as_completed(futures):
            code = futures[future]
            selected = future.result()
            selected_by_code[code] = selected
            checkpoint = checkpoint_root / f"{code.replace('.', '_')}.json"
            checkpoint.write_text(json.dumps(selected, ensure_ascii=False) + "\n", encoding="utf-8")
            completed += 1
            if completed % 10 == 0 or completed == len(codes):
                selected_count = sum(len(rows) for rows in selected_by_code.values())
                print(f"metadata {completed}/{len(codes)} selected={selected_count}", flush=True)
    metadata_rows = [item for code in codes for item in selected_by_code[code]]
    selected_metadata = pd.DataFrame(metadata_rows)
    selected_metadata.to_csv(output_root / "selected_announcement_metadata.csv", index=False, encoding="utf-8-sig")

    extracted = []
    with ThreadPoolExecutor(max_workers=int(config["download_workers"])) as pool:
        futures = {pool.submit(extract_pdf, row, int(config["max_pdf_pages"]), int(config["max_text_chars"])): row for row in metadata_rows}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                row = futures[future]
                published = pd.Timestamp(row["published_at"])
                event = {
                    "published_at": published.isoformat(), "source_name": "CNInfo via AKShare",
                    "source_item_id": str(row["announcementId"]),
                    "source_url": result["pdf_url"], "stock_code": row["project_stock_code"],
                    "title": str(row["announcementTitle"]), "body": result["body"],
                    "license_id": LICENSE_ID, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "source_pdf_sha256": result["pdf_sha256"], "source_pdf_bytes": result["pdf_bytes"],
                    "access_library": "AKShare 1.18.78", "upstream_source": "CNInfo",
                    "use_scope": "academic_research_only",
                }
                event["source_record_sha256"] = canonical_source_record_sha256(event)
                extracted.append(event)
            if index % 100 == 0:
                print(f"pdf {index}/{len(futures)} extracted={len(extracted)}", flush=True)
    events = pd.DataFrame(extracted).sort_values(["published_at", "stock_code", "source_item_id"])
    events.to_csv(output_root / "cninfo_announcements.csv", index=False, encoding="utf-8-sig")
    akshare_license = REPO_ROOT / ".venv-text/Lib/site-packages/akshare-1.18.78.dist-info/licenses/LICENSE"
    write_license_files(output_root, akshare_license)
    manifest = {
        "source": "CNInfo official disclosure PDFs discovered through AKShare-compatible interface",
        "akshare_version": "1.18.78", "start_date": config["start_date"], "end_date": config["end_date"],
        "stock_count": len(codes), "selected_metadata_rows": len(metadata_rows), "extracted_text_rows": len(events),
        "selection_policy": "one prioritized PDF per stock per calendar quarter, subject to PDF size and extraction limits",
        "files": [
            {"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(output_root.glob("*.csv"))
        ],
    }
    manifest["batch_sha256"] = stable_json_sha256(manifest)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
