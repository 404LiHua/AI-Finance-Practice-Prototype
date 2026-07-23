"""Fit train-only TF-IDF, SVD and text clusters, then transform the full panel."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("data_pipeline/configs/weekly_a_share.json"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_root = Path(config["output"]["root"])
    if not data_root.is_absolute():
        data_root = project_root / data_root
    panel = pd.read_csv(data_root / "panel.csv.gz")
    panel["text_document"] = (
        panel["text_title"].fillna("").astype(str) + " " + panel["text_body"].fillna("").astype(str)
    ).str.strip()
    train_mask = panel["split"].eq("train") & panel["text_document"].ne("")
    train_documents = panel.loc[train_mask, "text_document"]
    if len(train_documents) < 2:
        raise ValueError("at least two training text events are required")
    options = config.get("text_features", {})
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=int(options.get("max_tfidf_features", 512)),
        min_df=1,
    )
    train_tfidf = vectorizer.fit_transform(train_documents)
    components = min(
        int(options.get("svd_components", 8)),
        train_tfidf.shape[0] - 1,
        train_tfidf.shape[1] - 1,
    )
    if components < 1:
        raise ValueError("insufficient text dimensions for SVD")
    svd = TruncatedSVD(n_components=components, random_state=20260723)
    train_reduced = svd.fit_transform(train_tfidf)
    cluster_count = min(int(options.get("clusters", 4)), len(train_documents))
    clusterer = KMeans(n_clusters=cluster_count, random_state=20260723, n_init=20)
    clusterer.fit(train_reduced)
    all_tfidf = vectorizer.transform(panel["text_document"])
    all_reduced = svd.transform(all_tfidf)
    nonempty = panel["text_document"].ne("").to_numpy()
    clusters = np.full(len(panel), -1, dtype=np.int16)
    clusters[nonempty] = clusterer.predict(all_reduced[nonempty])
    output = panel[["stock_code", "calendar_week_end", "split", "text_count"]].copy()
    output["text_cluster"] = clusters
    for index in range(components):
        output[f"text_svd_{index + 1:02d}"] = all_reduced[:, index]
    output.to_csv(data_root / "text_features.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    with (data_root / "text_feature_model.pkl").open("wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "svd": svd, "clusterer": clusterer}, handle)
    metadata = {
        "fit_scope": "train split non-empty text only",
        "training_text_rows": int(train_mask.sum()),
        "all_text_rows": int(panel["text_document"].ne("").sum()),
        "tfidf_vocabulary_size": len(vectorizer.vocabulary_),
        "svd_components": components,
        "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "clusters": cluster_count,
    }
    (data_root / "text_features_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
