"""
Topic clustering pipeline for high-level domain grouping.

Steps
-----
1. Aggregate multiple dialogue/summary examples per unique topic.
2. Clean text, create TF-IDF friendly token strings.
3. Generate multilingual sentence embeddings (intfloat/multilingual-e5-base).
4. Reduce dimensionality with UMAP and cluster using HDBSCAN.
5. Extract representative keywords with TF-IDF.
6. Assign coarse domains via zero-shot classification (XLM-R large XNLI).
7. Persist assignments, cluster summary, and metadata under ``analysis/``.

Usage
-----
    python scripts/topic_clustering.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
import umap
from transformers import pipeline


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "train.csv"
OUTPUT_DIR = BASE_DIR / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# IPTC/IAB inspired macro categories (16 total)
DOMAIN_LABELS = [
    "Arts, culture and entertainment",
    "Business and finance",
    "Crime, law and justice",
    "Disaster and accident",
    "Education",
    "Environment",
    "Health and medical",
    "Human interest and relationships",
    "Labour and employment",
    "Lifestyle and leisure",
    "Politics and government",
    "Religion and belief",
    "Science and technology",
    "Shopping and retail",
    "Sports",
    "Travel and transportation",
    "Weather",
]


@dataclass
class ClusterArtifacts:
    topic_assignments: pd.DataFrame
    cluster_summary: pd.DataFrame
    metadata: dict


def aggregate_topics(df: pd.DataFrame, samples_per_topic: int = 3) -> pd.DataFrame:
    """Aggregate dialogue and summary texts per unique topic."""
    grouped = (
        df.groupby("topic")
        .apply(
            lambda g: pd.Series(
                {
                    "dialogue": g["dialogue"].head(samples_per_topic).tolist(),
                    "summary": g["summary"].head(samples_per_topic).tolist(),
                }
            )
        )
        .reset_index()
    )
    grouped = grouped.fillna("")
    grouped["aggregated_text"] = grouped.apply(
        lambda row: " ".join(
            [
                str(row["topic"]),
                " ".join(row["dialogue"]),
                " ".join(row["summary"]),
            ]
        )
        .strip(),
        axis=1,
    )
    return grouped


def clean_text(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_texts(df: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas(desc="Cleaning text")
    df["clean_text"] = df["aggregated_text"].progress_apply(clean_text)
    df["token_string"] = df["clean_text"]
    return df


def encode_texts(texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
    """Encode texts using multilingual E5-base embeddings."""
    model = SentenceTransformer("intfloat/multilingual-e5-base", device=DEVICE)
    prefixed_texts = [f"passage: {text}" for text in texts]
    embeddings = model.encode(
        prefixed_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def reduce_embeddings(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    n_components: int = 15,
    min_dist: float = 0.0,
    random_state: int = 42,
) -> np.ndarray:
    """Dimensionality reduction via UMAP (cosine metric)."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        verbose=True,
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 25,
    min_samples: int | None = 10,
) -> HDBSCAN:
    """Cluster embeddings using HDBSCAN."""
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    clusterer.fit(embeddings)
    return clusterer


def extract_cluster_keywords(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    top_k: int = 15,
) -> dict[int, List[str]]:
    """Compute top TF-IDF keywords per cluster."""
    tfidf = TfidfVectorizer(
        token_pattern=r"(?u)\b[\w가-힣]+\b",
        lowercase=False,
        max_features=8000,
    )
    tfidf_matrix = tfidf.fit_transform(df["token_string"])
    tfidf_matrix = normalize(tfidf_matrix, norm="l1", axis=1)

    keywords = {}
    feature_names = np.array(tfidf.get_feature_names_out())
    for label in np.unique(cluster_labels):
        mask = cluster_labels == label
        if label == -1 or mask.sum() == 0:
            continue
        topic_scores = tfidf_matrix[mask].sum(axis=0)
        top_indices = np.asarray(topic_scores).ravel().argsort()[::-1][:top_k]
        keywords[label] = feature_names[top_indices].tolist()
    return keywords


def zero_shot_label_clusters(
    descriptions: dict[int, str],
    candidate_labels: List[str],
) -> dict[int, dict[str, float]]:
    """Apply zero-shot classification to map cluster descriptions to macro labels."""
    zshot = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        device=0 if DEVICE == "cuda" else -1,
    )
    cluster_scores: dict[int, dict[str, float]] = {}
    for label, description in tqdm(descriptions.items(), desc="Zero-shot labeling"):
        result = zshot(description, candidate_labels)
        scores = dict(zip(result["labels"], result["scores"]))
        cluster_scores[label] = scores
    return cluster_scores


def build_cluster_descriptions(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_keywords: dict[int, List[str]],
    top_n_topics: int = 5,
) -> dict[int, str]:
    """Create descriptive text snippets per cluster for downstream labeling."""
    descriptions = {}
    df = df.copy()
    df["cluster"] = cluster_labels
    for label in sorted(set(cluster_labels)):
        if label == -1:
            continue
        subset = df[df["cluster"] == label]
        topic_counts = Counter(subset["topic"])
        top_topics = ", ".join(
            [topic for topic, _ in topic_counts.most_common(top_n_topics)]
        )
        keywords = ", ".join(cluster_keywords.get(label, [])[:10])
        descriptions[label] = (
            f"대표 토픽: {top_topics} || 주요 키워드: {keywords}"
        )
    return descriptions


def assemble_outputs(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_keywords: dict[int, List[str]],
    cluster_scores: dict[int, dict[str, float]],
) -> ClusterArtifacts:
    """Compile final dataframes for export."""
    df = df.copy()
    df["cluster_id"] = cluster_labels

    cluster_size_map = (
        df[df["cluster_id"] != -1]
        .groupby("cluster_id")
        .size()
        .to_dict()
    )

    cluster_records = []
    for label, keywords in cluster_keywords.items():
        scores = cluster_scores.get(label, {})
        top_label = max(scores, key=scores.get) if scores else None
        cluster_records.append(
            {
                "cluster_id": label,
                "cluster_size": cluster_size_map.get(label, 0),
                "top_keywords": keywords,
                "zero_shot_label": top_label,
                "zero_shot_score": scores.get(top_label, np.nan) if top_label else np.nan,
                "all_scores": scores,
            }
        )

    cluster_summary = pd.DataFrame(cluster_records).sort_values(
        "cluster_size", ascending=False
    )

    topic_assignments = df[
        ["topic", "cluster_id", "aggregated_text", "clean_text"]
    ].copy()
    topic_assignments["cluster_size"] = topic_assignments["cluster_id"].map(
        cluster_size_map
    )
    topic_assignments["top_keywords"] = topic_assignments["cluster_id"].map(
        lambda cid: cluster_keywords.get(cid, [])
    )
    topic_assignments["zero_shot_label"] = topic_assignments["cluster_id"].map(
        lambda cid: (
            max(cluster_scores.get(cid, {}), key=cluster_scores.get(cid, {}).get)
            if cluster_scores.get(cid)
            else None
        )
    )
    topic_assignments["zero_shot_score"] = topic_assignments["cluster_id"].map(
        lambda cid: (
            max(cluster_scores.get(cid, {}).values())
            if cluster_scores.get(cid)
            else np.nan
        )
    )

    metadata = {
        "n_topics": int(len(df)),
        "n_clusters": int(cluster_summary["cluster_id"].nunique()),
        "noise_topics": int((df["cluster_id"] == -1).sum()),
    }

    return ClusterArtifacts(topic_assignments, cluster_summary, metadata)


def main() -> None:
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    topics_df = aggregate_topics(df)
    print(f"Aggregated into {len(topics_df)} unique topics")

    topics_df = preprocess_texts(topics_df)

    print("Encoding texts with multilingual E5-base...")
    embeddings = encode_texts(topics_df["clean_text"].tolist())

    print("Reducing dimensionality with UMAP...")
    reduced_embeddings = reduce_embeddings(embeddings)

    print("Clustering with HDBSCAN...")
    clusterer = cluster_embeddings(reduced_embeddings)
    cluster_labels = clusterer.labels_.copy()

    if (cluster_labels == -1).any():
        unique_labels = [label for label in np.unique(cluster_labels) if label != -1]
        centroids = np.vstack(
            [reduced_embeddings[cluster_labels == label].mean(axis=0) for label in unique_labels]
        )
        noise_indices = np.where(cluster_labels == -1)[0]
        noise_vectors = reduced_embeddings[noise_indices]
        dists = np.linalg.norm(noise_vectors[:, None, :] - centroids[None, :, :], axis=2)
        nearest_centroid_idx = dists.argmin(axis=1)
        for idx, centroid_idx in zip(noise_indices, nearest_centroid_idx):
            cluster_labels[idx] = unique_labels[centroid_idx]

    print(
        f"Identified {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters "
        f"with {np.sum(cluster_labels == -1)} noise points"
    )

    cluster_keywords = extract_cluster_keywords(topics_df, cluster_labels)
    descriptions = build_cluster_descriptions(topics_df, cluster_labels, cluster_keywords)
    cluster_scores = zero_shot_label_clusters(descriptions, DOMAIN_LABELS)

    artifacts = assemble_outputs(
        topics_df, cluster_labels, cluster_keywords, cluster_scores
    )

    assignments_path = OUTPUT_DIR / "topic_cluster_assignments.csv"
    summary_path = OUTPUT_DIR / "topic_cluster_summary.csv"
    metadata_path = OUTPUT_DIR / "topic_cluster_metadata.json"

    print(f"Saving topic assignments to {assignments_path}")
    artifacts.topic_assignments.to_csv(assignments_path, index=False)

    print(f"Saving cluster summary to {summary_path}")
    artifacts.cluster_summary.to_csv(summary_path, index=False)

    print(f"Saving metadata to {metadata_path}")
    metadata_path.write_text(json.dumps(artifacts.metadata, ensure_ascii=False, indent=2))

    print("Done.")


if __name__ == "__main__":
    main()
