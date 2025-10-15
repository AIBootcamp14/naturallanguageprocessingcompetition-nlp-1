"""
Sub-cluster the topics previously labeled as "Human interest and relationships".

Steps:
1. Load analysis/topic_cluster_assignments.csv and apply manual relabel map
2. Filter to human-interest topics (after adjustments)
3. Encode with multilingual-e5-base
4. UMAP reduction (same params)
5. HDBSCAN clustering with smaller min_cluster_size
6. Extract TF-IDF keywords / assign zero-shot labels from a focused candidate list
7. Save assignments and summary under analysis/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
import umap
from hdbscan import HDBSCAN
from transformers import pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parents[1]
ASSIGN_PATH = BASE_DIR / "analysis" / "topic_cluster_assignments.csv"
OUTPUT_ASSIGN = BASE_DIR / "analysis" / "human_interest_subclusters.csv"
OUTPUT_SUMMARY = BASE_DIR / "analysis" / "human_interest_subclusters_summary.csv"
OUTPUT_META = BASE_DIR / "analysis" / "human_interest_subclusters_metadata.json"

ADJUST_MAP = {
    11: "Lifestyle and leisure",
    31: "Lifestyle and leisure",
    22: "Business and finance",
}

HUMAN_LABEL = "Human interest and relationships"

SUB_LABELS = [
    "Family relationships",
    "Friendship support",
    "Romantic relationship",
    "Conflict resolution",
    "Event or party planning",
    "Daily chit-chat",
    "Emotional support",
    "Advice and guidance",
    "Workplace relationship",
    "Parenting and children",
]


def encode_texts(texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
    model = SentenceTransformer("intfloat/multilingual-e5-base", device=DEVICE)
    prefixed = [f"passage: {text}" for text in texts]
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def reduce_embeddings(embeddings: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    return reducer.fit_transform(embeddings)


def cluster_embeddings(embeddings: np.ndarray) -> HDBSCAN:
    clusterer = HDBSCAN(
        min_cluster_size=40,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    clusterer.fit(embeddings)
    return clusterer


def extract_keywords(texts: List[str], labels: np.ndarray, top_k: int = 15) -> dict[int, List[str]]:
    tfidf = TfidfVectorizer(
        token_pattern=r"(?u)\b[\w가-힣]+\b",
        lowercase=False,
        max_features=6000,
    )
    tfidf_matrix = tfidf.fit_transform(texts)
    tfidf_matrix = normalize(tfidf_matrix, norm="l1", axis=1)
    feature_names = np.array(tfidf.get_feature_names_out())

    keywords: dict[int, List[str]] = {}
    for label in np.unique(labels):
        mask = labels == label
        if label == -1 or mask.sum() == 0:
            continue
        scores = tfidf_matrix[mask].sum(axis=0)
        top_idx = np.asarray(scores).ravel().argsort()[::-1][:top_k]
        keywords[label] = feature_names[top_idx].tolist()
    return keywords


def zero_shot(descriptions: dict[int, str]) -> dict[int, dict[str, float]]:
    clf = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        device=0 if DEVICE == "cuda" else -1,
    )
    scores: dict[int, dict[str, float]] = {}
    for label, desc in tqdm(descriptions.items(), desc="Subcluster zero-shot labeling"):
        result = clf(desc, SUB_LABELS)
        scores[label] = dict(zip(result["labels"], result["scores"]))
    return scores


def main() -> None:
    print(f"Loading assignments from {ASSIGN_PATH}")
    df = pd.read_csv(ASSIGN_PATH)

    df["adjusted_label"] = df["zero_shot_label"]
    for cid, label in ADJUST_MAP.items():
        df.loc[df["cluster_id"] == cid, "adjusted_label"] = label

    human_df = df[df["adjusted_label"] == HUMAN_LABEL].copy().reset_index(drop=True)
    print(f"Human-interest topics: {len(human_df)}")

    embeddings = encode_texts(human_df["clean_text"].tolist())
    reduced = reduce_embeddings(embeddings)

    clusterer = cluster_embeddings(reduced)
    labels = clusterer.labels_.copy()

    # Reassign noise to nearest centroid to avoid -1 labels
    if (labels == -1).any():
        unique_labels = [lab for lab in np.unique(labels) if lab != -1]
        centroids = np.vstack(
            [reduced[labels == lab].mean(axis=0) for lab in unique_labels]
        )
        noise_idx = np.where(labels == -1)[0]
        noise_vec = reduced[noise_idx]
        dists = np.linalg.norm(noise_vec[:, None, :] - centroids[None, :, :], axis=2)
        nearest = dists.argmin(axis=1)
        for idx, cid in zip(noise_idx, nearest):
            labels[idx] = unique_labels[cid]

    unique_subclusters = sorted(set(labels))
    print(f"Subclusters formed: {len(unique_subclusters)}")

    keywords = extract_keywords(human_df["clean_text"].tolist(), labels)

    descriptions = {}
    human_df["subcluster"] = labels
    for label in unique_subclusters:
        subset = human_df[human_df["subcluster"] == label]
        top_topics = ", ".join(subset["topic"].head(5))
        top_keywords = ", ".join(keywords.get(label, [])[:10])
        descriptions[label] = f"Topics: {top_topics} || Keywords: {top_keywords}"

    score_map = zero_shot(descriptions)

    human_df["subcluster_label"] = human_df["subcluster"].map(
        lambda cid: max(score_map.get(cid, {}), key=score_map.get(cid, {}).get)
        if score_map.get(cid)
        else None
    )
    human_df["subcluster_score"] = human_df["subcluster"].map(
        lambda cid: max(score_map.get(cid, {}).values())
        if score_map.get(cid)
        else np.nan
    )
    human_df["subcluster_keywords"] = human_df["subcluster"].map(keywords.get)

    sub_sizes = human_df.groupby("subcluster").size().rename("count").reset_index()
    summary_rows = []
    for _, row in sub_sizes.iterrows():
        cid = row["subcluster"]
        summary_rows.append(
            {
                "subcluster": int(cid),
                "count": int(row["count"]),
                "top_keywords": keywords.get(cid, []),
                "label": human_df.loc[human_df["subcluster"] == cid, "subcluster_label"].iloc[0],
                "score": human_df.loc[human_df["subcluster"] == cid, "subcluster_score"].iloc[0],
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("count", ascending=False)

    human_df.to_csv(OUTPUT_ASSIGN, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)
    OUTPUT_META.write_text(
        json.dumps(
            {
                "total_topics": int(len(human_df)),
                "subclusters": int(len(summary_df)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"Saved subcluster assignments → {OUTPUT_ASSIGN}")
    print(f"Saved subcluster summary → {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
