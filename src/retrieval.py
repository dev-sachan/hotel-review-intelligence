"""Semantic retrieval over the review corpus.

Because the corpus reduces to 87 unique sentences, the entire semantic index is 87
vectors — no FAISS, no approximate search needed. A query is embedded once and compared
to all 87 with a plain matrix multiply (cosine, since vectors are normalized). We then map
matching sentences back to their review instances, with optional hotel / aspect filters.

Used for (a) the app's free-text evidence search and (b) picking the most on-topic
supporting quotes for a recommendation.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config

EMB_PATH = os.path.join(config.PROCESSED_DIR, "sentence_embeddings.npz")

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def build_sentence_embeddings(labels: pd.DataFrame) -> None:
    """Embed the 87 unique sentences once and cache to disk."""
    emb = _get_embedder()
    sents = labels["sentence_text"].tolist()
    vecs = emb.encode(sents, normalize_embeddings=True, convert_to_numpy=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    np.savez(EMB_PATH, sentences=np.array(sents, dtype=object), vectors=vecs.astype(np.float32))


def load_sentence_embeddings() -> tuple[list[str], np.ndarray]:
    data = np.load(EMB_PATH, allow_pickle=True)
    return list(data["sentences"]), data["vectors"]


def search(query: str, aspect_instances: pd.DataFrame, k: int = 10,
           hotel_id: str | None = None, aspect: str | None = None) -> pd.DataFrame:
    """Return the top-k matching review instances for a free-text query."""
    sents, vecs = load_sentence_embeddings()
    qv = _get_embedder().encode([query], normalize_embeddings=True)[0]
    sims = vecs @ qv
    order = np.argsort(-sims)

    sim_by_sentence = {sents[i]: float(sims[i]) for i in range(len(sents))}
    pool = aspect_instances
    if hotel_id:
        pool = pool[pool["hotel_id"] == hotel_id]
    if aspect:
        pool = pool[pool["aspect"] == aspect]

    pool = pool.copy()
    pool["similarity"] = pool["sentence_text"].map(sim_by_sentence).fillna(-1.0)
    pool = pool[pool["similarity"] > 0]
    pool = pool.sort_values(["similarity", "recency_weight"], ascending=False)
    # dedupe by sentence so results are varied, keep the most recent instance of each
    pool = pool.drop_duplicates(subset="sentence_text")
    return pool.head(k)[["review_id", "hotel_id", "hotel_name", "sentence_text", "aspect",
                         "polarity", "review_date", "verified", "similarity"]]
