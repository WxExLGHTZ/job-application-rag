"""In-memory embedding index and semantic retrieval for profile chunks."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from copilot.ingest import build_chunks


MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_DEFAULT_INDEX: "ProfileIndex | None" = None


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence-transformers model once per Python process."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Convert texts into embedding vectors with the configured model.

    Args:
        texts: Text snippets that should be represented as vectors.

    Returns:
        A NumPy array with one embedding vector per input text.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    model = get_embedding_model()
    # Embedding: Text wird in einen Zahlenvektor uebersetzt, damit semantisch
    # aehnliche Texte im Vektorraum nahe beieinander liegen.
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to length 1 for readable cosine similarity."""
    if vectors.size == 0:
        return vectors

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


class ProfileIndex:
    """A small in-memory vector index for profile chunks."""

    def __init__(self, chunks: list[dict[str, Any]], embeddings: np.ndarray) -> None:
        """Store chunks and their normalized embedding vectors."""
        self.chunks = chunks
        self.embeddings = normalize_vectors(embeddings)

    @classmethod
    def from_profile(cls, profile_dir: str | Path = "profile") -> "ProfileIndex":
        """Build an in-memory index from all profile PDFs in a directory."""
        chunks = build_chunks(profile_dir)
        texts = [chunk["text"] for chunk in chunks]
        # Hier werden alle Chunks einmalig embeddet und als NumPy-Matrix im
        # Speicher gehalten. Fuer v1 reicht das ohne Vektor-Datenbank.
        embeddings = embed_texts(texts)
        return cls(chunks=chunks, embeddings=embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top_k chunks most similar to the query.

        Args:
            query: Search question or requirement text.
            top_k: Number of best matching chunks to return.

        Returns:
            Matching chunks with filename, text, and cosine score.
        """
        if top_k <= 0 or not self.chunks:
            return []

        query_embedding = embed_texts([query])
        query_vector = normalize_vectors(query_embedding)[0]

        # Cosine Similarity: Weil alle Vektoren auf Laenge 1 normalisiert sind,
        # ist der Cosine Score einfach das Skalarprodukt der Vektoren.
        scores = self.embeddings @ query_vector

        # Top-k bedeutet: Wir sortieren alle Chunks nach Score und nehmen nur
        # die k besten Treffer als Kontext fuer die naechste Stufe.
        result_count = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:result_count]

        results: list[dict[str, Any]] = []
        for index in top_indices:
            chunk = self.chunks[int(index)]
            results.append(
                {
                    "filename": chunk["filename"],
                    "text": chunk["text"],
                    "score": float(scores[int(index)]),
                }
            )

        return results


def build_index(profile_dir: str | Path = "profile") -> ProfileIndex:
    """Build a fresh in-memory profile index."""
    return ProfileIndex.from_profile(profile_dir)


def retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve matching chunks from the default profile index."""
    global _DEFAULT_INDEX

    if _DEFAULT_INDEX is None:
        _DEFAULT_INDEX = build_index()

    return _DEFAULT_INDEX.retrieve(query=query, top_k=top_k)
