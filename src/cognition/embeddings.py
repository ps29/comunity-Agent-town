import hashlib

import numpy as np

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(text: str) -> np.ndarray:
    try:
        return get_model().encode(text, convert_to_numpy=True).astype(np.float32)
    except Exception:
        return _hash_embed(text)


def _hash_embed(text: str, dims: int = 384) -> np.ndarray:
    vector = np.zeros(dims, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dims
        vector[idx] += 1.0
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def embedding_to_blob(emb: np.ndarray) -> bytes:
    return emb.astype(np.float32).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
