import hashlib
import logging
import os

import numpy as np

_model = None
_hash_fallback_used = False
_hash_fallback_reason = None
_hash_fallback_warned = False

logger = logging.getLogger(__name__)


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        hf_token = os.environ.get("HF_TOKEN") or None
        _model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            local_files_only=hf_token is None,
            token=hf_token,
        )
    return _model


def embed(text: str) -> np.ndarray:
    global _hash_fallback_reason, _hash_fallback_used, _hash_fallback_warned
    try:
        return get_model().encode(text, convert_to_numpy=True).astype(np.float32)
    except Exception as exc:
        _hash_fallback_used = True
        _hash_fallback_reason = repr(exc)
        if not _hash_fallback_warned:
            logger.warning("Falling back to deterministic hash embeddings: %s", _hash_fallback_reason)
            _hash_fallback_warned = True
        return _hash_embed(text)


def diagnostics() -> dict:
    return {
        "hash_fallback_used": _hash_fallback_used,
        "hash_fallback_reason": _hash_fallback_reason,
    }


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
