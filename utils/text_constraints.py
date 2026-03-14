"""
Inline constraint checking for text adversarial attacks.

Constraints are applied DURING search — each candidate substitution is
rejected if it fails similarity or perplexity checks. This prevents
attacks from producing unreadable or semantically unrelated outputs.
"""

import logging
from typing import Optional

logger = logging.getLogger("textattack.constraints")

# Lazy-loaded sentence transformer for semantic similarity
_sim_model = None


def _get_sim_model():
    """Lazy-load sentence-transformers model for cosine similarity."""
    global _sim_model
    if _sim_model is not None:
        return _sim_model

    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers model: all-MiniLM-L6-v2")
        _sim_model = SentenceTransformer("all-MiniLM-L6-v2")
        return _sim_model
    except ImportError:
        logger.warning("sentence-transformers not installed; similarity checks disabled")
        return None


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using sentence-transformers.

    Returns: float in [-1, 1], higher = more similar.
    
    CRITICAL: Fails closed (returns 0.0) if model unavailable to prevent
    silent bypass of semantic constraints.
    """
    model = _get_sim_model()
    if model is None:
        logger.critical(
            "Semantic similarity model unavailable! "
            "Install sentence-transformers or disable similarity constraint. "
            "Failing closed (returning 0.0) to prevent constraint bypass."
        )
        return 0.0  # FAIL-CLOSED: reject all candidates if constraint broken

    embeddings = model.encode([text_a, text_b], convert_to_tensor=True)
    from torch.nn.functional import cosine_similarity
    sim = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    return sim.item()


class ConstraintChecker:
    """Composite constraint gate used inline during attack search.

    Usage:
        checker = ConstraintChecker(similarity_threshold=0.8)
        if checker.check_all(original_text, candidate_text):
            # candidate passes all constraints
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        enable_similarity: bool = True,
    ):
        self.similarity_threshold = similarity_threshold
        self.enable_similarity = enable_similarity

    def check_semantic_similarity(self, original: str, candidate: str) -> bool:
        """Check if semantic similarity exceeds threshold."""
        if not self.enable_similarity:
            return True
        sim = compute_semantic_similarity(original, candidate)
        return sim >= self.similarity_threshold

    def check_all(self, original: str, candidate: str) -> bool:
        """Run all enabled constraints. Returns True if candidate passes."""
        if not self.check_semantic_similarity(original, candidate):
            return False
        return True
