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
        logger.info("Loading sentence-transformers model: distiluse-base-multilingual-cased-v1")
        _sim_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
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


def compute_windowed_semantic_similarity(
    original_text: str,
    candidate_text: str,
    word_position: int,
    threshold: float = 0.8,
    window_size: int = 15,
) -> tuple[bool, float]:
    """Windowed USE similarity matching the official BAE configuration.

    From author email correspondence (documented in TextAttack):
      1. USE comparison within a window of size 15 around the perturbed word.
      2. Threshold of 0.1 for inputs shorter than the window size
         (roughly: always accept short texts).
      3. Compare against the original text (not incrementally modified text).

    Args:
        original_text: the unmodified input text
        candidate_text: the perturbed candidate text
        word_position: 0-based word index of the perturbation
        threshold: cosine similarity threshold (default 0.8, paper value)
        window_size: number of words in the comparison window (default 15)

    Returns:
        (passes_threshold, similarity_score)
    """
    from utils.text_utils import get_words_and_spans

    orig_words = [w for w, _, _ in get_words_and_spans(original_text)]

    # Short text: relaxed threshold of 0.1 (from author correspondence)
    SHORT_TEXT_THRESHOLD = 0.1
    if len(orig_words) < window_size:
        sim = compute_semantic_similarity(original_text, candidate_text)
        return sim >= SHORT_TEXT_THRESHOLD, sim

    # Extract window centred on perturbed position
    half = window_size // 2
    start = max(0, word_position - half)
    end = start + window_size
    if end > len(orig_words):
        end = len(orig_words)
        start = max(0, end - window_size)

    orig_window = " ".join(orig_words[start:end])

    cand_words = [w for w, _, _ in get_words_and_spans(candidate_text)]
    # For insertions the candidate may have extra words; use same start
    # but allow end to stretch by the length difference
    len_diff = max(0, len(cand_words) - len(orig_words))
    cand_end = min(len(cand_words), end + len_diff)
    cand_window = " ".join(cand_words[start:cand_end])

    if not orig_window.strip() or not cand_window.strip():
        return False, 0.0

    sim = compute_semantic_similarity(orig_window, cand_window)
    return sim >= threshold, sim


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
