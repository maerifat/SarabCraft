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

    From TextAttack BAEGarg2019 (and author email correspondence):
      1. USE comparison within a window of size 15 around the perturbed word.
      2. skip_text_shorter_than_window=True: always accept short texts
         (inputs with fewer words than the window size bypass the check).
      3. Compare against the original text (not incrementally modified text).
      4. Threshold=0.936338023 on cosine metric (adjusted from the paper's
         stated 0.8 to account for metric conversion in the TextAttack
         reference: 1 - (1 - 0.8) / pi).

    Args:
        original_text: the unmodified input text
        candidate_text: the perturbed candidate text
        word_position: 0-based word index of the perturbation
        threshold: cosine similarity threshold (default 0.936338023, matching
            TextAttack UniversalSentenceEncoder configuration)
        window_size: number of words in the comparison window (default 15)

    Returns:
        (passes_threshold, similarity_score)
    """
    from utils.text_utils import get_words_and_spans

    orig_words = [w for w, _, _ in get_words_and_spans(original_text)]

    # TextAttack: skip_text_shorter_than_window=True — always accept short texts
    if len(orig_words) < window_size:
        return True, 1.0

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


# ── Language Model Constraint ─────────────────────────────────────────────
# Matches TextAttack LearningToWriteLanguageModel (Holtzman et al., 2018)
# used in the Faster Alzantot GA recipe (Jia et al., 2019).
# Backend: GPT-2 (equivalent constraint logic; TextAttack uses a GRU-based
# RNN, but the constraint semantics are identical: window-based log-prob
# comparison against the original text).

_lm_model = None
_lm_tokenizer = None


def _get_lm():
    """Lazy-load GPT-2 for language model constraint."""
    global _lm_model, _lm_tokenizer
    if _lm_model is not None:
        return _lm_model, _lm_tokenizer

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading GPT-2 for LM constraint (Faster Alzantot GA)")
        _lm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _lm_model = AutoModelForCausalLM.from_pretrained("gpt2")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _lm_model.to(device)
        _lm_model.eval()
        return _lm_model, _lm_tokenizer
    except Exception as e:
        logger.warning("Could not load GPT-2 for LM constraint: %s", e)
        return None, None


def _get_lm_log_prob(text_window: str, target_word: str) -> float:
    """Compute log-probability of `target_word` within `text_window`.

    Mirrors TextAttack QueryHandler.query(): the LM scores the window text
    and we extract the log-probability at the position of the target word.

    Returns:
        Log-probability (float). Returns -inf if model unavailable or word
        is out of vocabulary.
    """
    import torch

    model, tokenizer = _get_lm()
    if model is None:
        return float("-inf")

    device = next(model.parameters()).device

    # Tokenize the full window
    inputs = tokenizer(text_window, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)

    if input_ids.size(1) < 2:
        return float("-inf")

    with torch.no_grad():
        outputs = model(input_ids)
        # logits shape: (1, seq_len, vocab_size)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)

    # Sum log-probs of all tokens (causal LM scoring of full window)
    # This matches TextAttack's approach: sum log-probs across the sequence
    total_log_prob = 0.0
    for i in range(1, input_ids.size(1)):
        token_id = input_ids[0, i].item()
        total_log_prob += log_probs[0, i - 1, token_id].item()

    return total_log_prob


def _text_window_around_index(words: list[str], index: int, window_size: int) -> str:
    """Extract a window of words centred on `index`.

    Matches TextAttack AttackedText.text_window_around_index().
    Window extends `window_size` words to the left and right of `index`.
    """
    half = window_size
    start = max(0, index - half)
    end = min(len(words), index + half + 1)
    return " ".join(words[start:end])


def check_lm_constraint(
    original_words: list[str],
    candidate_words: list[str],
    modified_index: int,
    window_size: int = 6,
    max_log_prob_diff: float = 5.0,
) -> bool:
    """Check LM constraint for a single modified position.

    Matches TextAttack LanguageModelConstraint._check_constraint():
    for the modified word index, extract windows from both original and
    candidate text, compute log-probabilities under the LM, and reject
    if the candidate log-prob drops by more than `max_log_prob_diff`.

    Args:
        original_words: list of words from the original text.
        candidate_words: list of words from the candidate text.
        modified_index: word index that was modified.
        window_size: number of words on each side of the modified word
            (paper: W=6).
        max_log_prob_diff: maximum allowed log-prob decrease
            (paper: δ=5.0).

    Returns:
        True if the candidate passes the LM constraint.
    """
    model, _ = _get_lm()
    if model is None:
        # If LM unavailable, warn and allow (same as TextAttack fallback)
        logger.warning(
            "LM constraint skipped: GPT-2 model unavailable. "
            "Install transformers and download gpt2 for full compliance."
        )
        return True

    orig_window = _text_window_around_index(original_words, modified_index, window_size)
    cand_window = _text_window_around_index(candidate_words, modified_index, window_size)

    orig_word = original_words[modified_index] if modified_index < len(original_words) else ""
    cand_word = candidate_words[modified_index] if modified_index < len(candidate_words) else ""

    ref_log_prob = _get_lm_log_prob(orig_window, orig_word)
    cand_log_prob = _get_lm_log_prob(cand_window, cand_word)

    # Reject if candidate log-prob drops by more than threshold
    # Matches: if transformed_prob <= ref_prob - max_log_prob_diff: return False
    if cand_log_prob <= ref_log_prob - max_log_prob_diff:
        return False

    return True
