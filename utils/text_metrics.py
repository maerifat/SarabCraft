"""
Post-attack metrics for text adversarial examples.

Mirrors utils/metrics.py in the main SarabCraft codebase.
"""

from utils.text_utils import get_words_and_spans, clean_word


def compute_perturbation_ratio(original: str, adversarial: str) -> float:
    """Percentage of words changed between original and adversarial text."""
    orig_words = [clean_word(w) for w, _, _ in get_words_and_spans(original)]
    adv_words = [clean_word(w) for w, _, _ in get_words_and_spans(adversarial)]

    if not orig_words:
        return 0.0

    changed = 0
    for i in range(min(len(orig_words), len(adv_words))):
        if orig_words[i] != adv_words[i]:
            changed += 1
    changed += abs(len(orig_words) - len(adv_words))
    return changed / len(orig_words)


def word_edit_distance(original: str, adversarial: str) -> int:
    """Levenshtein distance at the word level."""
    orig = original.split()
    adv = adversarial.split()
    m, n = len(orig), len(adv)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if orig[i - 1].lower() == adv[j - 1].lower():
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def char_edit_distance(original: str, adversarial: str) -> int:
    """Levenshtein distance at the character level."""
    m, n = len(original), len(adversarial)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if original[i - 1] == adversarial[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def compute_all_metrics(original: str, adversarial: str) -> dict:
    """Compute all text metrics."""
    from utils.text_constraints import compute_semantic_similarity

    return {
        "perturbation_ratio": round(compute_perturbation_ratio(original, adversarial), 4),
        "word_edit_distance": word_edit_distance(original, adversarial),
        "char_edit_distance": char_edit_distance(original, adversarial),
        "semantic_similarity": round(compute_semantic_similarity(original, adversarial), 4),
    }
