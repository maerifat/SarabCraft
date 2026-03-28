"""
Text tokenization, word span tracking, and reconstruction utilities.
"""

import re
from typing import Optional

# NLTK English stopwords (179 words) — matches TextAttack StopwordModification.
# Loaded dynamically from NLTK when available; hardcoded fallback below.
_NLTK_STOPWORDS = None

_FALLBACK_STOPWORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
    "hers", "herself", "it", "it's", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "that'll", "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
    "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "don't", "should", "should've", "now", "d", "ll", "m", "o", "re", "ve",
    "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't",
    "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven",
    "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn",
    "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't",
    "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn",
    "wouldn't",
})


def _get_stopwords() -> frozenset:
    global _NLTK_STOPWORDS
    if _NLTK_STOPWORDS is not None:
        return _NLTK_STOPWORDS
    try:
        from nltk.corpus import stopwords
        _NLTK_STOPWORDS = frozenset(stopwords.words("english"))
    except (ImportError, LookupError):
        _NLTK_STOPWORDS = _FALLBACK_STOPWORDS
    return _NLTK_STOPWORDS

# Basic POS patterns (avoids NLTK dependency for simple cases)
_NOUN_SUFFIXES = {"tion", "ment", "ness", "ity", "ism", "ist", "ence", "ance", "er", "or"}
_ADJ_SUFFIXES = {"ful", "less", "ous", "ive", "able", "ible", "al", "ial", "ical"}
_VERB_SUFFIXES = {"ing", "ize", "ise", "ify", "ate", "ened"}
_ADV_SUFFIXES = {"ly"}


def get_words_and_spans(text: str) -> list[tuple[str, int, int]]:
    """Tokenize text into words with character-level spans.

    Returns: [(word, start_char, end_char), ...]
    """
    results = []
    for match in re.finditer(r"\S+", text):
        results.append((match.group(), match.start(), match.end()))
    return results


def replace_word_at(text: str, position: int, new_word: str) -> str:
    """Replace a word at the given position index, preserving spacing."""
    spans = get_words_and_spans(text)
    if position < 0 or position >= len(spans):
        return text
    _, start, end = spans[position]
    return text[:start] + new_word + text[end:]


def replace_words_at(text: str, replacements: dict[int, str]) -> str:
    """Replace multiple words at given positions, preserving spacing.

    replacements: {position_index: new_word}
    """
    spans = get_words_and_spans(text)
    # Process in reverse order to preserve character offsets
    result = text
    for pos in sorted(replacements.keys(), reverse=True):
        if 0 <= pos < len(spans):
            _, start, end = spans[pos]
            result = result[:start] + replacements[pos] + result[end:]
    return result


def is_stopword(word: str) -> bool:
    """Check if word is a stopword (case-insensitive).

    Uses NLTK English stopwords when available (matches TextAttack
    StopwordModification), with a 179-word hardcoded fallback.
    """
    return word.lower().strip(".,!?;:'\"()[]{}") in _get_stopwords()


def simple_pos_tag(word: str) -> str:
    """Basic POS tagging via suffix heuristics. Returns: 'noun', 'verb', 'adj', 'adv', 'other'."""
    w = word.lower().strip(".,!?;:'\"()[]{}").rstrip("s")
    for suffix in _ADV_SUFFIXES:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return "adv"
    for suffix in _ADJ_SUFFIXES:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return "adj"
    for suffix in _VERB_SUFFIXES:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return "verb"
    for suffix in _NOUN_SUFFIXES:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return "noun"
    return "other"


def pos_tag_words(words: list[str]) -> list[str]:
    """POS-tag a list of words. Tries NLTK first, falls back to heuristics."""
    try:
        import nltk
        tagged = nltk.pos_tag(words)
        return [tag for _, tag in tagged]
    except (ImportError, LookupError):
        return [simple_pos_tag(w) for w in words]


def clean_word(word: str) -> str:
    """Strip punctuation for comparison purposes."""
    return word.strip(".,!?;:'\"()[]{}").lower()
