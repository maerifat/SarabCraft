"""
Text tokenization, word span tracking, and reconstruction utilities.
"""

import re
from typing import Optional

# Simple stopword set (no external dependency needed)
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "am", "i", "me",
    "my", "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "it", "its", "they", "them", "their", "this", "that", "these", "those",
    "and", "but", "or", "nor", "not", "no", "so", "if", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "as", "into", "about", "up",
    "out", "off", "over", "then", "than", "too", "very", "just",
})

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
    """Check if word is a stopword (case-insensitive)."""
    return word.lower().strip(".,!?;:'\"()[]{}") in STOPWORDS


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
