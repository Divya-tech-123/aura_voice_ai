"""NLP preprocessing and Bag-of-Words feature extraction for AURA.

This module provides foundational Natural Language Processing (NLP) functions:
1. clean_and_tokenize: Splits raw text into normalized words (tokens).
2. build_vocabulary: Creates a unique, sorted list of known words.
3. bag_of_words: Converts a tokenized sentence into a fixed-length numerical vector.
"""

import re
from typing import List, Set, Iterable


def clean_and_tokenize(text: str) -> List[str]:
    """Normalize text and split it into word tokens.

    Steps:
    1. Lowercase the input so 'Hello' and 'hello' are treated identically.
    2. Extract alphanumeric word tokens using regex, discarding punctuation.

    Args:
        text: Raw input sentence (e.g., "Hello, Aura! What's the time?").

    Returns:
        A list of normalized word tokens (e.g., ["hello", "aura", "what", "s", "the", "time"]).
    """
    if not text:
        return []
    
    # Lowercase text and find all word character sequences (letters, numbers, underscores)
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


def build_vocabulary(tokenized_sentences: Iterable[List[str]]) -> List[str]:
    """Construct a sorted vocabulary of unique tokens from a corpus of sentences.

    A vocabulary is the master dictionary of all distinct words the model knows.
    Sorting ensures deterministic ordering so each word always receives the same index.

    Args:
        tokenized_sentences: Iterable of token lists (e.g., [["hello", "aura"], ["what", "time"]]).

    Returns:
        A sorted list of unique words in the vocabulary.
    """
    vocab_set: Set[str] = set()
    for sentence in tokenized_sentences:
        for token in sentence:
            vocab_set.add(token)
    
    return sorted(list(vocab_set))


def bag_of_words(tokenized_sentence: List[str], vocabulary: List[str]) -> List[float]:
    """Transform a tokenized sentence into a Bag-of-Words (BoW) numerical vector.

    How it works:
    - The vector has the same length as the vocabulary.
    - Each position corresponds to a specific word in the vocabulary.
    - If that word is present in the tokenized sentence, the value is 1.0 (or count).
    - If absent, the value is 0.0.

    Example:
        Vocabulary: ["aura", "hello", "time", "weather"]
        Sentence:   ["hello", "aura"]
        BoW Vector: [ 1.0,    1.0,     0.0,    0.0     ]

    Args:
        tokenized_sentence: List of tokens from the input text.
        vocabulary: Master list of unique known words.

    Returns:
        A list of 1.0 / 0.0 floats representing word presence in the vocabulary.
    """
    # Create a lookup set for fast O(1) membership checks
    sentence_words = set(tokenized_sentence)
    
    # Generate binary flag for each vocabulary word
    vector = [1.0 if word in sentence_words else 0.0 for word in vocabulary]
    return vector
