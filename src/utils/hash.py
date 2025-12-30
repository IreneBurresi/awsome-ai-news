"""Hashing utilities for deduplication."""

import hashlib
from typing import Any


def generate_content_hash(*fields: Any) -> str:
    """
    Generate a hash from multiple fields for deduplication.

    Args:
        *fields: Variable number of fields to hash (title, url, content, etc.)

    Returns:
        SHA256 hash as hex string

    Examples:
        >>> generate_content_hash("Article Title", "https://example.com")
        'a1b2c3...'
        >>> generate_content_hash("Title", "URL", "Content")
        'x7y8z9...'
    """
    # Combine all fields into a single string
    combined = "|".join(str(field).strip().lower() for field in fields if field)

    # Generate SHA256 hash
    hash_obj = hashlib.sha256(combined.encode("utf-8"))
    return hash_obj.hexdigest()


def normalize_url(url: str) -> str:
    """
    Normalize URL for comparison.

    Removes common variations:
    - Protocol (http/https)
    - www prefix
    - Trailing slashes
    - Query parameters (optional)
    - Fragment identifiers (#)

    Args:
        url: URL to normalize

    Returns:
        Normalized URL

    Examples:
        >>> normalize_url("https://www.example.com/page/")
        'example.com/page'
        >>> normalize_url("http://example.com/article?utm_source=feed#section")
        'example.com/article'
    """
    normalized = url.lower().strip()

    # Remove protocol
    for protocol in ["https://", "http://"]:
        if normalized.startswith(protocol):
            normalized = normalized[len(protocol) :]
            break

    # Remove www prefix
    if normalized.startswith("www."):
        normalized = normalized[4:]

    # Remove fragment
    if "#" in normalized:
        normalized = normalized.split("#")[0]

    # Remove query parameters (for better deduplication)
    if "?" in normalized:
        normalized = normalized.split("?")[0]

    # Remove trailing slash
    normalized = normalized.rstrip("/")

    return normalized


def calculate_similarity(hash1: str, hash2: str) -> float:
    """
    Calculate similarity between two hashes.

    For exact hash matching, returns 1.0 if equal, 0.0 if different.
    This is a simple exact match for now; can be extended with
    fuzzy matching if needed.

    Args:
        hash1: First hash
        hash2: Second hash

    Returns:
        Similarity score (0.0 to 1.0)

    Examples:
        >>> calculate_similarity("abc123", "abc123")
        1.0
        >>> calculate_similarity("abc123", "def456")
        0.0
    """
    return 1.0 if hash1 == hash2 else 0.0
