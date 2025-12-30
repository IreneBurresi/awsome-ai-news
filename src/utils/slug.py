"""URL slug generation utilities.

Note: This module provides generic slug generation utilities.
The main pipeline uses a custom hash-based implementation in
src/steps/step1_ingestion.py for better collision resistance.
"""

from slugify import slugify


def generate_slug(text: str, max_length: int = 50) -> str:
    """Generate a URL-safe slug from text.

    Args:
        text: Input text (e.g., article title)
        max_length: Maximum slug length

    Returns:
        URL-safe slug

    Examples:
        >>> generate_slug("OpenAI Releases GPT-5")
        'openai-releases-gpt-5'
        >>> generate_slug("This is a very long title that needs to be truncated", max_length=20)
        'this-is-a-very-long'
    """
    slug = slugify(text, max_length=max_length, word_boundary=True, separator="-")
    return slug or "untitled"


def generate_unique_slug(text: str, existing_slugs: set[str], max_length: int = 50) -> str:
    """
    Generate a unique URL-safe slug by appending a counter if needed.

    Args:
        text: Input text (e.g., article title)
        existing_slugs: Set of already used slugs
        max_length: Maximum slug length

    Returns:
        Unique URL-safe slug

    Examples:
        >>> generate_unique_slug("Test Article", set())
        'test-article'
        >>> generate_unique_slug("Test Article", {"test-article"})
        'test-article-2'
    """
    base_slug = generate_slug(text, max_length=max_length)

    if base_slug not in existing_slugs:
        return base_slug

    # Append counter to make unique
    counter = 2
    while True:
        unique_slug = f"{base_slug}-{counter}"
        if unique_slug not in existing_slugs:
            return unique_slug
        counter += 1
