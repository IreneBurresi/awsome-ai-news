"""Shared pytest fixtures and configuration."""

from datetime import datetime
from pathlib import Path

import pytest

from src.models.articles import RawArticle


@pytest.fixture
def sample_raw_article() -> RawArticle:
    """Create a sample raw article for testing."""
    return RawArticle(
        title="Sample AI Article",
        url="https://example.com/sample-article",
        published_date=datetime(2024, 1, 15, 10, 30),
        content="This is a sample article about artificial intelligence.",
        author="John Doe",
        feed_name="Tech News",
        feed_priority=8,
    )


@pytest.fixture
def sample_raw_articles() -> list[RawArticle]:
    """Create multiple sample raw articles for testing."""
    return [
        RawArticle(
            title="First AI Article",
            url="https://example.com/first",
            published_date=datetime(2024, 1, 15, 10, 0),
            content="Content about AI.",
            feed_name="Tech News",
            feed_priority=8,
        ),
        RawArticle(
            title="Second ML Article",
            url="https://example.com/second",
            published_date=datetime(2024, 1, 15, 11, 0),
            content="Content about machine learning.",
            feed_name="AI Weekly",
            feed_priority=9,
        ),
        RawArticle(
            title="Third DL Article",
            url="https://example.com/third",
            published_date=datetime(2024, 1, 15, 12, 0),
            content="Content about deep learning.",
            feed_name="Tech News",
            feed_priority=8,
        ),
    ]


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def mock_cache_dir(tmp_path: Path) -> Path:
    """Create a mock cache directory for tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir
