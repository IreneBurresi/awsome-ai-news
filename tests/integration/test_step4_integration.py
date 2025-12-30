"""Integration tests for Step 4: Multi-day News Deduplication."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.config import Step4Config
from src.models.news import NewsCluster
from src.steps.step4_multi_dedup import run_step4
from src.utils.cache import CacheManager


@pytest.fixture
def temp_cache(tmp_path: Path) -> CacheManager:
    """Create temporary cache for integration tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return CacheManager(cache_dir=cache_dir)


@pytest.fixture
def step4_config() -> Step4Config:
    """Standard Step 4 configuration."""
    return Step4Config(
        enabled=True,
        llm_model="gemini-2.5-flash-lite",
        lookback_days=3,
        similarity_threshold=0.85,
        timeout_seconds=30,
        retry_attempts=3,
        temperature=0.3,
        fallback_to_no_merge=True,
    )


def create_sample_news(news_id: str, title: str, slug: str) -> NewsCluster:
    """Helper to create sample news cluster."""
    return NewsCluster(
        news_id=news_id,
        title=title,
        summary=f"This is a summary about {title} with enough characters to pass validation requirements.",
        article_slugs=[slug],
        article_count=1,
        main_topic="test",
        keywords=["test", "integration"],
        created_at=datetime.utcnow(),
    )


@pytest.mark.asyncio
async def test_step4_first_run_no_cache(temp_cache: CacheManager, step4_config: Step4Config) -> None:
    """Test Step 4 on first run with no cached news."""
    today_news = [
        create_sample_news("news-001", "AI Model Released", "ai-model-released"),
        create_sample_news("news-002", "New Regulation Announced", "new-regulation"),
    ]

    result = await run_step4(step4_config, today_news, temp_cache, api_key="test-key")

    # Should succeed without deduplication
    assert result.success is True
    assert result.news_after_dedup == 2
    assert result.duplicates_found == 0
    assert result.api_calls == 0

    # Verify cache was created
    news_cache_dir = Path(temp_cache.cache_dir) / "news"
    assert news_cache_dir.exists()

    today_str = datetime.now().strftime("%Y-%m-%d")
    cache_file = news_cache_dir / f"news_{today_str}.json"
    assert cache_file.exists()


@pytest.mark.asyncio
async def test_step4_second_run_with_deduplication(temp_cache: CacheManager, step4_config: Step4Config) -> None:
    """Test Step 4 on second run with cached news and deduplication."""
    # First run - create cache
    yesterday_news = [
        create_sample_news("news-cached-001", "OpenAI Releases GPT-5", "openai-gpt5"),
    ]

    yesterday = datetime.now() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    # Manually create cache directory and save
    news_cache_dir = Path(temp_cache.cache_dir) / "news"
    news_cache_dir.mkdir(parents=True, exist_ok=True)
    temp_cache.save(f"news/news_{yesterday_str}", yesterday_news)

    # Second run - today's news (similar to cached)
    today_news = [
        create_sample_news("news-today-001", "GPT-5 Launch by OpenAI", "gpt5-launch-openai"),
    ]

    # Mock Gemini API to find duplicate
    mock_response = MagicMock()
    mock_response.text = """{
        "duplicate_pairs": [
            {
                "news_today_id": "news-today-001",
                "news_cached_id": "news-cached-001",
                "similarity_score": 0.92,
                "should_merge": true,
                "merge_reason": "Both about GPT-5 release from OpenAI"
            }
        ],
        "total_comparisons": 1,
        "rationale": "Found semantic duplicate"
    }"""

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step4(step4_config, today_news, temp_cache, api_key="test-key")

    # Should find and merge duplicate
    assert result.success is True
    assert result.duplicates_found == 1
    assert result.news_merged == 1
    assert result.api_calls == 1

    # Result should have merged news
    assert len(result.unique_news) == 1
    merged_news = result.unique_news[0]
    assert len(merged_news.article_slugs) == 2
    assert "openai-gpt5" in merged_news.article_slugs
    assert "gpt5-launch-openai" in merged_news.article_slugs


@pytest.mark.asyncio
async def test_step4_multi_day_cache_loading(temp_cache: CacheManager, step4_config: Step4Config) -> None:
    """Test that Step 4 loads cache from multiple days within lookback window."""
    news_cache_dir = Path(temp_cache.cache_dir) / "news"
    news_cache_dir.mkdir(parents=True, exist_ok=True)

    # Create news for last 2 days (within 3-day lookback)
    for days_ago in range(1, 3):
        date = datetime.now() - timedelta(days=days_ago)
        date_str = date.strftime("%Y-%m-%d")

        news = [
            create_sample_news(
                f"news-day{days_ago}-001",
                f"News from {days_ago} days ago",
                f"news-day-{days_ago}"
            )
        ]
        temp_cache.save(f"news/news_{date_str}", news)

    # Today's news
    today_news = [
        create_sample_news("news-today-001", "Today's News", "today-news"),
    ]

    # Mock API to return no duplicates
    mock_response = MagicMock()
    mock_response.text = """{
        "duplicate_pairs": [],
        "total_comparisons": 3,
        "rationale": "No semantic duplicates found"
    }"""

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step4(step4_config, today_news, temp_cache, api_key="test-key")

    # Should load 2 days of cache and process
    assert result.success is True
    assert result.api_calls == 1
    # Result = 2 cached + 1 today = 3 total
    assert result.news_after_dedup == 3


@pytest.mark.asyncio
async def test_step4_filters_old_cache(temp_cache: CacheManager, step4_config: Step4Config) -> None:
    """Test that Step 4 filters cache older than lookback window."""
    news_cache_dir = Path(temp_cache.cache_dir) / "news"
    news_cache_dir.mkdir(parents=True, exist_ok=True)

    # Create old cache (5 days ago, outside 3-day window)
    old_date = datetime.now() - timedelta(days=5)
    old_date_str = old_date.strftime("%Y-%m-%d")
    old_news = [create_sample_news("news-old-001", "Old News Article", "old-news")]
    temp_cache.save(f"news/news_{old_date_str}", old_news)

    # Create recent cache (1 day ago, inside window)
    recent_date = datetime.now() - timedelta(days=1)
    recent_date_str = recent_date.strftime("%Y-%m-%d")
    recent_news = [create_sample_news("news-recent-001", "Recent News Item", "recent-news")]
    temp_cache.save(f"news/news_{recent_date_str}", recent_news)

    today_news = [create_sample_news("news-today-001", "Today News Item", "today-news")]

    # Mock API
    mock_response = MagicMock()
    mock_response.text = """{
        "duplicate_pairs": [],
        "total_comparisons": 1,
        "rationale": "No duplicates"
    }"""

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step4(step4_config, today_news, temp_cache, api_key="test-key")

    # Should only load recent cache (1 day), not old (5 days)
    # Result = 1 recent + 1 today = 2 total
    assert result.news_after_dedup == 2
