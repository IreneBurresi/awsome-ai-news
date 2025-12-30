"""Unit tests for Step 4: Multi-day News Deduplication."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.config import Step4Config
from src.models.news import NewsCluster, NewsDeduplicationPair
from src.steps.step4_multi_dedup import (
    _load_cached_news,
    _merge_duplicate_news,
    _save_news_to_cache,
    run_step4,
)
from src.utils.cache import CacheManager

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def step4_config() -> Step4Config:
    """Create Step 4 configuration for testing."""
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


@pytest.fixture
def cache_manager(tmp_path: Path) -> CacheManager:
    """Create a temporary cache manager for testing."""
    return CacheManager(cache_dir=tmp_path)


@pytest.fixture
def sample_news_today() -> list[NewsCluster]:
    """Create sample today's news clusters."""
    return [
        NewsCluster(
            news_id="news-today-0001234",
            title="GPT-5 Released by OpenAI",
            summary="OpenAI announces the release of GPT-5 with major improvements.",
            article_slugs=["openai-gpt5-release"],
            article_count=1,
            main_topic="model release",
            keywords=["GPT-5", "OpenAI", "release"],
            created_at=datetime.utcnow(),
        ),
        NewsCluster(
            news_id="news-today-0005678",
            title="EU AI Regulation Update",
            summary="European Union updates its AI regulation framework.",
            article_slugs=["eu-ai-regulation-update"],
            article_count=1,
            main_topic="policy",
            keywords=["EU", "regulation", "AI"],
            created_at=datetime.utcnow(),
        ),
    ]


@pytest.fixture
def sample_cached_news() -> list[NewsCluster]:
    """Create sample cached news from previous days."""
    yesterday = datetime.utcnow() - timedelta(days=1)
    two_days_ago = datetime.utcnow() - timedelta(days=2)

    return [
        NewsCluster(
            news_id="news-cache-0001111",
            title="OpenAI Unveils GPT-5",  # Similar to today's news
            summary="OpenAI has unveiled GPT-5, the latest and most advanced version of their large language model with significant improvements in reasoning and capabilities.",
            article_slugs=["openai-unveils-gpt5"],
            article_count=1,
            main_topic="model release",
            keywords=["GPT-5", "OpenAI", "model"],
            created_at=yesterday,
        ),
        NewsCluster(
            news_id="news-cache-0002222",
            title="Google Gemini 2.0 Update",
            summary="Google releases a major update to Gemini 2.0 with enhanced multimodal capabilities and improved performance across various benchmarks.",
            article_slugs=["google-gemini-2-update"],
            article_count=1,
            main_topic="model release",
            keywords=["Google", "Gemini", "update"],
            created_at=two_days_ago,
        ),
    ]


# ============================================================================
# Helper Function Tests
# ============================================================================


def test_save_news_to_cache(
    cache_manager: CacheManager, sample_news_today: list[NewsCluster]
) -> None:
    """Test saving news to cache creates correct directory and file."""
    _save_news_to_cache(cache_manager, sample_news_today)

    # Check that directory was created
    news_dir = Path(cache_manager.cache_dir) / "news"
    assert news_dir.exists()
    assert news_dir.is_dir()

    # Check that file was created with today's date
    today_str = datetime.now().strftime("%Y-%m-%d")
    news_file = news_dir / f"news_{today_str}.json"
    assert news_file.exists()

    # Load and verify content
    loaded_news = cache_manager.load(f"news/news_{today_str}", NewsCluster)
    assert len(loaded_news) == 2
    assert loaded_news[0].news_id == "news-today-0001234"


def test_save_news_to_cache_empty_list(cache_manager: CacheManager) -> None:
    """Test saving empty news list."""
    _save_news_to_cache(cache_manager, [])

    today_str = datetime.now().strftime("%Y-%m-%d")
    loaded_news = cache_manager.load(f"news/news_{today_str}", NewsCluster)
    assert loaded_news == []


def test_load_cached_news_no_directory(cache_manager: CacheManager) -> None:
    """Test loading cached news when directory doesn't exist."""
    result = _load_cached_news(cache_manager, lookback_days=3)
    assert result == []


def test_load_cached_news_with_files(
    cache_manager: CacheManager, sample_cached_news: list[NewsCluster]
) -> None:
    """Test loading cached news from multiple dated files."""
    # Create news directory and save some files
    news_dir = Path(cache_manager.cache_dir) / "news"
    news_dir.mkdir(parents=True)

    # Save news from yesterday and 2 days ago
    yesterday = datetime.now() - timedelta(days=1)
    two_days_ago = datetime.now() - timedelta(days=2)

    cache_manager.save(f"news/news_{yesterday.strftime('%Y-%m-%d')}", [sample_cached_news[0]])
    cache_manager.save(f"news/news_{two_days_ago.strftime('%Y-%m-%d')}", [sample_cached_news[1]])

    # Load cached news
    result = _load_cached_news(cache_manager, lookback_days=3)

    assert len(result) == 2
    # Files are loaded in sorted order (oldest first due to filename sorting)
    news_ids = {news.news_id for news in result}
    assert news_ids == {"news-cache-0001111", "news-cache-0002222"}


def test_load_cached_news_filters_old_files(cache_manager: CacheManager) -> None:
    """Test that old files outside lookback window are filtered out."""
    news_dir = Path(cache_manager.cache_dir) / "news"
    news_dir.mkdir(parents=True)

    # Create files: 2 days ago (should load), 5 days ago (should skip)
    two_days_ago = datetime.now() - timedelta(days=2)
    five_days_ago = datetime.now() - timedelta(days=5)

    news_recent = NewsCluster(
        news_id="news-recent-0001",
        title="Recent News Article",
        summary="Recent news summary with enough characters to pass validation requirements for this test case.",
        article_slugs=["recent"],
        article_count=1,
        main_topic="test",
        keywords=["recent"],
        created_at=two_days_ago,
    )

    news_old = NewsCluster(
        news_id="news-old-0001",
        title="Old News Article",
        summary="Old news summary with enough characters to pass validation requirements for this test case.",
        article_slugs=["old"],
        article_count=1,
        main_topic="test",
        keywords=["old"],
        created_at=five_days_ago,
    )

    cache_manager.save(f"news/news_{two_days_ago.strftime('%Y-%m-%d')}", [news_recent])
    cache_manager.save(f"news/news_{five_days_ago.strftime('%Y-%m-%d')}", [news_old])

    # Load with 3-day lookback
    result = _load_cached_news(cache_manager, lookback_days=3)

    # Should only get recent news, not old news
    assert len(result) == 1
    assert result[0].news_id == "news-recent-0001"


def test_merge_duplicate_news_basic(
    sample_news_today: list[NewsCluster], sample_cached_news: list[NewsCluster]
) -> None:
    """Test basic news merging with one duplicate pair."""
    # Create a duplicate pair (today's GPT-5 news duplicates cached GPT-5 news)
    duplicate_pairs = [
        NewsDeduplicationPair(
            news_today_id="news-today-0001234",
            news_cached_id="news-cache-0001111",
            similarity_score=0.95,
            should_merge=True,
            merge_reason="Both about GPT-5 release",
        )
    ]

    result = _merge_duplicate_news(sample_news_today, sample_cached_news, duplicate_pairs)

    # Result should have:
    # - 2 cached news (GPT-5 merged with today's + Gemini)
    # - 1 today news that wasn't merged (EU regulation)
    # Total: 3 news
    assert len(result) == 3

    # Find the merged GPT-5 news (should have both article slugs)
    gpt5_news = [n for n in result if "gpt" in n.title.lower()]
    assert len(gpt5_news) >= 1

    # Check that one of them has merged slugs
    merged = [n for n in gpt5_news if len(n.article_slugs) > 1]
    assert len(merged) == 1
    assert set(merged[0].article_slugs) == {"openai-gpt5-release", "openai-unveils-gpt5"}
    assert merged[0].article_count == 2
    assert merged[0].updated_at is not None


def test_merge_duplicate_news_keeps_larger_cluster(sample_cached_news: list[NewsCluster]) -> None:
    """Test that merge keeps the news with more articles as base."""
    # Create today news with more articles
    today_news_large = NewsCluster(
        news_id="news-today-large",
        title="GPT-5 Major Release",
        summary="Major release of GPT-5 with groundbreaking capabilities in reasoning, multimodal understanding, and advanced AI applications.",
        article_slugs=["gpt5-1", "gpt5-2", "gpt5-3"],
        article_count=3,
        main_topic="model release",
        keywords=["GPT-5"],
        created_at=datetime.utcnow(),
    )

    duplicate_pairs = [
        NewsDeduplicationPair(
            news_today_id="news-today-large",
            news_cached_id="news-cache-0001111",
            similarity_score=0.9,
            should_merge=True,
            merge_reason="Same GPT-5 release",
        )
    ]

    result = _merge_duplicate_news([today_news_large], sample_cached_news, duplicate_pairs)

    # Find the merged news
    merged = [n for n in result if "gpt" in n.title.lower() and len(n.article_slugs) > 1]
    assert len(merged) == 1

    # Should keep the today news as base (it has more articles)
    assert merged[0].news_id == "news-today-large"
    assert merged[0].article_count == 4  # 3 + 1
    assert "gpt5-1" in merged[0].article_slugs
    assert "openai-unveils-gpt5" in merged[0].article_slugs


def test_merge_duplicate_news_no_duplicates(
    sample_news_today: list[NewsCluster], sample_cached_news: list[NewsCluster]
) -> None:
    """Test merge with no duplicate pairs."""
    result = _merge_duplicate_news(sample_news_today, sample_cached_news, [])

    # Result should have all cached + all today news
    assert len(result) == 4  # 2 cached + 2 today


def test_merge_duplicate_news_invalid_ids(
    sample_news_today: list[NewsCluster], sample_cached_news: list[NewsCluster]
) -> None:
    """Test merge handles invalid news IDs gracefully."""
    duplicate_pairs = [
        NewsDeduplicationPair(
            news_today_id="invalid-id",
            news_cached_id="invalid-id-2",
            similarity_score=0.9,
            should_merge=True,
            merge_reason="Test invalid",
        )
    ]

    # Should not crash, just skip invalid pairs
    result = _merge_duplicate_news(sample_news_today, sample_cached_news, duplicate_pairs)
    assert len(result) == 4  # All news preserved


# ============================================================================
# Main Function Tests
# ============================================================================


@pytest.mark.asyncio
async def test_run_step4_disabled(
    step4_config: Step4Config, cache_manager: CacheManager, sample_news_today: list[NewsCluster]
) -> None:
    """Test Step 4 when disabled in config."""
    step4_config.enabled = False

    result = await run_step4(step4_config, sample_news_today, cache_manager)

    assert result.success is True
    assert result.unique_news == sample_news_today
    assert result.news_before_dedup == 2
    assert result.news_after_dedup == 2
    assert result.duplicates_found == 0
    assert result.api_calls == 0


@pytest.mark.asyncio
async def test_run_step4_empty_input(
    step4_config: Step4Config, cache_manager: CacheManager
) -> None:
    """Test Step 4 with empty input."""
    result = await run_step4(step4_config, [], cache_manager, api_key="test-key")

    assert result.success is True
    assert result.unique_news == []
    assert result.news_before_dedup == 0
    assert result.news_after_dedup == 0
    assert result.duplicates_found == 0


@pytest.mark.asyncio
async def test_run_step4_no_cached_news(
    step4_config: Step4Config, cache_manager: CacheManager, sample_news_today: list[NewsCluster]
) -> None:
    """Test Step 4 with no cached news (first run)."""
    result = await run_step4(step4_config, sample_news_today, cache_manager, api_key="test-key")

    assert result.success is True
    assert len(result.unique_news) == 2
    assert result.news_before_dedup == 2
    assert result.news_after_dedup == 2
    assert result.duplicates_found == 0
    assert result.api_calls == 0

    # Verify news was saved to cache
    today_str = datetime.now().strftime("%Y-%m-%d")
    loaded = cache_manager.load(f"news/news_{today_str}", NewsCluster)
    assert len(loaded) == 2


@pytest.mark.asyncio
async def test_run_step4_no_api_key_with_fallback(
    step4_config: Step4Config,
    cache_manager: CacheManager,
    sample_news_today: list[NewsCluster],
    sample_cached_news: list[NewsCluster],
) -> None:
    """Test Step 4 without API key using fallback."""
    # Add some cached news
    _save_news_to_cache(cache_manager, sample_cached_news)

    result = await run_step4(step4_config, sample_news_today, cache_manager, api_key=None)

    assert result.success is True
    assert result.fallback_used is True
    assert len(result.unique_news) == 2
    assert result.api_calls == 0
    assert "No API key provided" in result.errors


@pytest.mark.asyncio
async def test_run_step4_no_api_key_without_fallback(
    step4_config: Step4Config,
    cache_manager: CacheManager,
    sample_news_today: list[NewsCluster],
    sample_cached_news: list[NewsCluster],
) -> None:
    """Test Step 4 without API key and fallback disabled."""
    step4_config.fallback_to_no_merge = False

    # Add some cached news
    _save_news_to_cache(cache_manager, sample_cached_news)

    result = await run_step4(step4_config, sample_news_today, cache_manager, api_key=None)

    assert result.success is False
    assert len(result.unique_news) == 0
    assert "No API key provided and fallback disabled" in result.errors


@pytest.mark.asyncio
async def test_run_step4_successful_deduplication(
    step4_config: Step4Config,
    cache_manager: CacheManager,
    sample_news_today: list[NewsCluster],
    sample_cached_news: list[NewsCluster],
) -> None:
    """Test successful deduplication with Gemini API."""
    # Save cached news
    _save_news_to_cache(cache_manager, sample_cached_news)

    # Mock the Gemini API call
    mock_response = MagicMock()
    mock_response.text = """{
        "duplicate_pairs": [
            {
                "news_today_id": "news-today-0001234",
                "news_cached_id": "news-cache-0001111",
                "similarity_score": 0.95,
                "should_merge": true,
                "merge_reason": "Both about GPT-5 release from OpenAI"
            }
        ],
        "total_comparisons": 4,
        "rationale": "Found one duplicate pair about GPT-5 release"
    }"""

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step4(
            step4_config, sample_news_today, cache_manager, api_key="test-api-key"
        )

    assert result.success is True
    assert result.duplicates_found == 1
    assert result.news_merged == 1
    assert result.api_calls == 1
    assert result.fallback_used is False

    # Check that merged news has combined article slugs
    merged_news = [n for n in result.unique_news if len(n.article_slugs) > 1]
    assert len(merged_news) >= 1


@pytest.mark.asyncio
async def test_run_step4_api_failure_with_fallback(
    step4_config: Step4Config,
    cache_manager: CacheManager,
    sample_news_today: list[NewsCluster],
    sample_cached_news: list[NewsCluster],
) -> None:
    """Test Step 4 handles API failure with fallback."""
    # Save cached news
    _save_news_to_cache(cache_manager, sample_cached_news)

    # Mock API failure
    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.side_effect = Exception("API timeout")

        result = await run_step4(
            step4_config, sample_news_today, cache_manager, api_key="test-api-key"
        )

    assert result.success is True
    assert result.fallback_used is True
    assert result.api_failures == 1
    assert len(result.errors) > 0
    assert "Gemini API call failed" in result.errors[0]


@pytest.mark.asyncio
async def test_run_step4_api_failure_without_fallback(
    step4_config: Step4Config,
    cache_manager: CacheManager,
    sample_news_today: list[NewsCluster],
    sample_cached_news: list[NewsCluster],
) -> None:
    """Test Step 4 fails when API fails and fallback disabled."""
    step4_config.fallback_to_no_merge = False

    # Save cached news
    _save_news_to_cache(cache_manager, sample_cached_news)

    # Mock API failure
    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.side_effect = Exception("API timeout")

        result = await run_step4(
            step4_config, sample_news_today, cache_manager, api_key="test-api-key"
        )

    assert result.success is False
    assert len(result.unique_news) == 0


@pytest.mark.asyncio
async def test_run_step4_saves_deduplicated_news(
    step4_config: Step4Config,
    cache_manager: CacheManager,
    sample_news_today: list[NewsCluster],
    sample_cached_news: list[NewsCluster],
) -> None:
    """Test that Step 4 saves deduplicated news to cache."""
    # Save cached news
    _save_news_to_cache(cache_manager, sample_cached_news)

    # Mock successful deduplication
    mock_response = MagicMock()
    mock_response.text = """{
        "duplicate_pairs": [],
        "total_comparisons": 4,
        "rationale": "No duplicates found"
    }"""

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step4(
            step4_config, sample_news_today, cache_manager, api_key="test-api-key"
        )

    # Verify news was saved
    today_str = datetime.now().strftime("%Y-%m-%d")
    loaded = cache_manager.load(f"news/news_{today_str}", NewsCluster)
    assert len(loaded) > 0


@pytest.mark.asyncio
async def test_run_step4_critical_error_handling(
    step4_config: Step4Config, cache_manager: CacheManager, sample_news_today: list[NewsCluster]
) -> None:
    """Test Step 4 handles unexpected critical errors."""
    # Mock cache_manager.save to raise an exception
    with patch.object(cache_manager, "save", side_effect=Exception("Disk full")):
        result = await run_step4(step4_config, sample_news_today, cache_manager, api_key="test-key")

    assert result.success is False
    assert len(result.errors) > 0
    assert "Step 4 failed critically" in result.errors[0]
