"""Unit tests for Step 2: Article Deduplication."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.models.articles import ProcessedArticle
from src.models.config import Step2Config
from src.steps.step2_dedup import CachedArticlesDay, _load_cached_articles, run_step2
from src.utils.cache import CacheManager


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def cache_manager(temp_cache_dir: Path) -> CacheManager:
    """Create cache manager with temporary directory."""
    return CacheManager(cache_dir=temp_cache_dir)


@pytest.fixture
def step2_config() -> Step2Config:
    """Create Step 2 configuration."""
    return Step2Config(
        enabled=True,
    )


@pytest.fixture
def sample_articles() -> list[ProcessedArticle]:
    """Create sample processed articles."""
    return [
        ProcessedArticle(
            title="AI Model Released",
            url="https://example.com/ai-model",
            published_date=datetime(2024, 12, 22),
            content="New AI model",
            author="John Doe",
            feed_name="Test Feed",
            feed_priority=8,
            slug="ai-model-released-abc123",
            content_hash="hash1",
        ),
        ProcessedArticle(
            title="Machine Learning News",
            url="https://example.com/ml-news",
            published_date=datetime(2024, 12, 22),
            content="ML news",
            author="Jane Smith",
            feed_name="Test Feed",
            feed_priority=8,
            slug="machine-learning-news-def456",
            content_hash="hash2",
        ),
    ]


class TestStep2Deduplication:
    """Test Step 2: Article deduplication."""

    @pytest.mark.asyncio
    async def test_step2_with_empty_cache(
        self,
        step2_config: Step2Config,
        sample_articles: list[ProcessedArticle],
        cache_manager: CacheManager,
    ) -> None:
        """Test Step 2 with empty cache - all articles should be unique."""
        result = await run_step2(step2_config, sample_articles, cache_manager)

        assert result.success is True
        assert len(result.unique_articles) == 2
        assert result.stats.input_articles == 2
        assert result.stats.cache_articles == 0
        assert result.stats.duplicates_found == 0
        assert result.stats.unique_articles == 2
        assert result.stats.deduplication_rate == 0.0
        assert result.cache_updated is True

    @pytest.mark.asyncio
    async def test_step2_with_all_duplicates(
        self,
        step2_config: Step2Config,
        sample_articles: list[ProcessedArticle],
        cache_manager: CacheManager,
        temp_cache_dir: Path,
    ) -> None:
        """Test Step 2 when all articles are duplicates."""
        # Pre-populate cache with same articles
        articles_cache_dir = temp_cache_dir / "articles"
        articles_cache_dir.mkdir(parents=True, exist_ok=True)

        # Use recent date (2 days ago)
        cache_date = datetime.now() - timedelta(days=2)
        cache_file = articles_cache_dir / f"{cache_date:%Y-%m-%d}.json"
        cached_day = CachedArticlesDay(
            date=cache_date,
            articles=sample_articles,
            total_count=len(sample_articles),
        )
        cache_file.write_text(json.dumps(cached_day.model_dump(mode="json"), indent=2))

        # Run Step 2 with same articles
        result = await run_step2(step2_config, sample_articles, cache_manager)

        assert result.success is True
        assert len(result.unique_articles) == 0
        assert result.stats.input_articles == 2
        assert result.stats.cache_articles == 2
        assert result.stats.duplicates_found == 2
        assert result.stats.unique_articles == 0
        assert result.stats.deduplication_rate == 1.0  # 100% duplicates

    @pytest.mark.asyncio
    async def test_step2_with_partial_duplicates(
        self,
        step2_config: Step2Config,
        cache_manager: CacheManager,
        temp_cache_dir: Path,
    ) -> None:
        """Test Step 2 with some duplicates and some new articles."""
        # Use recent date (3 days ago)
        cache_date = datetime.now() - timedelta(days=3)

        # Create cached articles
        cached_articles = [
            ProcessedArticle(
                title="Cached Article 1",
                url="https://example.com/cached-1",
                published_date=cache_date,
                content="Cached",
                author="Author",
                feed_name="Test",
                feed_priority=5,
                slug="cached-article-1-xyz789",
                content_hash="hash_cached_1",
            ),
            ProcessedArticle(
                title="Cached Article 2",
                url="https://example.com/cached-2",
                published_date=cache_date,
                content="Cached",
                author="Author",
                feed_name="Test",
                feed_priority=5,
                slug="cached-article-2-xyz790",
                content_hash="hash_cached_2",
            ),
        ]

        # Save to cache
        articles_cache_dir = temp_cache_dir / "articles"
        articles_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = articles_cache_dir / f"{cache_date:%Y-%m-%d}.json"
        cached_day = CachedArticlesDay(
            date=cache_date,
            articles=cached_articles,
            total_count=len(cached_articles),
        )
        cache_file.write_text(json.dumps(cached_day.model_dump(mode="json"), indent=2))

        # New articles: 1 duplicate + 2 new
        new_articles = [
            cached_articles[0],  # Duplicate
            ProcessedArticle(
                title="New Article 1",
                url="https://example.com/new-1",
                published_date=datetime(2024, 12, 22),
                content="New",
                author="Author",
                feed_name="Test",
                feed_priority=5,
                slug="new-article-1-abc123",
                content_hash="hash_new_1",
            ),
            ProcessedArticle(
                title="New Article 2",
                url="https://example.com/new-2",
                published_date=datetime(2024, 12, 22),
                content="New",
                author="Author",
                feed_name="Test",
                feed_priority=5,
                slug="new-article-2-def456",
                content_hash="hash_new_2",
            ),
        ]

        result = await run_step2(step2_config, new_articles, cache_manager)

        assert result.success is True
        assert len(result.unique_articles) == 2
        assert result.stats.input_articles == 3
        assert result.stats.cache_articles == 2
        assert result.stats.duplicates_found == 1
        assert result.stats.unique_articles == 2
        assert abs(result.stats.deduplication_rate - 0.333) < 0.01  # ~33%

    @pytest.mark.asyncio
    async def test_step2_with_old_cache_files(
        self,
        step2_config: Step2Config,
        sample_articles: list[ProcessedArticle],
        cache_manager: CacheManager,
        temp_cache_dir: Path,
    ) -> None:
        """Test that cache files older than 10 days are ignored."""
        articles_cache_dir = temp_cache_dir / "articles"
        articles_cache_dir.mkdir(parents=True, exist_ok=True)

        # Create old cache file (15 days old)
        old_date = datetime.now() - timedelta(days=15)
        old_cache_file = articles_cache_dir / f"{old_date:%Y-%m-%d}.json"
        cached_day = CachedArticlesDay(
            date=old_date,
            articles=sample_articles,
            total_count=len(sample_articles),
        )
        old_cache_file.write_text(json.dumps(cached_day.model_dump(mode="json"), indent=2))

        # Run Step 2 - old cache should be ignored
        result = await run_step2(step2_config, sample_articles, cache_manager)

        assert result.success is True
        assert result.stats.cache_articles == 0
        assert result.stats.duplicates_found == 0
        assert len(result.unique_articles) == 2

    @pytest.mark.asyncio
    async def test_step2_with_corrupted_cache_file(
        self,
        step2_config: Step2Config,
        sample_articles: list[ProcessedArticle],
        cache_manager: CacheManager,
        temp_cache_dir: Path,
    ) -> None:
        """Test that corrupted cache files are skipped."""
        articles_cache_dir = temp_cache_dir / "articles"
        articles_cache_dir.mkdir(parents=True, exist_ok=True)

        # Use recent dates
        corrupted_date = datetime.now() - timedelta(days=2)
        valid_date = datetime.now() - timedelta(days=3)

        # Create corrupted cache file
        corrupted_file = articles_cache_dir / f"{corrupted_date:%Y-%m-%d}.json"
        corrupted_file.write_text("{ invalid json content }")

        # Create valid cache file
        valid_file = articles_cache_dir / f"{valid_date:%Y-%m-%d}.json"
        cached_day = CachedArticlesDay(
            date=valid_date,
            articles=[sample_articles[0]],
            total_count=1,
        )
        valid_file.write_text(json.dumps(cached_day.model_dump(mode="json"), indent=2))

        result = await run_step2(step2_config, sample_articles, cache_manager)

        assert result.success is True
        assert result.stats.cache_files_corrupted == 1
        assert result.stats.cache_files_loaded == 1
        assert result.stats.cache_articles == 1

    @pytest.mark.asyncio
    async def test_step2_creates_daily_cache_file(
        self,
        step2_config: Step2Config,
        sample_articles: list[ProcessedArticle],
        cache_manager: CacheManager,
        temp_cache_dir: Path,
    ) -> None:
        """Test that Step 2 creates daily cache file."""
        result = await run_step2(step2_config, sample_articles, cache_manager)

        assert result.success is True
        assert result.cache_updated is True

        # Check that cache file was created
        articles_cache_dir = temp_cache_dir / "articles"
        today_file = articles_cache_dir / f"{datetime.now():%Y-%m-%d}.json"
        assert today_file.exists()

        # Verify content
        cache_data = json.loads(today_file.read_text())
        cached_day = CachedArticlesDay(**cache_data)
        assert len(cached_day.articles) == 2
        assert cached_day.total_count == 2

    @pytest.mark.asyncio
    async def test_step2_with_empty_input(
        self, step2_config: Step2Config, cache_manager: CacheManager
    ) -> None:
        """Test Step 2 with no input articles."""
        result = await run_step2(step2_config, [], cache_manager)

        assert result.success is True
        assert len(result.unique_articles) == 0
        assert result.stats.input_articles == 0
        assert result.stats.deduplication_rate == 0.0


class TestCacheLoading:
    """Test cache loading functionality."""

    def test_load_cached_articles_empty_dir(self, temp_cache_dir: Path) -> None:
        """Test loading from empty cache directory."""
        articles_cache_dir = temp_cache_dir / "articles"
        articles_cache_dir.mkdir()

        cutoff = datetime.now() - timedelta(days=10)
        articles, stats = _load_cached_articles(articles_cache_dir, cutoff)

        assert len(articles) == 0
        assert stats["files_loaded"] == 0
        assert stats["files_corrupted"] == 0

    def test_load_cached_articles_nonexistent_dir(self, temp_cache_dir: Path) -> None:
        """Test loading from nonexistent directory."""
        articles_cache_dir = temp_cache_dir / "nonexistent"

        cutoff = datetime.now() - timedelta(days=10)
        articles, stats = _load_cached_articles(articles_cache_dir, cutoff)

        assert len(articles) == 0
        assert stats["files_loaded"] == 0
        assert stats["files_corrupted"] == 0
