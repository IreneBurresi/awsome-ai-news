"""Integration tests for Step 2 with full pipeline simulation."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.models.articles import ProcessedArticle
from src.models.config import Step2Config
from src.steps.step2_dedup import CachedArticlesDay, run_step2
from src.utils.cache import CacheManager


@pytest.fixture
def pipeline_cache_dir(tmp_path: Path) -> Path:
    """Create pipeline-like cache directory structure."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def step2_config() -> Step2Config:
    """Create realistic Step 2 configuration."""
    return Step2Config(
        enabled=True,
    )


@pytest.fixture
def realistic_articles() -> list[ProcessedArticle]:
    """Create realistic articles simulating Step 1 output."""
    base_date = datetime.now()

    return [
        ProcessedArticle(
            title="OpenAI Releases GPT-5 with Enhanced Reasoning",
            url="https://openai.com/blog/gpt-5",
            published_date=base_date,
            content="OpenAI has announced GPT-5, featuring improved reasoning capabilities...",
            author="OpenAI Team",
            feed_name="OpenAI Blog",
            feed_priority=10,
            slug="openai-releases-gpt-5-with-enhanced-reasoning-abc123",
            content_hash="hash_gpt5",
        ),
        ProcessedArticle(
            title="Google DeepMind Achieves Breakthrough in Protein Folding",
            url="https://deepmind.google/research/protein-folding",
            published_date=base_date - timedelta(hours=2),
            content="DeepMind's AlphaFold 3 has achieved unprecedented accuracy...",
            author="DeepMind Research",
            feed_name="DeepMind Blog",
            feed_priority=9,
            slug="google-deepmind-achieves-breakthrough-in-protein-folding-def456",
            content_hash="hash_alphafold",
        ),
        ProcessedArticle(
            title="Meta Introduces New AI Safety Framework",
            url="https://ai.meta.com/blog/safety-framework",
            published_date=base_date - timedelta(hours=4),
            content="Meta has unveiled a comprehensive AI safety framework...",
            author="Meta AI",
            feed_name="Meta AI Blog",
            feed_priority=8,
            slug="meta-introduces-new-ai-safety-framework-ghi789",
            content_hash="hash_meta_safety",
        ),
    ]


class TestStep2Integration:
    """Integration tests for Step 2 deduplication."""

    @pytest.mark.asyncio
    async def test_full_deduplication_workflow(
        self,
        step2_config: Step2Config,
        realistic_articles: list[ProcessedArticle],
        pipeline_cache_dir: Path,
    ) -> None:
        """Test complete deduplication workflow with realistic data."""
        cache_manager = CacheManager(cache_dir=pipeline_cache_dir)

        # First run: no cache, all articles unique
        result1 = await run_step2(step2_config, realistic_articles, cache_manager)

        assert result1.success is True
        assert len(result1.unique_articles) == 3
        assert result1.stats.input_articles == 3
        assert result1.stats.cache_articles == 0
        assert result1.stats.duplicates_found == 0
        assert result1.stats.unique_articles == 3
        assert result1.cache_updated is True

        # Verify cache file created
        articles_cache_dir = pipeline_cache_dir / "articles"
        today_file = articles_cache_dir / f"{datetime.now():%Y-%m-%d}.json"
        assert today_file.exists()

        # Second run: same articles, all duplicates
        result2 = await run_step2(step2_config, realistic_articles, cache_manager)

        assert result2.success is True
        assert len(result2.unique_articles) == 0
        assert result2.stats.cache_articles == 3
        assert result2.stats.duplicates_found == 3
        assert result2.stats.deduplication_rate == 1.0

    @pytest.mark.asyncio
    async def test_multiple_days_accumulation(
        self,
        step2_config: Step2Config,
        pipeline_cache_dir: Path,
    ) -> None:
        """Test cache accumulation over multiple days."""
        cache_manager = CacheManager(cache_dir=pipeline_cache_dir)
        articles_cache_dir = pipeline_cache_dir / "articles"
        articles_cache_dir.mkdir(parents=True, exist_ok=True)

        # Simulate 5 days of articles
        for day_offset in range(5, 0, -1):
            cache_date = datetime.now() - timedelta(days=day_offset)

            articles = [
                ProcessedArticle(
                    title=f"Article from {day_offset} days ago",
                    url=f"https://example.com/article-day{day_offset}",
                    published_date=cache_date,
                    content=f"Content from day {day_offset}",
                    author="Test Author",
                    feed_name="Test Feed",
                    feed_priority=5,
                    slug=f"article-from-{day_offset}-days-ago-hash{day_offset}",
                    content_hash=f"hash_day{day_offset}",
                )
            ]

            # Save to cache
            cache_file = articles_cache_dir / f"{cache_date:%Y-%m-%d}.json"
            cached_day = CachedArticlesDay(
                date=cache_date,
                articles=articles,
                total_count=len(articles),
            )
            cache_file.write_text(json.dumps(cached_day.model_dump(mode="json"), indent=2))

        # New article today
        new_article = [
            ProcessedArticle(
                title="Today's Article",
                url="https://example.com/article-today",
                published_date=datetime.now(),
                content="Fresh content",
                author="Test Author",
                feed_name="Test Feed",
                feed_priority=5,
                slug="todays-article-hashtoday",
                content_hash="hash_today",
            )
        ]

        # Run deduplication
        result = await run_step2(step2_config, new_article, cache_manager)

        assert result.success is True
        assert result.stats.cache_articles == 5  # 5 days of cached articles
        assert result.stats.duplicates_found == 0
        assert len(result.unique_articles) == 1

    @pytest.mark.asyncio
    async def test_step1_to_step2_integration(
        self,
        step2_config: Step2Config,
        pipeline_cache_dir: Path,
    ) -> None:
        """Test Step 2 receiving realistic output from Step 1."""
        cache_manager = CacheManager(cache_dir=pipeline_cache_dir)

        # Simulate Step 1 output (mix of new and old articles)
        step1_articles = [
            ProcessedArticle(
                title="Breaking: New AI Model Released",
                url="https://example.com/new-model",
                published_date=datetime.now(),
                content="A new AI model was released today...",
                author="Tech Reporter",
                feed_name="AI News Daily",
                feed_priority=9,
                slug="breaking-new-ai-model-released-abc123",
                content_hash="hash_new_model",
            ),
            ProcessedArticle(
                title="AI Ethics Conference Announced",
                url="https://example.com/ethics-conf",
                published_date=datetime.now() - timedelta(hours=3),
                content="Annual AI ethics conference details...",
                author="Conference Organizer",
                feed_name="AI Events",
                feed_priority=7,
                slug="ai-ethics-conference-announced-def456",
                content_hash="hash_ethics",
            ),
            ProcessedArticle(
                title="Machine Learning Tutorial Series",
                url="https://example.com/ml-tutorial",
                published_date=datetime.now() - timedelta(hours=6),
                content="Part 1 of our ML tutorial series...",
                author="ML Educator",
                feed_name="Learn AI",
                feed_priority=6,
                slug="machine-learning-tutorial-series-ghi789",
                content_hash="hash_tutorial",
            ),
        ]

        # First processing
        result1 = await run_step2(step2_config, step1_articles, cache_manager)

        assert result1.success is True
        assert len(result1.unique_articles) == 3

        # Simulate next day's Step 1 output (some duplicates, some new)
        step1_next_day = [
            step1_articles[0],  # Duplicate from yesterday
            ProcessedArticle(
                title="AI Startup Raises $100M",
                url="https://example.com/startup-funding",
                published_date=datetime.now(),
                content="AI startup secures major funding...",
                author="Business Reporter",
                feed_name="Tech Business",
                feed_priority=8,
                slug="ai-startup-raises-100m-jkl012",
                content_hash="hash_funding",
            ),
        ]

        result2 = await run_step2(step2_config, step1_next_day, cache_manager)

        assert result2.success is True
        assert result2.stats.duplicates_found == 1
        assert len(result2.unique_articles) == 1
        assert result2.unique_articles[0].title == "AI Startup Raises $100M"

    @pytest.mark.asyncio
    async def test_cache_rotation_after_10_days(
        self,
        step2_config: Step2Config,
        pipeline_cache_dir: Path,
    ) -> None:
        """Test that cache files older than 10 days are ignored."""
        cache_manager = CacheManager(cache_dir=pipeline_cache_dir)
        articles_cache_dir = pipeline_cache_dir / "articles"
        articles_cache_dir.mkdir(parents=True, exist_ok=True)

        # Create cache files: 5 days old (should be loaded) and 15 days old (should be ignored)
        old_article = ProcessedArticle(
            title="Old Article from 15 Days Ago",
            url="https://example.com/very-old",
            published_date=datetime.now() - timedelta(days=15),
            content="Very old content",
            author="Old Author",
            feed_name="Archive",
            feed_priority=5,
            slug="old-article-from-15-days-ago-old123",
            content_hash="hash_very_old",
        )

        recent_article = ProcessedArticle(
            title="Recent Article from 5 Days Ago",
            url="https://example.com/recent",
            published_date=datetime.now() - timedelta(days=5),
            content="Recent content",
            author="Recent Author",
            feed_name="News",
            feed_priority=5,
            slug="recent-article-from-5-days-ago-rec123",
            content_hash="hash_recent",
        )

        # Save old cache file (15 days)
        old_date = datetime.now() - timedelta(days=15)
        old_file = articles_cache_dir / f"{old_date:%Y-%m-%d}.json"
        old_cached = CachedArticlesDay(
            date=old_date,
            articles=[old_article],
            total_count=1,
        )
        old_file.write_text(json.dumps(old_cached.model_dump(mode="json"), indent=2))

        # Save recent cache file (5 days)
        recent_date = datetime.now() - timedelta(days=5)
        recent_file = articles_cache_dir / f"{recent_date:%Y-%m-%d}.json"
        recent_cached = CachedArticlesDay(
            date=recent_date,
            articles=[recent_article],
            total_count=1,
        )
        recent_file.write_text(json.dumps(recent_cached.model_dump(mode="json"), indent=2))

        # Test with both articles as input
        test_articles = [old_article, recent_article]
        result = await run_step2(step2_config, test_articles, cache_manager)

        assert result.success is True
        # Only recent article should be in cache (old one ignored)
        assert result.stats.cache_articles == 1
        # Old article is unique (not in cache), recent article is duplicate
        assert result.stats.duplicates_found == 1
        assert len(result.unique_articles) == 1
        assert result.unique_articles[0].slug == old_article.slug

    @pytest.mark.asyncio
    async def test_high_volume_deduplication(
        self,
        step2_config: Step2Config,
        pipeline_cache_dir: Path,
    ) -> None:
        """Test deduplication with high volume of articles."""
        cache_manager = CacheManager(cache_dir=pipeline_cache_dir)

        # Generate 100 articles
        large_batch = [
            ProcessedArticle(
                title=f"AI Article {i}",
                url=f"https://example.com/article-{i}",
                published_date=datetime.now() - timedelta(hours=i),
                content=f"Content for article {i}",
                author=f"Author {i % 10}",
                feed_name=f"Feed {i % 5}",
                feed_priority=5 + (i % 5),
                slug=f"ai-article-{i}-hash{i:04d}",
                content_hash=f"hash_{i}",
            )
            for i in range(100)
        ]

        # First run: all unique
        result1 = await run_step2(step2_config, large_batch, cache_manager)

        assert result1.success is True
        assert len(result1.unique_articles) == 100
        assert result1.stats.duplicates_found == 0

        # Second run: 50 duplicates + 50 new
        mixed_batch = large_batch[:50] + [
            ProcessedArticle(
                title=f"New AI Article {i}",
                url=f"https://example.com/new-article-{i}",
                published_date=datetime.now(),
                content=f"New content {i}",
                author=f"New Author {i % 10}",
                feed_name=f"Feed {i % 5}",
                feed_priority=5 + (i % 5),
                slug=f"new-ai-article-{i}-newhash{i:04d}",
                content_hash=f"new_hash_{i}",
            )
            for i in range(50)
        ]

        result2 = await run_step2(step2_config, mixed_batch, cache_manager)

        assert result2.success is True
        assert result2.stats.input_articles == 100
        assert result2.stats.cache_articles == 100
        assert result2.stats.duplicates_found == 50
        assert len(result2.unique_articles) == 50
        assert result2.stats.deduplication_rate == 0.5

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(
        self,
        step2_config: Step2Config,
        pipeline_cache_dir: Path,
    ) -> None:
        """Test Step 2 handles errors gracefully in realistic scenarios."""
        cache_manager = CacheManager(cache_dir=pipeline_cache_dir)
        articles_cache_dir = pipeline_cache_dir / "articles"
        articles_cache_dir.mkdir(parents=True, exist_ok=True)

        # Create mix of valid and corrupted cache files
        valid_date = datetime.now() - timedelta(days=2)
        valid_file = articles_cache_dir / f"{valid_date:%Y-%m-%d}.json"
        valid_article = ProcessedArticle(
            title="Valid Article",
            url="https://example.com/valid",
            published_date=valid_date,
            content="Valid content",
            author="Author",
            feed_name="Feed",
            feed_priority=5,
            slug="valid-article-validhash",
            content_hash="hash_valid",
        )
        cached_day = CachedArticlesDay(
            date=valid_date,
            articles=[valid_article],
            total_count=1,
        )
        valid_file.write_text(json.dumps(cached_day.model_dump(mode="json"), indent=2))

        # Create corrupted file
        corrupt_date = datetime.now() - timedelta(days=3)
        corrupt_file = articles_cache_dir / f"{corrupt_date:%Y-%m-%d}.json"
        corrupt_file.write_text("{ corrupted json content")

        # New articles including duplicate of valid article
        new_articles = [
            valid_article,  # Duplicate
            ProcessedArticle(
                title="New Article",
                url="https://example.com/new",
                published_date=datetime.now(),
                content="New content",
                author="Author",
                feed_name="Feed",
                feed_priority=5,
                slug="new-article-newhash",
                content_hash="hash_new",
            ),
        ]

        result = await run_step2(step2_config, new_articles, cache_manager)

        # Should succeed despite corrupted file
        assert result.success is True
        assert result.stats.cache_files_corrupted == 1
        assert result.stats.cache_files_loaded == 1
        assert result.stats.duplicates_found == 1
        assert len(result.unique_articles) == 1
        assert result.unique_articles[0].title == "New Article"

    @pytest.mark.asyncio
    async def test_concurrent_cache_updates(
        self,
        step2_config: Step2Config,
        pipeline_cache_dir: Path,
    ) -> None:
        """Test cache consistency with multiple operations."""
        cache_manager = CacheManager(cache_dir=pipeline_cache_dir)

        # First batch
        batch1 = [
            ProcessedArticle(
                title=f"Batch 1 Article {i}",
                url=f"https://example.com/b1-{i}",
                published_date=datetime.now(),
                content=f"Content {i}",
                author="Author",
                feed_name="Feed",
                feed_priority=5,
                slug=f"batch-1-article-{i}-b1hash{i}",
                content_hash=f"hash_b1_{i}",
            )
            for i in range(5)
        ]

        result1 = await run_step2(step2_config, batch1, cache_manager)
        assert result1.success is True
        assert len(result1.unique_articles) == 5

        # Second batch (different articles)
        batch2 = [
            ProcessedArticle(
                title=f"Batch 2 Article {i}",
                url=f"https://example.com/b2-{i}",
                published_date=datetime.now(),
                content=f"Content {i}",
                author="Author",
                feed_name="Feed",
                feed_priority=5,
                slug=f"batch-2-article-{i}-b2hash{i}",
                content_hash=f"hash_b2_{i}",
            )
            for i in range(5)
        ]

        result2 = await run_step2(step2_config, batch2, cache_manager)
        assert result2.success is True
        # Should see previous batch in cache
        assert result2.stats.cache_articles == 5
        assert len(result2.unique_articles) == 5

        # Verify cache file has all articles (accumulated from both batches)
        articles_cache_dir = pipeline_cache_dir / "articles"
        today_file = articles_cache_dir / f"{datetime.now():%Y-%m-%d}.json"

        cache_data = json.loads(today_file.read_text())
        cached_day = CachedArticlesDay.model_validate(cache_data)

        # Should have articles from both batches (accumulated)
        assert len(cached_day.articles) == 10
        batch1_count = sum(1 for art in cached_day.articles if "Batch 1" in art.title)
        batch2_count = sum(1 for art in cached_day.articles if "Batch 2" in art.title)
        assert batch1_count == 5
        assert batch2_count == 5
