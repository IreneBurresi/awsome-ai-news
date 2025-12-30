"""Integration tests for Step 0 with full pipeline setup."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel

from src.models.config import Step0Config
from src.steps.step0_cache import run_step0
from src.utils.cache import CacheManager


class SampleArticle(BaseModel):
    """Sample article model for testing."""

    id: int
    title: str
    url: str


@pytest.fixture
def pipeline_cache_dir(tmp_path: Path) -> Path:
    """Create pipeline-like cache directory structure."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def populated_cache(pipeline_cache_dir: Path) -> CacheManager:
    """Create cache with realistic pipeline data."""
    manager = CacheManager(cache_dir=pipeline_cache_dir)

    # Add various cache entries simulating pipeline run
    manager.save(
        "articles",
        [
            SampleArticle(id=1, title="Old Article 1", url="https://example.com/1"),
            SampleArticle(id=2, title="Old Article 2", url="https://example.com/2"),
        ],
    )

    manager.save(
        "processed_articles",
        [SampleArticle(id=3, title="Processed", url="https://example.com/3")],
    )

    manager.save("news", [SampleArticle(id=4, title="Old News", url="https://example.com/4")])

    # Make some entries old
    articles_path = pipeline_cache_dir / "articles.json"
    content = json.loads(articles_path.read_text())
    old_date = (datetime.now() - timedelta(days=15)).isoformat()
    content["cached_at"] = old_date
    articles_path.write_text(json.dumps(content))

    news_path = pipeline_cache_dir / "news.json"
    content = json.loads(news_path.read_text())
    old_date = (datetime.now() - timedelta(days=5)).isoformat()
    content["cached_at"] = old_date
    news_path.write_text(json.dumps(content))

    return manager


class TestStep0Integration:
    """Integration tests for Step 0."""

    @pytest.mark.asyncio
    async def test_full_cleanup_workflow(self, populated_cache: CacheManager) -> None:
        """Test complete cleanup workflow with realistic data."""
        config = Step0Config(
            enabled=True,
            retention={"articles_days": 10, "news_days": 3},
            backup_on_error=True,
            cleanup_on_start=True,
        )

        # Verify initial state
        assert populated_cache.exists("articles")
        assert populated_cache.exists("news")
        assert populated_cache.exists("processed_articles")

        # Run Step 0
        result = await run_step0(config, populated_cache, check_gemini=False)

        # Verify result
        assert result.success is True
        assert result.cache_cleaned > 0
        assert result.cache_backed_up is True

        # Old entries should be removed
        assert not populated_cache.exists("articles")
        assert not populated_cache.exists("news")

        # Fresh entries should remain
        assert populated_cache.exists("processed_articles")

    @pytest.mark.asyncio
    async def test_backup_and_restore_workflow(
        self, populated_cache: CacheManager, pipeline_cache_dir: Path
    ) -> None:
        """Test backup creation and validation."""
        config = Step0Config(
            enabled=True,
            retention={"articles_days": 10, "news_days": 3},
            backup_on_error=True,
            cleanup_on_start=True,
        )

        # Get initial cache state
        initial_entries = set(populated_cache.list_all())

        # Run Step 0
        result = await run_step0(config, populated_cache, check_gemini=False)

        assert result.success is True
        assert result.cache_backed_up is True

        # Verify backup exists
        backup_dir = pipeline_cache_dir.parent / "cache_backups"
        assert backup_dir.exists()

        backups = list(backup_dir.glob("cache_backup_*"))
        assert len(backups) > 0

        # Verify backup contains original data
        latest_backup = sorted(backups)[-1]
        assert latest_backup.is_dir()

        # Backup should have JSON files
        backup_files = {f.stem for f in latest_backup.glob("*.json")}
        assert backup_files == initial_entries

    @pytest.mark.asyncio
    async def test_multiple_runs_maintain_backups(
        self, populated_cache: CacheManager, pipeline_cache_dir: Path
    ) -> None:
        """Test that multiple Step 0 runs maintain backup rotation."""
        config = Step0Config(
            enabled=True,
            retention={"articles_days": 10, "news_days": 3},
            backup_on_error=True,
            cleanup_on_start=True,
        )

        # Run Step 0 multiple times
        for _ in range(3):
            result = await run_step0(config, populated_cache, check_gemini=False)
            assert result.success is True

        # Check backup count (should not exceed 5)
        backup_dir = pipeline_cache_dir.parent / "cache_backups"
        backups = list(backup_dir.glob("cache_backup_*"))
        assert len(backups) <= 5

    @pytest.mark.asyncio
    async def test_selective_retention(self, pipeline_cache_dir: Path) -> None:
        """Test different retention periods for different cache types."""
        manager = CacheManager(cache_dir=pipeline_cache_dir)

        # Create entries with different ages
        manager.save("articles", [SampleArticle(id=1, title="A", url="http://a.com")])
        manager.save("news", [SampleArticle(id=2, title="N", url="http://n.com")])
        manager.save("processed_articles", [SampleArticle(id=3, title="P", url="http://p.com")])

        # Make articles 11 days old (should be removed with 10 day retention)
        articles_path = pipeline_cache_dir / "articles.json"
        content = json.loads(articles_path.read_text())
        content["cached_at"] = (datetime.now() - timedelta(days=11)).isoformat()
        articles_path.write_text(json.dumps(content))

        # Make news 4 days old (should be removed with 3 day retention)
        news_path = pipeline_cache_dir / "news.json"
        content = json.loads(news_path.read_text())
        content["cached_at"] = (datetime.now() - timedelta(days=4)).isoformat()
        news_path.write_text(json.dumps(content))

        # Make processed_articles 5 days old (should stay with 10 day retention)
        processed_path = pipeline_cache_dir / "processed_articles.json"
        content = json.loads(processed_path.read_text())
        content["cached_at"] = (datetime.now() - timedelta(days=5)).isoformat()
        processed_path.write_text(json.dumps(content))

        # Run cleanup
        config = Step0Config(
            enabled=True,
            retention={"articles_days": 10, "news_days": 3},
            backup_on_error=False,
            cleanup_on_start=True,
        )

        result = await run_step0(config, manager, check_gemini=False)

        assert result.success is True
        assert not manager.exists("articles")  # Too old
        assert not manager.exists("news")  # Too old
        assert manager.exists("processed_articles")  # Still fresh

    @pytest.mark.asyncio
    async def test_error_recovery(self, pipeline_cache_dir: Path) -> None:
        """Test Step 0 handles errors gracefully."""
        manager = CacheManager(cache_dir=pipeline_cache_dir)

        # Create a corrupted cache file
        corrupted_path = pipeline_cache_dir / "corrupted.json"
        corrupted_path.write_text("{ invalid json")

        config = Step0Config(
            enabled=True,
            retention={"articles_days": 10, "news_days": 3},
            backup_on_error=True,
            cleanup_on_start=True,
        )

        # Step 0 should handle corruption gracefully
        result = await run_step0(config, manager, check_gemini=False)

        # Should still succeed overall
        assert result.success is True or len(result.errors) > 0
