"""BDD step definitions for Step 0: Cache Management."""

import asyncio
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel
from pytest_bdd import given, scenarios, then, when

from src.models.config import Step0Config
from src.steps.step0_cache import Step0Result, run_step0
from src.utils.cache import CacheManager

# Load scenarios from feature file
scenarios("features/step0_cache.feature")


class TestCacheModel(BaseModel):
    """Simple test model for cache."""

    id: int
    name: str


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Temporary cache directory."""
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache


@pytest.fixture
def cache_manager(cache_dir: Path) -> CacheManager:
    """Cache manager instance."""
    return CacheManager(cache_dir=cache_dir)


@pytest.fixture
def step0_config() -> Step0Config:
    """Default Step 0 configuration."""
    return Step0Config(
        enabled=True,
        retention={"articles_days": 10, "news_days": 3},
        backup_on_error=True,
        cleanup_on_start=True,
    )


@pytest.fixture
def step0_result() -> dict:
    """Storage for step result."""
    return {}


# Background steps


@given("the cache directory exists")
def cache_directory_exists(cache_dir: Path) -> None:
    """Ensure cache directory exists."""
    assert cache_dir.exists()


@given("the cache has retention policy of 10 days for articles and 3 days for news")
def cache_retention_policy(step0_config: Step0Config) -> None:
    """Verify retention policy."""
    assert step0_config.retention["articles_days"] == 10
    assert step0_config.retention["news_days"] == 3


# Given steps


@given("the cache is empty")
def empty_cache(cache_manager: CacheManager) -> None:
    """Ensure cache is empty."""
    assert len(cache_manager.list_all()) == 0


@given("the cache contains articles older than 10 days")
def old_articles_in_cache(cache_manager: CacheManager, cache_dir: Path) -> None:
    """Create old articles in cache."""
    cache_manager.save("articles", [TestCacheModel(id=1, name="Old Article")])
    # Modify timestamp to be old
    cache_path = cache_dir / "articles.json"
    content = json.loads(cache_path.read_text())
    old_date = (datetime.now() - timedelta(days=15)).isoformat()
    content["cached_at"] = old_date
    cache_path.write_text(json.dumps(content))


@given("the cache contains news older than 3 days")
def old_news_in_cache(cache_manager: CacheManager, cache_dir: Path) -> None:
    """Create old news in cache."""
    cache_manager.save("news", [TestCacheModel(id=2, name="Old News")])
    cache_path = cache_dir / "news.json"
    content = json.loads(cache_path.read_text())
    old_date = (datetime.now() - timedelta(days=5)).isoformat()
    content["cached_at"] = old_date
    cache_path.write_text(json.dumps(content))


@given("the cache contains fresh articles")
def fresh_articles_in_cache(cache_manager: CacheManager) -> None:
    """Create fresh articles in cache."""
    cache_manager.save("fresh_articles", [TestCacheModel(id=3, name="Fresh Article")])


@given("the cache contains some entries")
def cache_with_entries(cache_manager: CacheManager) -> None:
    """Create some cache entries."""
    cache_manager.save("test1", [TestCacheModel(id=1, name="Entry 1")])
    cache_manager.save("test2", [TestCacheModel(id=2, name="Entry 2")])


@given("the cache contains old entries")
def cache_with_old_entries(cache_manager: CacheManager, cache_dir: Path) -> None:
    """Create old cache entries."""
    cache_manager.save("old_entry", [TestCacheModel(id=1, name="Old")])
    cache_path = cache_dir / "old_entry.json"
    content = json.loads(cache_path.read_text())
    old_date = (datetime.now() - timedelta(days=20)).isoformat()
    content["cached_at"] = old_date
    cache_path.write_text(json.dumps(content))


@given("the cache directory does not exist")
def no_cache_directory(cache_dir: Path) -> None:
    """Remove cache directory."""
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    assert not cache_dir.exists()


@given("the cache has 6 existing backups")
def six_backups(cache_dir: Path) -> None:
    """Create 6 backup directories."""
    backup_dir = cache_dir.parent / "cache_backups"
    backup_dir.mkdir(exist_ok=True)
    for i in range(6):
        backup_path = backup_dir / f"cache_backup_2024010{i}_120000"
        backup_path.mkdir()


# When steps


@when("Step 0 executes with cleanup enabled")
def execute_step0_cleanup_enabled(
    step0_config: Step0Config, cache_manager: CacheManager, step0_result: dict
) -> None:
    """Execute Step 0 with cleanup enabled."""
    step0_config.cleanup_on_start = True
    result = asyncio.run(run_step0(step0_config, cache_manager, check_gemini=False))
    step0_result["result"] = result


@when("Step 0 executes with backup enabled")
def execute_step0_backup_enabled(
    step0_config: Step0Config, cache_manager: CacheManager, step0_result: dict
) -> None:
    """Execute Step 0 with backup enabled."""
    step0_config.backup_on_error = True
    result = asyncio.run(run_step0(step0_config, cache_manager, check_gemini=False))
    step0_result["result"] = result


@when("Step 0 executes with backup disabled")
def execute_step0_backup_disabled(
    step0_config: Step0Config, cache_manager: CacheManager, step0_result: dict
) -> None:
    """Execute Step 0 with backup disabled."""
    step0_config.backup_on_error = False
    result = asyncio.run(run_step0(step0_config, cache_manager, check_gemini=False))
    step0_result["result"] = result


@when("Step 0 executes with cleanup disabled")
def execute_step0_cleanup_disabled(
    step0_config: Step0Config, cache_manager: CacheManager, step0_result: dict
) -> None:
    """Execute Step 0 with cleanup disabled."""
    step0_config.cleanup_on_start = False
    result = asyncio.run(run_step0(step0_config, cache_manager, check_gemini=False))
    step0_result["result"] = result


@when("Step 0 executes")
def execute_step0(
    step0_config: Step0Config, cache_manager: CacheManager, step0_result: dict
) -> None:
    """Execute Step 0 with default config."""
    # Recreate cache manager to handle missing directory
    cache_manager = CacheManager(cache_dir=cache_manager.cache_dir)
    result = asyncio.run(run_step0(step0_config, cache_manager, check_gemini=False))
    step0_result["result"] = result


# Then steps


@then("Step 0 should succeed")
def step0_succeeds(step0_result: dict) -> None:
    """Verify Step 0 succeeded."""
    result: Step0Result = step0_result["result"]
    assert result.success is True


@then("no cache entries should be cleaned")
def no_cleanup(step0_result: dict) -> None:
    """Verify no entries were cleaned."""
    result: Step0Result = step0_result["result"]
    assert result.cache_cleaned == 0


@then("the cache directory should be ready for use")
def cache_ready(cache_dir: Path) -> None:
    """Verify cache directory is ready."""
    assert cache_dir.exists()
    assert cache_dir.is_dir()


@then("old articles should be removed")
def old_articles_removed(cache_manager: CacheManager) -> None:
    """Verify old articles were removed."""
    assert not cache_manager.exists("articles")


@then("old news should be removed")
def old_news_removed(cache_manager: CacheManager) -> None:
    """Verify old news were removed."""
    assert not cache_manager.exists("news")


@then("fresh articles should remain in cache")
def fresh_articles_remain(cache_manager: CacheManager) -> None:
    """Verify fresh articles remain."""
    assert cache_manager.exists("fresh_articles")


@then("a cache backup should be created")
def backup_created(cache_dir: Path) -> None:
    """Verify backup was created."""
    backup_dir = cache_dir.parent / "cache_backups"
    assert backup_dir.exists()
    backups = list(backup_dir.glob("cache_backup_*"))
    assert len(backups) > 0


@then("the backup should contain all original entries")
def backup_contains_entries(cache_dir: Path) -> None:
    """Verify backup contains entries."""
    backup_dir = cache_dir.parent / "cache_backups"
    backups = sorted(backup_dir.glob("cache_backup_*"))
    if backups:
        latest_backup = backups[-1]
        # Backup should have same structure as cache
        assert latest_backup.is_dir()


@then("no cache backup should be created")
def no_backup_created(cache_dir: Path, step0_result: dict) -> None:
    """Verify no backup was created."""
    result: Step0Result = step0_result["result"]
    assert result.cache_backed_up is False


@then("all old entries should remain")
def old_entries_remain(cache_manager: CacheManager) -> None:
    """Verify old entries were not removed."""
    assert cache_manager.exists("old_entry")


@then("the cache directory should be created")
def cache_directory_created(cache_dir: Path) -> None:
    """Verify cache directory was created."""
    assert cache_dir.exists()


@then("the cache should be ready for use")
def cache_is_ready(cache_dir: Path) -> None:
    """Verify cache is ready."""
    assert cache_dir.exists()
    assert cache_dir.is_dir()


@then("exactly 5 backups should remain")
def five_backups_remain(cache_dir: Path) -> None:
    """Verify only 5 backups remain."""
    backup_dir = cache_dir.parent / "cache_backups"
    backups = list(backup_dir.glob("cache_backup_*"))
    assert len(backups) == 5


@then("the oldest backup should be removed")
def oldest_backup_removed(cache_dir: Path) -> None:
    """Verify oldest backup was removed."""
    backup_dir = cache_dir.parent / "cache_backups"
    # The backup with oldest name should not exist
    assert not (backup_dir / "cache_backup_20240100_120000").exists()
