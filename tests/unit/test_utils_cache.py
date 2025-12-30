"""Unit tests for cache utilities."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from src.utils.cache import CacheManager


class TestModel(BaseModel):
    """Test model for cache tests."""

    id: int
    name: str
    value: float = Field(default=0.0)


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


class TestCacheManager:
    """Test CacheManager class."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test cache manager creates directory."""
        cache_dir = tmp_path / "new_cache"
        manager = CacheManager(cache_dir=cache_dir)
        assert cache_dir.exists()
        assert manager.cache_dir == cache_dir

    def test_save_and_load_single_item(self, cache_manager: CacheManager) -> None:
        """Test saving and loading single item."""
        item = TestModel(id=1, name="Test", value=42.0)
        cache_manager.save("test", item)

        loaded = cache_manager.load("test", TestModel)
        assert loaded is not None
        assert loaded.id == 1
        assert loaded.name == "Test"
        assert loaded.value == 42.0

    def test_save_and_load_list(self, cache_manager: CacheManager) -> None:
        """Test saving and loading list of items."""
        items = [
            TestModel(id=1, name="First", value=1.0),
            TestModel(id=2, name="Second", value=2.0),
            TestModel(id=3, name="Third", value=3.0),
        ]
        cache_manager.save("test_list", items)

        loaded = cache_manager.load("test_list", TestModel)
        assert loaded is not None
        assert len(loaded) == 3
        assert loaded[0].id == 1
        assert loaded[1].name == "Second"
        assert loaded[2].value == 3.0

    def test_load_nonexistent_key(self, cache_manager: CacheManager) -> None:
        """Test loading non-existent key returns None."""
        result = cache_manager.load("nonexistent", TestModel)
        assert result is None

    def test_exists(self, cache_manager: CacheManager) -> None:
        """Test cache exists check."""
        assert not cache_manager.exists("test")

        item = TestModel(id=1, name="Test")
        cache_manager.save("test", item)

        assert cache_manager.exists("test")

    def test_get_age(self, cache_manager: CacheManager) -> None:
        """Test getting cache age."""
        item = TestModel(id=1, name="Test")
        cache_manager.save("test", item)

        age = cache_manager.get_age("test")
        assert age is not None
        assert age < timedelta(seconds=1)  # Just created

    def test_get_age_nonexistent(self, cache_manager: CacheManager) -> None:
        """Test get age for non-existent key."""
        age = cache_manager.get_age("nonexistent")
        assert age is None

    def test_is_fresh(self, cache_manager: CacheManager, temp_cache_dir: Path) -> None:
        """Test cache freshness check."""
        item = TestModel(id=1, name="Test")
        cache_manager.save("test", item)

        # Fresh cache (< 1 day old)
        assert cache_manager.is_fresh("test", max_age_days=1)

        # Simulate old cache by modifying cached_at timestamp
        cache_path = temp_cache_dir / "test.json"
        cache_content = json.loads(cache_path.read_text())
        old_date = (datetime.now() - timedelta(days=5)).isoformat()
        cache_content["cached_at"] = old_date
        cache_path.write_text(json.dumps(cache_content))

        # Now it should not be fresh
        assert not cache_manager.is_fresh("test", max_age_days=1)
        # But fresh if we allow 10 days
        assert cache_manager.is_fresh("test", max_age_days=10)

    def test_delete(self, cache_manager: CacheManager) -> None:
        """Test cache deletion."""
        item = TestModel(id=1, name="Test")
        cache_manager.save("test", item)

        assert cache_manager.exists("test")

        cache_manager.delete("test")

        assert not cache_manager.exists("test")

    def test_delete_nonexistent(self, cache_manager: CacheManager) -> None:
        """Test deleting non-existent key doesn't raise error."""
        cache_manager.delete("nonexistent")  # Should not raise

    def test_cleanup(self, cache_manager: CacheManager, temp_cache_dir: Path) -> None:
        """Test cache cleanup based on retention policy."""
        # Create fresh cache
        item1 = TestModel(id=1, name="Fresh")
        cache_manager.save("fresh", item1)

        # Create old cache
        item2 = TestModel(id=2, name="Old")
        cache_manager.save("old", item2)

        # Make "old" cache actually old
        cache_path = temp_cache_dir / "old.json"
        cache_content = json.loads(cache_path.read_text())
        old_date = (datetime.now() - timedelta(days=5)).isoformat()
        cache_content["cached_at"] = old_date
        cache_path.write_text(json.dumps(cache_content))

        # Cleanup with retention policy
        retention = {"fresh": 10, "old": 3}
        cache_manager.cleanup(retention)

        # Fresh should still exist
        assert cache_manager.exists("fresh")
        # Old should be deleted
        assert not cache_manager.exists("old")

    def test_list_all(self, cache_manager: CacheManager) -> None:
        """Test listing all cache keys."""
        assert cache_manager.list_all() == []

        cache_manager.save("test1", TestModel(id=1, name="Test1"))
        cache_manager.save("test2", TestModel(id=2, name="Test2"))
        cache_manager.save("test3", TestModel(id=3, name="Test3"))

        keys = cache_manager.list_all()
        assert len(keys) == 3
        assert "test1" in keys
        assert "test2" in keys
        assert "test3" in keys

    def test_save_corrupted_data_handling(
        self, cache_manager: CacheManager, temp_cache_dir: Path
    ) -> None:
        """Test loading corrupted cache returns None."""
        # Create corrupted cache file
        cache_path = temp_cache_dir / "corrupted.json"
        cache_path.write_text("{ invalid json }")

        result = cache_manager.load("corrupted", TestModel)
        assert result is None
