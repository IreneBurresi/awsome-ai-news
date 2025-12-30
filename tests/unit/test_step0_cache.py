"""Unit tests for Step 0: Cache management."""

from pathlib import Path

import pytest

from src.models.config import Step0Config
from src.steps.step0_cache import (
    acquire_lock_file,
    release_lock_file,
    run_step0,
    verify_repo_permissions,
)
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
def step0_config() -> Step0Config:
    """Create Step 0 configuration."""
    return Step0Config(
        enabled=True,
        retention={"articles_days": 10, "news_days": 3},
        backup_on_error=True,
        cleanup_on_start=True,
    )


class TestStep0Cache:
    """Test Step 0: Cache management."""

    @pytest.mark.asyncio
    async def test_step0_basic_execution(
        self, step0_config: Step0Config, cache_manager: CacheManager
    ) -> None:
        """Test basic Step 0 execution."""
        result = await run_step0(step0_config, cache_manager, check_gemini=False)

        assert result.success is True
        assert result.cache_cleaned >= 0
        assert isinstance(result.errors, list)
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_step0_creates_cache_directory(
        self, step0_config: Step0Config, tmp_path: Path
    ) -> None:
        """Test Step 0 creates cache directory if missing."""
        cache_dir = tmp_path / "new_cache"
        manager = CacheManager(cache_dir=cache_dir)

        result = await run_step0(step0_config, manager, check_gemini=False)

        assert cache_dir.exists()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_step0_with_backup_disabled(
        self, cache_manager: CacheManager, temp_cache_dir: Path
    ) -> None:
        """Test Step 0 with backup disabled."""
        config = Step0Config(
            enabled=True,
            retention={"articles_days": 10, "news_days": 3},
            backup_on_error=False,
            cleanup_on_start=True,
        )

        result = await run_step0(config, cache_manager, check_gemini=False)

        assert result.success is True
        assert result.cache_backed_up is False

        # Backup directory should not exist
        backup_dir = temp_cache_dir.parent / "cache_backups"
        assert not backup_dir.exists() or len(list(backup_dir.glob("*"))) == 0

    @pytest.mark.asyncio
    async def test_step0_with_cleanup_disabled(self, cache_manager: CacheManager) -> None:
        """Test Step 0 with cleanup disabled."""
        config = Step0Config(
            enabled=True,
            retention={"articles_days": 10, "news_days": 3},
            backup_on_error=False,
            cleanup_on_start=False,
        )

        result = await run_step0(config, cache_manager, check_gemini=False)

        assert result.success is True
        assert result.cache_cleaned == 0

    @pytest.mark.asyncio
    async def test_step0_disabled(self, cache_manager: CacheManager) -> None:
        """Test Step 0 can be configured as disabled."""
        config = Step0Config(
            enabled=False,
            retention={"articles_days": 10, "news_days": 3},
            backup_on_error=False,
            cleanup_on_start=False,
        )

        # Step should still execute even if disabled flag is set
        # (The pipeline orchestrator decides whether to skip)
        result = await run_step0(config, cache_manager, check_gemini=False)
        assert result.success is True


class TestHealthChecks:
    """Test health check functions."""

    def test_verify_repo_permissions_success(self) -> None:
        """Test repository permissions check succeeds."""
        result = verify_repo_permissions()
        assert result is True

    def test_acquire_lock_file(self, temp_cache_dir: Path) -> None:
        """Test lock file acquisition."""
        # First acquisition should succeed
        assert acquire_lock_file(temp_cache_dir) is True

        # Lock file should exist
        lock_file = temp_cache_dir / ".lock"
        assert lock_file.exists()

        # Second acquisition should fail (already locked)
        assert acquire_lock_file(temp_cache_dir) is False

    def test_release_lock_file(self, temp_cache_dir: Path) -> None:
        """Test lock file release."""
        # Acquire lock
        acquire_lock_file(temp_cache_dir)
        lock_file = temp_cache_dir / ".lock"
        assert lock_file.exists()

        # Release lock
        release_lock_file(temp_cache_dir)
        assert not lock_file.exists()

    def test_stale_lock_file_removal(self, temp_cache_dir: Path) -> None:
        """Test stale lock file is automatically removed."""
        import time

        lock_file = temp_cache_dir / ".lock"

        # Create a stale lock file (older than 1 hour)
        lock_file.write_text("stale lock")

        # Modify the timestamp to make it appear old
        old_time = time.time() - 3700  # > 1 hour ago
        import os

        os.utime(lock_file, (old_time, old_time))

        # Acquiring lock should succeed (stale lock is removed)
        assert acquire_lock_file(temp_cache_dir) is True
        assert lock_file.exists()  # New lock file created

        # Clean up
        release_lock_file(temp_cache_dir)
