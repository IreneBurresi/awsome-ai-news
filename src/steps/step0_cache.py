"""Step 0: Cache management and cleanup."""

import os
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from src.models.config import Step0Config
from src.utils.cache import CacheManager
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Step0Result(BaseModel):
    """Result from Step 0: Cache management."""

    success: bool = Field(description="Whether step completed successfully")
    cache_cleaned: int = Field(description="Number of cache entries cleaned")
    cache_backed_up: bool = Field(description="Whether cache was backed up")
    errors: list[str] = Field(default_factory=list, description="List of errors encountered")
    timestamp: datetime = Field(default_factory=datetime.now)


async def run_step0(
    config: Step0Config,
    cache_manager: CacheManager,
    api_key: str | None = None,
    check_gemini: bool = False,
) -> Step0Result:
    """
    Execute Step 0: Cache management.

    This step:
    1. Acquires lock file to prevent concurrent runs
    2. Validates Gemini API availability (optional)
    3. Validates repository write permissions
    4. Validates cache directory structure
    5. Cleans up expired cache entries based on retention policy
    6. Creates backup if needed
    7. Prepares cache for new pipeline run

    Args:
        config: Step 0 configuration
        cache_manager: Cache manager instance
        api_key: Optional Gemini API key (defaults to GEMINI_API_KEY env var)
        check_gemini: Whether to perform Gemini API health check
            (default False, only needed for debugging)

    Returns:
        Step0Result with cleanup statistics

    Raises:
        Exception: On critical cache errors
    """
    logger.info("Starting Step 0: Cache management")
    errors: list[str] = []
    cache_cleaned = 0
    cache_backed_up = False

    try:
        # Acquire lock file to prevent concurrent runs
        if not acquire_lock_file(cache_manager.cache_dir):
            error_msg = "Failed to acquire lock file - another instance may be running"
            logger.error(error_msg)
            return Step0Result(
                success=False,
                cache_cleaned=0,
                cache_backed_up=False,
                errors=[error_msg],
            )

        # Verify Gemini API health (optional, can be disabled for local testing)
        if check_gemini and not await verify_gemini_health(api_key):
            error_msg = "Gemini API health check failed - API may be unavailable"
            logger.warning(error_msg)
            errors.append(error_msg)
            # Note: This is a warning, not a critical error for Step 0
            # The pipeline can continue, but Steps 3 and 6 will fail

        # Verify repository write permissions
        if not verify_repo_permissions():
            error_msg = "Repository write permissions check failed"
            logger.error(error_msg)
            errors.append(error_msg)
            # This is critical - we can't proceed without write permissions

        # Ensure cache directory exists
        cache_manager.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Cache directory validated", path=str(cache_manager.cache_dir))

        # Backup existing cache if configured
        if config.backup_on_error:
            try:
                _backup_cache(cache_manager)
                cache_backed_up = True
                logger.info("Cache backed up successfully")
            except Exception as e:
                error_msg = f"Cache backup failed: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)

        # Cleanup old cache entries if configured
        if config.cleanup_on_start:
            try:
                retention_policy = {
                    "articles": config.retention["articles_days"],
                    "news": config.retention["news_days"],
                    "processed_articles": config.retention["articles_days"],
                    "clustered_articles": config.retention["articles_days"],
                    "selected_articles": config.retention["news_days"],
                    "enhanced_news": config.retention["news_days"],
                }

                initial_count = len(cache_manager.list_all())
                cache_manager.cleanup(retention_policy)
                final_count = len(cache_manager.list_all())
                cache_cleaned = initial_count - final_count

                logger.info(
                    "Cache cleanup completed",
                    cleaned=cache_cleaned,
                    remaining=final_count,
                )

            except Exception as e:
                error_msg = f"Cache cleanup failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Log current cache state
        cache_entries = cache_manager.list_all()
        logger.info(
            "Cache state",
            total_entries=len(cache_entries),
            entries=cache_entries,
        )

        result = Step0Result(
            success=len(errors) == 0,
            cache_cleaned=cache_cleaned,
            cache_backed_up=cache_backed_up,
            errors=errors,
        )

        logger.info(
            "Step 0 completed",
            success=result.success,
            cleaned=cache_cleaned,
            backed_up=cache_backed_up,
        )

        return result

    except Exception as e:
        logger.error("Step 0 failed critically", error=str(e), exc_info=True)
        return Step0Result(
            success=False,
            cache_cleaned=cache_cleaned,
            cache_backed_up=cache_backed_up,
            errors=[f"Critical failure: {e}"],
        )
    finally:
        # Always release lock file, even if an error occurred
        release_lock_file(cache_manager.cache_dir)


async def verify_gemini_health(api_key: str | None = None) -> bool:
    """
    Verify Gemini API availability with a simple test call.

    Args:
        api_key: Gemini API key (defaults to GEMINI_API_KEY env var)

    Returns:
        True if API is available and working, False otherwise
    """
    try:
        import google.generativeai as genai

        # Use provided key or fallback to environment variable
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            logger.error("Gemini API key not provided and GEMINI_API_KEY env var not set")
            return False

        genai.configure(api_key=key)  # type: ignore[attr-defined]

        # Make a simple test call with minimal tokens
        model = genai.GenerativeModel("gemini-2.0-flash-exp")  # type: ignore[attr-defined]
        response = await model.generate_content_async("Test")

        # Check if we got a valid response
        if response and response.text:
            logger.info("Gemini API health check passed")
            return True

        logger.warning("Gemini API returned empty response")
        return False

    except ImportError:
        logger.error("google-generativeai package not installed")
        return False
    except Exception as e:
        logger.error("Gemini API health check failed", error=str(e), exc_info=True)
        return False


def verify_repo_permissions() -> bool:
    """
    Verify that the repository has write permissions.

    Checks if we can write to the current directory (repository root).

    Returns:
        True if write permissions are available, False otherwise
    """
    try:
        # Try to create a temporary test file
        test_file = Path(".write_test_temp")

        test_file.write_text("test")
        test_file.unlink()

        logger.info("Repository write permissions verified")
        return True

    except PermissionError as e:
        logger.error("Insufficient repository write permissions", error=str(e))
        return False
    except Exception as e:
        logger.warning("Repository permissions check failed", error=str(e))
        return False


def acquire_lock_file(cache_dir: Path) -> bool:
    """
    Acquire a lock file to prevent concurrent pipeline executions.

    Args:
        cache_dir: Cache directory path

    Returns:
        True if lock acquired successfully, False if already locked
    """
    lock_file = cache_dir / ".lock"

    try:
        if lock_file.exists():
            # Check if lock is stale (> 1 hour old)
            import time

            lock_age = time.time() - lock_file.stat().st_mtime
            if lock_age > 3600:  # 1 hour in seconds
                logger.warning(
                    "Removing stale lock file",
                    age_seconds=lock_age,
                )
                lock_file.unlink()
            else:
                logger.error(
                    "Pipeline already running (lock file exists)",
                    lock_file=str(lock_file),
                )
                return False

        # Create lock file with current timestamp and PID
        lock_file.write_text(f"{datetime.now().isoformat()}\nPID: {os.getpid()}\n")
        logger.info("Lock file acquired", lock_file=str(lock_file))
        return True

    except Exception as e:
        logger.error("Failed to acquire lock file", error=str(e), exc_info=True)
        return False


def release_lock_file(cache_dir: Path) -> None:
    """
    Release the lock file after pipeline execution.

    Args:
        cache_dir: Cache directory path
    """
    lock_file = cache_dir / ".lock"

    try:
        if lock_file.exists():
            lock_file.unlink()
            logger.info("Lock file released")
        else:
            logger.debug("No lock file to release")

    except Exception as e:
        logger.warning("Failed to release lock file", error=str(e))


def _backup_cache(cache_manager: CacheManager) -> None:
    """
    Create a backup of the cache directory.

    Args:
        cache_manager: Cache manager instance
    """
    import shutil
    from datetime import datetime

    backup_dir = cache_manager.cache_dir.parent / "cache_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"cache_backup_{timestamp}"

    if cache_manager.cache_dir.exists():
        shutil.copytree(cache_manager.cache_dir, backup_path, dirs_exist_ok=True)
        logger.debug("Cache backed up", backup_path=str(backup_path))

    # Keep only last 5 backups
    backups = sorted(backup_dir.glob("cache_backup_*"))
    if len(backups) > 5:
        for old_backup in backups[:-5]:
            shutil.rmtree(old_backup)
            logger.debug("Old backup removed", path=str(old_backup))
