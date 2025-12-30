"""Cache management utilities for pipeline data persistence."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from src.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class CacheManager:
    """Manages cache storage and retrieval for pipeline data."""

    def __init__(self, cache_dir: Path | str = "cache") -> None:
        """
        Initialize cache manager.

        Args:
            cache_dir: Base directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cache manager initialized", cache_dir=str(self.cache_dir))

    def _get_cache_path(self, key: str) -> Path:
        """
        Get cache file path for a key.

        Args:
            key: Cache key (e.g., 'articles', 'news')

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{key}.json"

    def save(self, key: str, data: list[BaseModel] | BaseModel) -> None:
        """
        Save data to cache.

        Args:
            key: Cache key
            data: Data to cache (Pydantic model or list of models)
        """
        cache_path = self._get_cache_path(key)

        try:
            if isinstance(data, list):
                json_data = [item.model_dump(mode="json") for item in data]
            else:
                json_data = data.model_dump(mode="json")

            cache_content = {
                "data": json_data,
                "cached_at": datetime.now().isoformat(),
            }

            cache_path.write_text(json.dumps(cache_content, indent=2))
            logger.info("Cache saved", key=key, path=str(cache_path))

        except Exception as e:
            logger.error("Failed to save cache", key=key, error=str(e))
            raise

    def load(self, key: str, model_class: type[T]) -> list[T] | T | None:
        """
        Load data from cache.

        Args:
            key: Cache key
            model_class: Pydantic model class to deserialize into

        Returns:
            List of model instances, single instance, or None if not found
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            logger.debug("Cache miss", key=key)
            return None

        try:
            cache_content = json.loads(cache_path.read_text())
            data = cache_content["data"]

            if isinstance(data, list):
                items = [model_class.model_validate(item) for item in data]
                logger.info("Cache loaded", key=key, count=len(items))
                return items
            else:
                item = model_class.model_validate(data)
                logger.info("Cache loaded", key=key)
                return item

        except Exception as e:
            logger.error(
                "Failed to load cache",
                key=key,
                error=str(e),
                model_class=model_class.__name__,
                exc_info=True,
            )
            return None

    def exists(self, key: str) -> bool:
        """
        Check if cache exists for a key.

        Args:
            key: Cache key

        Returns:
            True if cache exists
        """
        return self._get_cache_path(key).exists()

    def get_age(self, key: str) -> timedelta | None:
        """
        Get age of cached data.

        Args:
            key: Cache key

        Returns:
            Age as timedelta or None if not found
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            cache_content = json.loads(cache_path.read_text())
            cached_at = datetime.fromisoformat(cache_content["cached_at"])
            return datetime.now() - cached_at
        except Exception as e:
            logger.warning("Failed to get cache age", key=key, error=str(e))
            return None

    def is_fresh(self, key: str, max_age_days: int) -> bool:
        """
        Check if cache is fresh (within max age).

        Args:
            key: Cache key
            max_age_days: Maximum age in days

        Returns:
            True if cache is fresh
        """
        age = self.get_age(key)
        if age is None:
            return False

        is_fresh = age.days < max_age_days
        logger.debug(
            "Cache freshness check",
            key=key,
            age_days=age.days,
            max_age_days=max_age_days,
            is_fresh=is_fresh,
        )
        return is_fresh

    def delete(self, key: str) -> None:
        """
        Delete cache for a key.

        Args:
            key: Cache key
        """
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            cache_path.unlink()
            logger.info("Cache deleted", key=key)
        else:
            logger.debug("Cache not found for deletion", key=key)

    def cleanup(self, retention_days: dict[str, int]) -> None:
        """
        Clean up old cache files based on retention policy.

        Args:
            retention_days: Dict mapping cache keys to retention periods in days
                           e.g., {"articles": 10, "news": 3}
        """
        cleaned = 0

        for key, max_days in retention_days.items():
            if self.exists(key) and not self.is_fresh(key, max_days):
                self.delete(key)
                cleaned += 1

        logger.info("Cache cleanup completed", cleaned_count=cleaned)

    def list_all(self) -> list[str]:
        """
        List all cache keys.

        Returns:
            List of cache keys
        """
        return [p.stem for p in self.cache_dir.glob("*.json")]
