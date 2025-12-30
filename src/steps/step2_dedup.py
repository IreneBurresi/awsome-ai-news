"""Step 2: Article Deduplication based on slug matching."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from src.models.articles import DeduplicationStats, ProcessedArticle, Step2Result
from src.models.config import Step2Config
from src.utils.cache import CacheManager
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CachedArticlesDay(BaseModel):
    """Articles cached for a specific day."""

    date: datetime = Field(description="Reference date (YYYY-MM-DD)")
    articles: list[ProcessedArticle] = Field(description="Articles for the day")
    total_count: int = Field(ge=0, description="Article count")

    def model_post_init(self, __context) -> None:
        """Validate that total_count matches articles length."""
        if self.total_count != len(self.articles):
            raise ValueError("total_count must match articles length")


async def run_step2(
    config: Step2Config,
    articles: list[ProcessedArticle],
    cache_manager: CacheManager,
) -> Step2Result:
    """
    Execute Step 2: Article Deduplication.

    This step:
    1. Loads articles from cache (last 10 days)
    2. Deduplicates new articles by exact slug matching
    3. Saves unique articles to daily cache file

    Args:
        config: Step 2 configuration
        articles: Articles from Step 1
        cache_manager: Cache manager instance

    Returns:
        Step2Result with deduplicated articles and statistics
    """
    logger.info("Starting Step 2: Article deduplication")
    errors: list[str] = []

    try:
        # Create articles subdirectory if needed
        articles_cache_dir = cache_manager.cache_dir / "articles"
        articles_cache_dir.mkdir(parents=True, exist_ok=True)

        # Load cached articles from last N days
        cutoff_date = datetime.now() - timedelta(days=10)
        cached_articles, load_stats = _load_cached_articles(articles_cache_dir, cutoff_date)

        logger.info(
            "Loaded cached articles",
            count=len(cached_articles),
            files_loaded=load_stats["files_loaded"],
            files_corrupted=load_stats["files_corrupted"],
        )

        # Create slug map for O(1) lookup
        slug_map: dict[str, ProcessedArticle] = {art.slug: art for art in cached_articles}

        # Deduplicate new articles
        unique_articles = []
        duplicates = 0

        for article in articles:
            if article.slug in slug_map:
                duplicates += 1
                logger.debug(
                    "Duplicate article found",
                    slug=article.slug,
                    title=article.title[:50],
                )
                continue

            unique_articles.append(article)
            slug_map[article.slug] = article  # Add to prevent internal duplicates

        # Calculate statistics
        dedup_rate = duplicates / len(articles) if articles else 0.0

        stats = DeduplicationStats(
            input_articles=len(articles),
            cache_articles=len(cached_articles),
            duplicates_found=duplicates,
            unique_articles=len(unique_articles),
            deduplication_rate=dedup_rate,
            cache_files_loaded=load_stats["files_loaded"],
            cache_files_corrupted=load_stats["files_corrupted"],
        )

        # Save unique articles to daily cache file
        cache_updated = False
        if unique_articles:
            today = datetime.now()
            cache_file = articles_cache_dir / f"{today:%Y-%m-%d}.json"
            cache_updated = _save_articles_to_daily_cache(unique_articles, cache_file)

        logger.info(
            "Step 2 completed",
            input_articles=stats.input_articles,
            duplicates_found=stats.duplicates_found,
            unique_articles=stats.unique_articles,
            deduplication_rate=f"{stats.deduplication_rate:.1%}",
            cache_updated=cache_updated,
        )

        return Step2Result(
            success=True,
            unique_articles=unique_articles,
            stats=stats,
            cache_updated=cache_updated,
            errors=errors,
        )

    except Exception as e:
        error_msg = f"Step 2 failed: {e}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)

        return Step2Result(
            success=False,
            unique_articles=[],
            stats=DeduplicationStats(
                input_articles=len(articles),
                cache_articles=0,
                duplicates_found=0,
                unique_articles=0,
                deduplication_rate=0.0,
                cache_files_loaded=0,
                cache_files_corrupted=0,
            ),
            cache_updated=False,
            errors=errors,
        )


def _load_cached_articles(
    cache_dir: Path, cutoff_date: datetime
) -> tuple[list[ProcessedArticle], dict[str, int]]:
    """
    Load articles from cache files filtered by date.

    Args:
        cache_dir: Directory containing cache files (cache/articles/)
        cutoff_date: Minimum date to consider

    Returns:
        Tuple of (articles list, loading statistics)
    """
    articles: list[ProcessedArticle] = []
    files_loaded = 0
    files_corrupted = 0

    if not cache_dir.exists():
        logger.info("Cache directory does not exist, no cached articles")
        return articles, {"files_loaded": 0, "files_corrupted": 0}

    # Scan JSON cache files (YYYY-MM-DD.json format)
    for cache_file in sorted(cache_dir.glob("*.json")):
        try:
            # Parse date from filename
            file_date_str = cache_file.stem
            file_date = datetime.strptime(file_date_str, "%Y-%m-%d")

            # Skip if too old
            if file_date < cutoff_date:
                logger.debug("Skipping old cache file", file=cache_file.name)
                continue

            # Load and validate
            cache_data = json.loads(cache_file.read_text())

            # Validate as CachedArticlesDay
            cached_day = CachedArticlesDay.model_validate(cache_data)
            articles.extend(cached_day.articles)
            files_loaded += 1

            logger.debug(
                "Loaded cache file",
                file=cache_file.name,
                articles=len(cached_day.articles),
            )

        except (ValueError, json.JSONDecodeError, ValidationError) as e:
            logger.warning(
                "Corrupted cache file",
                file=cache_file.name,
                error=str(e),
            )
            files_corrupted += 1
            continue
        except Exception as e:
            logger.error(
                "Unexpected error loading cache",
                file=cache_file.name,
                error=str(e),
            )
            files_corrupted += 1
            continue

    logger.info(
        "Cache loading completed",
        total_articles=len(articles),
        files_loaded=files_loaded,
        files_corrupted=files_corrupted,
    )

    return articles, {"files_loaded": files_loaded, "files_corrupted": files_corrupted}


def _save_articles_to_daily_cache(articles: list[ProcessedArticle], cache_file: Path) -> bool:
    """
    Save articles to daily cache file.

    If the cache file for today already exists, loads existing articles
    and appends new ones to maintain accumulation across multiple runs.

    Args:
        articles: Articles to save
        cache_file: Path to cache file

    Returns:
        True if saved successfully
    """
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing articles from today's cache file if it exists
        existing_articles: list[ProcessedArticle] = []
        if cache_file.exists():
            try:
                cache_data = json.loads(cache_file.read_text())
                existing_day = CachedArticlesDay.model_validate(cache_data)
                existing_articles = existing_day.articles
                logger.debug(
                    "Loaded existing cache for today",
                    file=cache_file.name,
                    existing_count=len(existing_articles),
                )
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(
                    "Failed to load existing cache file, will overwrite",
                    file=cache_file.name,
                    error=str(e),
                )
                existing_articles = []

        # Combine existing and new articles, avoiding duplicates by slug
        all_articles_dict = {art.slug: art for art in existing_articles}
        for art in articles:
            all_articles_dict[art.slug] = art

        all_articles = list(all_articles_dict.values())

        cached_day = CachedArticlesDay(
            date=datetime.now(),
            articles=all_articles,
            total_count=len(all_articles),
        )

        # Save as JSON with indentation
        cache_file.write_text(
            json.dumps(
                cached_day.model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
            )
        )

        logger.info(
            "Saved articles to cache",
            file=cache_file.name,
            new_count=len(articles),
            total_count=len(all_articles),
        )
        return True

    except Exception as e:
        logger.error("Failed to save cache file", file=cache_file.name, error=str(e))
        return False
