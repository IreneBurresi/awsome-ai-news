"""Step 4: Multi-day News Deduplication.

Uses Gemini API to identify and merge semantically duplicate news clusters
across the last 3 days to avoid presenting the same news multiple times.
"""

from datetime import datetime, timedelta

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.config import Step4Config
from src.models.news import NewsCluster, NewsDeduplicationPair, Step4Result
from src.utils.cache import CacheManager


def _save_news_to_cache(cache_manager: CacheManager, news: list[NewsCluster]) -> None:
    """Save news clusters to cache with today's date.

    Args:
        cache_manager: Cache manager instance
        news: List of news clusters to save
    """
    from pathlib import Path

    # Ensure news cache directory exists
    news_cache_dir = Path(cache_manager.cache_dir) / "news"
    news_cache_dir.mkdir(parents=True, exist_ok=True)

    # Save with today's date
    today_str = datetime.now().strftime("%Y-%m-%d")
    cache_manager.save(f"news/news_{today_str}", news)


class GeminiDeduplicationResponse(BaseModel):
    """Structured response from Gemini for news deduplication.

    Only includes pairs that are actual duplicates (same event/story).
    """

    duplicate_pairs: list[NewsDeduplicationPair] = Field(
        description="List of news pairs that are duplicates (same event/story)"
    )
    rationale: str = Field(description="Overall deduplication strategy explanation")


async def run_step4(
    config: Step4Config,
    today_news: list[NewsCluster],
    cache_manager: CacheManager,
    api_key: str | None = None,
) -> Step4Result:
    """Execute Step 4: Multi-day news deduplication.

    Args:
        config: Step 4 configuration
        today_news: News clusters from today's Step 3
        cache_manager: Cache manager to load previous news
        api_key: Gemini API key (optional if disabled)

    Returns:
        Step4Result with deduplicated news

    Raises:
        ValueError: If config is invalid
    """
    try:
        logger.info("Starting Step 4: Multi-day news deduplication")

        if not config.enabled:
            logger.info("Step 4 disabled, returning original news")
            return Step4Result(
                success=True,
                unique_news=today_news,
                news_before_dedup=len(today_news),
                news_after_dedup=len(today_news),
                duplicates_found=0,
                news_merged=0,
                api_calls=0,
                api_failures=0,
            )

        # Handle empty input
        if not today_news:
            logger.info("No news to deduplicate")
            return Step4Result(
                success=True,
                unique_news=[],
                news_before_dedup=0,
                news_after_dedup=0,
                duplicates_found=0,
                news_merged=0,
                api_calls=0,
                api_failures=0,
            )

        api_calls = 0
        api_failures = 0
        errors: list[str] = []
        fallback_used = False
        duplicates_found = 0
        news_merged = 0

        # Load cached news from last N days
        logger.info(f"Loading cached news from last {config.lookback_days} days")
        cached_news = _load_cached_news(cache_manager, config.lookback_days)
        logger.info(
            f"Loaded {len(cached_news)} news from cache",
            lookback_days=config.lookback_days,
        )

        # If no cached news, save today's news and return as-is
        if not cached_news:
            logger.info("No cached news found, no deduplication needed")
            logger.debug("Saving today's news to cache for future deduplication")
            _save_news_to_cache(cache_manager, today_news)
            return Step4Result(
                success=True,
                unique_news=today_news,
                news_before_dedup=len(today_news),
                news_after_dedup=len(today_news),
                duplicates_found=0,
                news_merged=0,
                api_calls=0,
                api_failures=0,
            )

        # Check for API key
        if not api_key:
            logger.warning("No API key provided")
            if config.fallback_to_no_merge:
                logger.warning("Using fallback: no merge, keeping all news")
                logger.debug("Saving news to cache despite fallback")
                _save_news_to_cache(cache_manager, today_news)
                fallback_used = True
                return Step4Result(
                    success=True,
                    unique_news=today_news,
                    news_before_dedup=len(today_news),
                    news_after_dedup=len(today_news),
                    duplicates_found=0,
                    news_merged=0,
                    api_calls=0,
                    api_failures=0,
                    fallback_used=True,
                    errors=["No API key provided"],
                )
            else:
                return Step4Result(
                    success=False,
                    unique_news=[],
                    news_before_dedup=len(today_news),
                    news_after_dedup=0,
                    duplicates_found=0,
                    news_merged=0,
                    api_calls=0,
                    api_failures=0,
                    errors=["No API key provided and fallback disabled"],
                )

        try:
            # Call Gemini API for semantic deduplication
            logger.info("Calling Gemini API for semantic deduplication")
            dedup_response = await _call_gemini_deduplication(
                today_news, cached_news, config, api_key
            )
            api_calls += 1

            duplicate_pairs = dedup_response.duplicate_pairs
            duplicates_found = len(duplicate_pairs)

            logger.info(
                f"Gemini identified {duplicates_found} duplicate pairs",
                today_news_count=len(today_news),
                cached_news_count=len(cached_news),
            )

            # Merge duplicate news
            unique_news = _merge_duplicate_news(today_news, cached_news, duplicate_pairs)
            news_merged = duplicates_found  # Each pair results in one merge

            logger.info(
                "Deduplication complete",
                before=len(today_news),
                after=len(unique_news),
                merged=news_merged,
            )

        except Exception as e:
            error_msg = f"Gemini API call failed: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            api_failures += 1

            # Use fallback if configured
            if config.fallback_to_no_merge:
                logger.warning("Using fallback: no merge, keeping all news")
                unique_news = today_news
                fallback_used = True
                # Save to cache even with fallback
                logger.debug("Saving news to cache despite API failure fallback")
                _save_news_to_cache(cache_manager, unique_news)
            else:
                return Step4Result(
                    success=False,
                    unique_news=[],
                    news_before_dedup=len(today_news),
                    news_after_dedup=0,
                    duplicates_found=0,
                    news_merged=0,
                    api_calls=api_calls,
                    api_failures=api_failures,
                    errors=errors,
                )

        # Save deduplicated news to cache
        logger.debug("Saving deduplicated news to cache")
        _save_news_to_cache(cache_manager, unique_news)

        logger.info("Step 4 completed successfully")

        return Step4Result(
            success=True,
            unique_news=unique_news,
            news_before_dedup=len(today_news),
            news_after_dedup=len(unique_news),
            duplicates_found=duplicates_found,
            news_merged=news_merged,
            api_calls=api_calls,
            api_failures=api_failures,
            errors=errors,
            fallback_used=fallback_used,
        )

    except Exception as e:
        error_msg = f"Step 4 failed critically: {e}"
        logger.error(error_msg, exc_info=True)
        return Step4Result(
            success=False,
            unique_news=[],
            news_before_dedup=len(today_news) if today_news else 0,
            news_after_dedup=0,
            duplicates_found=0,
            news_merged=0,
            api_calls=0,
            api_failures=0,
            errors=[error_msg],
        )


def _load_cached_news(cache_manager: CacheManager, lookback_days: int) -> list[NewsCluster]:
    """Load news clusters from cache for the last N days.

    Args:
        cache_manager: Cache manager instance
        lookback_days: Number of days to look back

    Returns:
        List of NewsCluster from cache (last N days)
    """
    from pathlib import Path

    all_cached_news: list[NewsCluster] = []
    cutoff_date = datetime.now() - timedelta(days=lookback_days)

    # Scan cache directory for news files
    cache_dir = Path(cache_manager.cache_dir) / "news"
    if not cache_dir.exists():
        logger.debug("News cache directory does not exist")
        return []

    # Load news from each daily cache file
    for news_file in sorted(cache_dir.glob("*.json")):
        try:
            # Extract date from filename (e.g., news_2024-12-24.json)
            date_str = news_file.stem.split("_")[-1]
            file_date = datetime.strptime(date_str, "%Y-%m-%d")

            # Skip files older than lookback window
            if file_date < cutoff_date:
                logger.debug(f"Skipping old cache file: {news_file.name}")
                continue

            # Load news from file
            news_list = cache_manager.load(f"news/{news_file.stem}", NewsCluster)
            if news_list:
                all_cached_news.extend(news_list)
                logger.debug(f"Loaded {len(news_list)} news from {news_file.name}")

        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse date from {news_file.name}: {e}")
            continue

    logger.info(f"Loaded {len(all_cached_news)} total news from cache")
    return all_cached_news


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
async def _call_gemini_deduplication(
    today_news: list[NewsCluster],
    cached_news: list[NewsCluster],
    config: Step4Config,
    api_key: str,
) -> GeminiDeduplicationResponse:
    """Call Gemini API to identify duplicate news pairs.

    Args:
        today_news: News clusters from today
        cached_news: News clusters from cache (last N days)
        config: Step 4 configuration
        api_key: Gemini API key

    Returns:
        GeminiDeduplicationResponse with duplicate pairs

    Raises:
        Exception: On API failures after retries
    """
    from google import genai

    from src.utils.prompt_loader import get_prompt_loader

    # Create client
    client = genai.Client(api_key=api_key)

    # Prepare news data for prompt
    today_data = _prepare_news_for_prompt(today_news)
    cached_data = _prepare_news_for_prompt(cached_news)

    # Load and format prompt from YAML
    prompt_loader = get_prompt_loader()
    prompt = prompt_loader.format_prompt(
        "step4_multi_dedup",
        num_today_news=len(today_news),
        today_news_formatted=today_data,
        num_cached_news=len(cached_news),
        cached_news_formatted=cached_data,
        lookback_days=config.lookback_days,
    )

    logger.debug("Calling Gemini API for deduplication", model=config.llm_model)

    # Make API call with structured output (using Pydantic class directly)
    response = client.models.generate_content(
        model=config.llm_model,
        contents=prompt,
        config={
            "temperature": config.temperature,
            "response_mime_type": "application/json",
            "response_schema": GeminiDeduplicationResponse,
        },
    )

    logger.debug("Gemini API response received", response_text=response.text[:200])

    # Parse and validate with Pydantic
    dedup_response = GeminiDeduplicationResponse.model_validate_json(response.text)

    return dedup_response


def _prepare_news_for_prompt(news_list: list[NewsCluster]) -> str:
    """Format news for inclusion in prompt.

    Args:
        news_list: List of news clusters

    Returns:
        Formatted string for prompt
    """
    lines = []
    for i, news in enumerate(news_list, 1):
        lines.append(
            f"{i}. [ID: {news.news_id}] {news.title}\n"
            f"   Summary: {news.summary[:200]}...\n"
            f"   Topic: {news.main_topic} | Articles: {news.article_count} | "
            f"Created: {news.created_at.strftime('%Y-%m-%d %H:%M')}"
        )

    return "\n\n".join(lines)


def _merge_duplicate_news(
    today_news: list[NewsCluster],
    cached_news: list[NewsCluster],
    duplicate_pairs: list[NewsDeduplicationPair],
) -> list[NewsCluster]:
    """Merge duplicate news by combining article slugs.

    Strategy:
    - Keep the news with more articles as the base
    - Merge article slugs from the duplicate
    - Update article_count
    - Set updated_at timestamp
    - Remove the duplicate from today's news

    Args:
        today_news: News clusters from today
        cached_news: News clusters from cache
        duplicate_pairs: Pairs identified as duplicates

    Returns:
        List of unique news (merged + non-duplicates)
    """
    # Create lookup maps
    today_map = {news.news_id: news for news in today_news}
    cached_map = {news.news_id: news for news in cached_news}

    # Track which today news have been merged
    merged_today_ids = set()

    # Result list starting with all cached news
    result_news: list[NewsCluster] = list(cached_news)

    # Process each duplicate pair (all pairs are meant to be merged)
    for pair in duplicate_pairs:
        today_item = today_map.get(pair.news_today_id)
        cached_item = cached_map.get(pair.news_cached_id)

        if not today_item or not cached_item:
            logger.warning(
                f"Could not find news for merge pair: "
                f"today={pair.news_today_id}, cached={pair.news_cached_id}"
            )
            continue

        # Determine which news to keep as base (more articles)
        if today_item.article_count >= cached_item.article_count:
            base = today_item
            to_merge = cached_item
            logger.debug(
                f"Merging into today news (more articles): {base.news_id}",
                base_count=base.article_count,
                merge_count=to_merge.article_count,
            )
        else:
            base = cached_item
            to_merge = today_item
            logger.debug(
                f"Merging into cached news (more articles): {base.news_id}",
                base_count=base.article_count,
                merge_count=to_merge.article_count,
            )

        # Merge article slugs (avoid duplicates)
        merged_slugs = list(set(base.article_slugs + to_merge.article_slugs))

        # Create updated news cluster
        updated_news = NewsCluster(
            news_id=base.news_id,
            title=base.title,
            summary=base.summary,
            article_slugs=merged_slugs,
            article_count=len(merged_slugs),
            main_topic=base.main_topic,
            keywords=list(set(base.keywords + to_merge.keywords))[
                :10
            ],  # Merge keywords, limit to 10
            created_at=base.created_at,
            updated_at=datetime.utcnow(),  # Mark as updated
        )

        # Update in result list
        if base == cached_item:
            # Find and replace cached news in result
            for i, news in enumerate(result_news):
                if news.news_id == cached_item.news_id:
                    result_news[i] = updated_news
                    break
        else:
            # Base is today news, need to add to result and remove cached
            result_news = [n for n in result_news if n.news_id != cached_item.news_id]
            result_news.append(updated_news)

        # Mark today news as merged
        merged_today_ids.add(pair.news_today_id)

        logger.info(
            f"Merged news: {base.news_id}",
            articles_before=base.article_count + to_merge.article_count,
            articles_after=len(merged_slugs),
            reason=pair.merge_reason,
        )

    # Add non-merged today news to result
    for news in today_news:
        if news.news_id not in merged_today_ids:
            result_news.append(news)

    return result_news
