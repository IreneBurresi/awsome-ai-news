"""Step 1: RSS Ingestion and Slug Generation."""

import asyncio
import hashlib
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import aiohttp
import feedparser
from loguru import logger

from src.constants import SLUG_HASH_LENGTH, SLUG_WORD_COUNT
from src.models.articles import ProcessedArticle, RawArticle, Step1Result
from src.models.config import FeedConfig, FeedFilter, FeedsConfig, Step1Config
from src.utils.cache import CacheManager

if TYPE_CHECKING:
    from feedparser import FeedParserDict


def generate_slug(title: str, existing_slugs: set[str]) -> str:
    """Generate unique slug: {first-N-words}-{sha256-hash[:M]}.

    Raises ValueError if >10 collisions occur.
    """
    # Normalize text
    normalized = title.lower().strip()
    # Remove punctuation, keep only alphanumeric and spaces/dashes
    normalized = re.sub(r"[^\w\s-]", "", normalized)
    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", normalized)
    # Normalize consecutive dashes to single dash
    normalized = re.sub(r"-+", "-", normalized)

    # Take first N words
    words = normalized.split()[:SLUG_WORD_COUNT]
    word_part = "-".join(words)

    # Generate hash from original title
    hash_input = title.strip().encode("utf-8")
    hash_hex = hashlib.sha256(hash_input).hexdigest()[:SLUG_HASH_LENGTH]

    # Create base slug
    base_slug = f"{word_part}-{hash_hex}"

    # Handle collisions with counter
    slug = base_slug
    counter = 1

    while slug in existing_slugs:
        if counter >= 10:
            logger.error(f"Too many slug collisions for title: {title}")
            raise ValueError(f"Too many slug collisions for title: {title}")

        slug = f"{base_slug}_{counter}"
        counter += 1

    return slug


def apply_filters(
    articles: list[RawArticle],
    filter_config: FeedFilter | None,
) -> list[tuple[RawArticle, bool, str | None]]:
    """Apply whitelist/blacklist filters to articles.

    Returns tuples of (article, passed: bool, rejection_reason: str | None).
    """
    if filter_config is None:
        # Specialized feed - accept all
        return [(article, True, None) for article in articles]

    results = []

    for article in articles:
        # Extract text to check based on apply_to_fields configuration
        texts_to_check = []

        for field in filter_config.apply_to_fields:
            value = getattr(article, field, None)
            if value:
                texts_to_check.append(value.lower())

        combined_text = " ".join(texts_to_check)

        # Check whitelist keywords
        if filter_config.whitelist_keywords:
            matched = any(kw.lower() in combined_text for kw in filter_config.whitelist_keywords)
            if not matched:
                results.append((article, False, "No whitelist keywords matched"))
                continue

        # Check whitelist regex
        if filter_config.whitelist_regex and not re.search(
            filter_config.whitelist_regex, combined_text, re.IGNORECASE
        ):
            results.append((article, False, "Whitelist regex not matched"))
            continue

        # Check blacklist keywords
        if filter_config.blacklist_keywords:
            blocked = any(kw.lower() in combined_text for kw in filter_config.blacklist_keywords)
            if blocked:
                results.append((article, False, "Blacklist keyword matched"))
                continue

        # Check blacklist regex
        if filter_config.blacklist_regex and re.search(
            filter_config.blacklist_regex, combined_text, re.IGNORECASE
        ):
            results.append((article, False, "Blacklist regex matched"))
            continue

        # Passed all filters
        results.append((article, True, None))

    return results


def apply_filters_with_categories(
    article: RawArticle,
    filter_config: FeedFilter,
    categories: list[str],
) -> bool:
    """
    Apply category filtering to an article.

    Args:
        article: Raw article
        filter_config: Filter configuration with whitelist_categories
        categories: RSS categories for this article

    Returns:
        True if article passes category filter, False otherwise
    """
    if not filter_config.whitelist_categories:
        return True

    # Check if any article category matches whitelist
    return any(cat in filter_config.whitelist_categories for cat in categories)


def _parse_entry_date(entry: "FeedParserDict") -> datetime | None:
    """
    Parse publication date from feed entry.

    Args:
        entry: Feed entry dictionary

    Returns:
        Parsed datetime or None if not available
    """
    # Try different date fields
    date_fields = ["published_parsed", "updated_parsed", "created_parsed"]

    for field in date_fields:
        if field in entry and entry[field]:
            try:
                import time

                time_tuple = entry[field]
                timestamp = time.mktime(time_tuple)
                return datetime.fromtimestamp(timestamp)
            except Exception as e:
                logger.debug(f"Failed to parse date from {field}: {e}")
                continue

    return None


async def fetch_single_feed(feed: FeedConfig, max_articles: int = 50) -> list[RawArticle]:
    """
    Fetch and parse a single RSS/Atom feed.

    Args:
        feed: Feed configuration
        max_articles: Maximum number of articles to return (default 50, most recent first)

    Returns:
        List of raw articles (limited to max_articles, sorted by date descending)

    Raises:
        aiohttp.ClientError: For HTTP errors
        asyncio.TimeoutError: For timeout
        ValueError: For malformed feed
    """
    headers = {"User-Agent": "awesome-ai-news-bot/1.0 (+https://github.com/user/awesome-ai-news)"}

    timeout = aiohttp.ClientTimeout(total=10)

    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.get(str(feed.url), headers=headers) as response,
    ):
        response.raise_for_status()
        content = await response.text()

    # Parse with feedparser (outside context to avoid scope issues)
    parsed = feedparser.parse(content)

    if parsed.bozo:  # Feed is malformed
        error_msg = str(parsed.get("bozo_exception", "Unknown parse error"))
        logger.warning(f"Malformed feed {feed.name}: {error_msg}")
        raise ValueError(f"Malformed feed: {error_msg}")

    articles = []

    for entry in parsed.entries:
        try:
            # Get link
            link = entry.get("link", "")
            if not link:
                logger.debug(f"Skipping entry without link in {feed.name}")
                continue

            # Get title
            title = entry.get("title", "").strip()
            if not title:
                logger.debug(f"Skipping entry without title in {feed.name}")
                continue

            # Get content/description
            content = None
            if "content" in entry and entry.content:
                content = entry.content[0].get("value", "")
            elif "summary" in entry:
                content = entry.summary
            elif "description" in entry:
                content = entry.description

            # Create raw article
            article = RawArticle(
                title=title,
                url=link,
                published_date=_parse_entry_date(entry),
                content=content,
                author=entry.get("author"),
                feed_name=feed.name,
                feed_priority=feed.priority,
            )

            articles.append(article)

        except Exception as e:
            logger.debug(f"Skipping invalid entry in {feed.name}: {e}")
            continue

    # Filter to last 2 days only
    cutoff_date = datetime.now() - timedelta(days=2)
    articles_before_filter = len(articles)
    articles = [art for art in articles if art.published_date and art.published_date >= cutoff_date]

    if articles_before_filter > len(articles):
        logger.debug(
            f"Filtered {articles_before_filter - len(articles)} old articles from {feed.name} "
            f"(older than 2 days)"
        )

    # Sort by published date (newest first) and limit to max_articles
    articles.sort(
        key=lambda x: x.published_date or datetime.min,
        reverse=True,
    )
    articles = articles[:max_articles]

    logger.info(f"Fetched {len(articles)} articles from {feed.name}")
    return articles


async def _fetch_feed_with_retry(
    feed: FeedConfig,
    max_articles: int = 50,
) -> tuple[FeedConfig, list[RawArticle] | Exception]:
    """
    Fetch feed with exception handling.

    Args:
        feed: Feed configuration
        max_articles: Maximum articles per feed

    Returns:
        Tuple of (feed, articles or exception)
    """
    try:
        articles = await fetch_single_feed(feed, max_articles)
        return (feed, articles)
    except Exception as e:
        logger.warning(f"Failed to fetch feed {feed.name}: {e}")
        return (feed, e)


async def run_step1(
    config: Step1Config,
    feeds_config: FeedsConfig,
    cache_manager: CacheManager,
) -> Step1Result:
    """
    Execute Step 1: RSS Ingestion.

    Fetches RSS feeds in parallel, applies filters, generates slugs,
    and caches results.

    Args:
        config: Step 1 configuration
        feeds_config: Feeds configuration
        cache_manager: Cache manager instance

    Returns:
        Step1Result with processed articles and statistics
    """
    if not config.enabled:
        logger.info("Step 1 is disabled, skipping")
        return Step1Result(
            success=True,
            articles=[],
            feeds_fetched=0,
            feeds_failed=0,
            total_articles_raw=0,
            articles_after_filter=0,
        )

    # Filter enabled feeds
    enabled_feeds = [f for f in feeds_config.feeds if f.enabled]

    if not enabled_feeds:
        logger.info("No enabled feeds, skipping Step 1")
        return Step1Result(
            success=True,
            articles=[],
            feeds_fetched=0,
            feeds_failed=0,
            total_articles_raw=0,
            articles_after_filter=0,
        )

    # Sort by priority (highest first)
    sorted_feeds = sorted(enabled_feeds, key=lambda x: x.priority, reverse=True)

    logger.info(f"Fetching {len(sorted_feeds)} feeds...")

    # Fetch feeds in parallel with semaphore for rate limiting
    semaphore = asyncio.Semaphore(config.max_concurrent_feeds)

    async def fetch_with_semaphore(feed: FeedConfig):
        async with semaphore:
            return await _fetch_feed_with_retry(feed, config.max_articles_per_feed)

    tasks = [fetch_with_semaphore(feed) for feed in sorted_feeds]
    feed_results = await asyncio.gather(*tasks)

    # Process results
    all_articles: list[ProcessedArticle] = []
    feeds_ok = 0
    feeds_fail = 0
    total_raw = 0
    existing_slugs: set[str] = set()
    slug_collision_count = 0

    for feed, result in feed_results:
        if isinstance(result, Exception):
            feeds_fail += 1
            continue

        feeds_ok += 1
        raw_articles = result
        total_raw += len(raw_articles)

        # Apply filters for generalist feeds
        if feed.feed_type == "generalist":
            filtered = apply_filters(raw_articles, feed.filter)
        else:
            filtered = [(art, True, None) for art in raw_articles]

        # Process filtered articles
        for raw_art, passed, reason in filtered:
            # Generate slug
            try:
                slug = generate_slug(raw_art.title, existing_slugs)
                if slug.endswith(tuple(f"_{i}" for i in range(1, 10))):
                    slug_collision_count += 1
                existing_slugs.add(slug)
            except ValueError as e:
                logger.error(f"Failed to generate slug for article: {e}")
                continue

            # Create processed article
            processed = ProcessedArticle(
                title=raw_art.title,
                url=raw_art.url,
                published_date=raw_art.published_date,
                content=raw_art.content,
                author=raw_art.author,
                feed_name=raw_art.feed_name,
                feed_priority=raw_art.feed_priority,
                slug=slug,
                content_hash="",  # Will be set in Step 2
            )

            # Only keep articles that passed filters
            if passed:
                all_articles.append(processed)
            else:
                logger.debug(f"Filtered out article '{raw_art.title}': {reason}")

    # Sort by published date (newest first)
    all_articles.sort(
        key=lambda x: x.published_date or datetime.min,
        reverse=True,
    )

    # Save to cache
    try:
        cache_manager.save("articles", all_articles)
        logger.info(f"Saved {len(all_articles)} articles to cache")
    except Exception as e:
        logger.error(f"Failed to save articles to cache: {e}")

    logger.info(
        f"Step 1 completed: {feeds_ok} feeds OK, {feeds_fail} failed, "
        f"{len(all_articles)} articles after filtering (from {total_raw} raw)"
    )

    return Step1Result(
        success=True,
        articles=all_articles,
        feeds_fetched=feeds_ok,
        feeds_failed=feeds_fail,
        total_articles_raw=total_raw,
        articles_after_filter=len(all_articles),
        slug_collisions=slug_collision_count,
    )
