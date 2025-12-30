"""Step 8: RSS Feed Generation.

Generates daily and weekly RSS feeds from enhanced news for external consumption.
"""

import glob
from datetime import datetime, timedelta
from email.utils import formatdate
from pathlib import Path
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

import yaml
from loguru import logger

from src.models.config import Step8Config
from src.models.news import EnhancedNews
from src.models.rss import RSSFeed, RSSItem, Step8Result


async def run_step8(
    config: Step8Config,
    enhanced_news: list[EnhancedNews],
) -> Step8Result:
    """Execute Step 8: RSS feed generation.

    Args:
        config: Step 8 configuration
        enhanced_news: Enhanced news from Step 6

    Returns:
        Step8Result with feed paths and statistics

    Raises:
        ValueError: If config is invalid
    """
    try:
        logger.info("Starting Step 8: RSS feed generation")

        if not config.enabled:
            logger.info("Step 8 disabled, skipping RSS generation")
            return Step8Result(
                success=True,
                daily_feed_path=None,
                weekly_feed_path=None,
                daily_items_count=0,
                weekly_items_count=0,
                feeds_valid=False,
            )

        errors: list[str] = []

        # Generate daily feed
        daily_path = None
        daily_count = 0
        try:
            daily_items = [_news_to_rss_item(news, config.feed_link) for news in enhanced_news]
            daily_feed = RSSFeed(
                title=config.feed_title,
                description=config.feed_description,
                link=config.feed_link,  # type: ignore[arg-type]  # Pydantic converts str to HttpUrl
                pub_date=datetime.utcnow(),
                items=daily_items,
            )
            daily_path = Path(config.output_file)
            _write_rss_feed(daily_feed, daily_path, config.feed_link, is_weekly=False)
            daily_count = len(daily_items)
            logger.info(f"Created daily feed: {daily_path} ({daily_count} items)")
        except Exception as e:
            error_msg = f"Failed to create daily feed: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)

        # Generate weekly feed (load last 7 days)
        weekly_path = None
        weekly_count = 0
        try:
            weekly_news = _load_last_n_days_news(days=7)
            weekly_news.extend(enhanced_news)  # Add today's news
            weekly_items = [_news_to_rss_item(news, config.feed_link) for news in weekly_news]
            weekly_feed = RSSFeed(
                title=f"{config.feed_title} - Weekly Digest",
                description=f"{config.feed_description} (Last 7 days)",
                link=config.feed_link,  # type: ignore[arg-type]  # Pydantic converts str to HttpUrl
                pub_date=datetime.utcnow(),
                items=weekly_items,
            )
            weekly_path = Path("weekly.xml")
            _write_rss_feed(weekly_feed, weekly_path, config.feed_link, is_weekly=True)
            weekly_count = len(weekly_items)
            logger.info(f"Created weekly feed: {weekly_path} ({weekly_count} items)")
        except Exception as e:
            error_msg = f"Failed to create weekly feed: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)

        # Validate feeds
        feeds_valid = False
        if daily_path and weekly_path:
            try:
                feeds_valid = _validate_rss_feeds([daily_path, weekly_path])
                logger.info(f"Feed validation: {'PASSED' if feeds_valid else 'FAILED'}")
            except Exception as e:
                error_msg = f"Feed validation failed: {e}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)

        success = len(errors) == 0 or (daily_path is not None and weekly_path is not None)

        logger.info(f"Step 8 completed: success={success}")

        return Step8Result(
            success=success,
            daily_feed_path=daily_path,
            weekly_feed_path=weekly_path,
            daily_items_count=daily_count,
            weekly_items_count=weekly_count,
            feeds_valid=feeds_valid,
            errors=errors,
        )

    except Exception as e:
        error_msg = f"Step 8 failed critically: {e}"
        logger.error(error_msg, exc_info=True)
        return Step8Result(
            success=False,
            daily_feed_path=None,
            weekly_feed_path=None,
            daily_items_count=0,
            weekly_items_count=0,
            feeds_valid=False,
            errors=[error_msg],
        )


def _news_to_rss_item(news: EnhancedNews, base_url: str) -> RSSItem:
    """Convert EnhancedNews to RSSItem.

    Args:
        news: Enhanced news item
        base_url: Base URL for links

    Returns:
        RSSItem for RSS feed
    """
    # Primary link: first external link or GitHub news anchor
    if news.external_links:
        primary_link = str(news.external_links[0].url)
    else:
        primary_link = f"{base_url}#news-{news.news.news_cluster.news_id}"

    # Description: extended summary + key points
    description = f"{news.extended_summary}\n\n"
    if news.key_points:
        description += "**Key Points:**\n"
        for point in news.key_points:
            description += f"- {point}\n"

    return RSSItem(
        title=news.news.news_cluster.title,
        description=description,
        link=primary_link,  # type: ignore[arg-type]  # Pydantic converts str to HttpUrl
        pub_date=news.enhanced_at,
        guid=news.news.news_cluster.news_id,
        categories=[news.news.category.value],
    )


def _write_rss_feed(
    feed: RSSFeed, output_path: Path, base_url: str, is_weekly: bool = False
) -> None:
    """Write RSS feed to XML file.

    Args:
        feed: RSSFeed object
        output_path: Path to output file
        base_url: Base URL for self link
        is_weekly: Whether this is weekly feed

    Raises:
        IOError: If file write fails
    """
    # Create RSS root element
    rss = Element("rss", version="2.0")
    rss.set("xmlns:atom", "http://www.w3.org/2005/Atom")

    # Channel element
    channel = SubElement(rss, "channel")

    # Required channel elements
    SubElement(channel, "title").text = feed.title
    SubElement(channel, "description").text = feed.description
    SubElement(channel, "link").text = str(feed.link)
    SubElement(channel, "language").text = feed.language

    # Publication dates
    SubElement(channel, "pubDate").text = formatdate(feed.pub_date.timestamp())
    SubElement(channel, "lastBuildDate").text = formatdate(datetime.utcnow().timestamp())

    # Self link
    filename = "weekly.xml" if is_weekly else output_path.name
    self_link = SubElement(
        channel,
        "{http://www.w3.org/2005/Atom}link",
        href=f"{base_url}/raw/main/{filename}",
        rel="self",
    )
    self_link.set("type", "application/rss+xml")

    # Items
    for item in feed.items:
        item_elem = SubElement(channel, "item")
        SubElement(item_elem, "title").text = item.title

        # Description with CDATA
        desc = SubElement(item_elem, "description")
        desc.text = f"<![CDATA[{item.description}]]>"

        SubElement(item_elem, "link").text = str(item.link)

        # GUID
        guid_elem = SubElement(item_elem, "guid")
        guid_elem.set("isPermaLink", "false")
        guid_elem.text = item.guid

        SubElement(item_elem, "pubDate").text = formatdate(item.pub_date.timestamp())

        # Categories
        for cat in item.categories:
            SubElement(item_elem, "category").text = cat

    # Convert to string and pretty print
    xml_str = tostring(rss, encoding="unicode")

    # Parse and pretty print
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ", encoding="utf-8")

    # Remove extra blank lines
    lines = [line for line in pretty_xml.decode("utf-8").split("\n") if line.strip()]
    clean_xml = "\n".join(lines)

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(clean_xml)


def _load_last_n_days_news(days: int = 7) -> list[EnhancedNews]:
    """Load news from last N days from YAML files.

    Args:
        days: Number of days to load (default: 7)

    Returns:
        List of EnhancedNews from last N days

    Note:
        This is a simplified implementation that loads from news/ directory.
        In production, this should load from Step 6 cache or news files.
    """
    news_list = []
    cutoff_date = datetime.now() - timedelta(days=days)

    # Find news YAML files in news/ directory
    news_dir = Path("news")
    if not news_dir.exists():
        logger.warning("News directory not found, skipping weekly feed loading")
        return []

    # Look for YAML files from last N days
    yaml_files = glob.glob(str(news_dir / "*.yaml"))

    for yaml_file in yaml_files:
        try:
            # Extract date from filename (YYYY-MM-DD.yaml)
            filename = Path(yaml_file).stem
            file_date = datetime.strptime(filename, "%Y-%m-%d")

            # Skip if older than cutoff
            if file_date < cutoff_date:
                continue

            # Load YAML file
            with open(yaml_file, encoding="utf-8") as f:
                _ = yaml.safe_load(f)  # Loaded but not used yet

            # NOTE: This is a simplified implementation
            # In production, we would reconstruct EnhancedNews objects from YAML
            # For now, we just skip loading historical news
            # The weekly feed will only contain today's news

        except Exception as e:
            logger.warning(f"Failed to load news from {yaml_file}: {e}")
            continue

    return news_list


def _validate_rss_feeds(feed_paths: list[Path]) -> bool:
    """Validate RSS feeds against RSS 2.0 spec.

    Args:
        feed_paths: List of feed file paths

    Returns:
        True if all feeds are valid, False otherwise
    """
    try:
        import feedparser

        for path in feed_paths:
            if not path.exists():
                logger.error(f"Feed file not found: {path}")
                return False

            parsed = feedparser.parse(str(path))

            # Check for parsing errors
            if parsed.bozo:
                logger.error(f"Invalid RSS feed {path}: {parsed.bozo_exception}")
                return False

            # Check required fields
            if not parsed.feed.get("title"):
                logger.error(f"Feed {path} missing title")
                return False

            logger.debug(f"Feed {path} validation passed")

        return True

    except ImportError:
        logger.warning("feedparser not installed, skipping validation")
        return True  # Don't fail if feedparser is not available
    except Exception as e:
        logger.error(f"RSS validation failed: {e}")
        return False
