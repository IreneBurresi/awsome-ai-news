"""Unit tests for Step 1: RSS Ingestion."""

import asyncio

import aiohttp
import pytest
from aioresponses import aioresponses
from pydantic import HttpUrl

from src.models.articles import RawArticle
from src.models.config import FeedConfig, FeedFilter, Step1Config


class TestSlugGeneration:
    """Test slug generation functionality."""

    def test_generate_slug_4_words(self) -> None:
        """Test slug generation with >= 4 words."""
        from src.steps.step1_ingestion import generate_slug

        title = "Breaking AI News From OpenAI"
        slug = generate_slug(title, set())

        assert slug.startswith("breaking-ai-news-from-")
        parts = slug.split("-")
        assert len(parts) == 5  # 4 words + hash
        hash_part = parts[-1]
        assert len(hash_part) == 8  # hash length

    def test_generate_slug_less_than_4_words(self) -> None:
        """Test slug generation with < 4 words."""
        from src.steps.step1_ingestion import generate_slug

        title = "AI Wins"
        slug = generate_slug(title, set())

        assert slug.startswith("ai-wins-")
        parts = slug.split("-")
        assert len(parts) == 3  # 2 words + hash
        hash_part = parts[-1]
        assert len(hash_part) == 8

    def test_generate_slug_removes_punctuation(self) -> None:
        """Test that punctuation is removed from slug words."""
        from src.steps.step1_ingestion import generate_slug

        title = "AI's Big Win! Here's Why..."
        slug = generate_slug(title, set())

        # Should remove punctuation
        assert "'" not in slug
        assert "!" not in slug
        assert "." not in slug

    def test_generate_slug_lowercase(self) -> None:
        """Test that slug is lowercase."""
        from src.steps.step1_ingestion import generate_slug

        title = "BREAKING AI NEWS"
        slug = generate_slug(title, set())

        assert slug == slug.lower()

    def test_generate_slug_collision_handling(self) -> None:
        """Test slug collision handling with counter."""
        from src.steps.step1_ingestion import generate_slug

        title = "Same Title"
        slug1 = generate_slug(title, set())
        existing = {slug1}

        # Second generation should append _1
        slug2 = generate_slug(title, existing)

        assert slug2 == f"{slug1}_1"
        assert slug2 not in existing

    def test_generate_slug_multiple_collisions(self) -> None:
        """Test handling multiple collisions."""
        from src.steps.step1_ingestion import generate_slug

        title = "Same Title"
        slug1 = generate_slug(title, set())
        existing = {slug1, f"{slug1}_1", f"{slug1}_2"}

        slug3 = generate_slug(title, existing)

        assert slug3 == f"{slug1}_3"

    def test_generate_slug_too_many_collisions_raises(self) -> None:
        """Test that too many collisions raises error."""
        from src.steps.step1_ingestion import generate_slug

        title = "Same Title"
        slug1 = generate_slug(title, set())
        # Create 10 existing slugs
        existing = {f"{slug1}_{i}" if i > 0 else slug1 for i in range(10)}

        with pytest.raises(ValueError, match="Too many slug collisions"):
            generate_slug(title, existing)

    def test_generate_slug_deterministic(self) -> None:
        """Test that slug generation is deterministic."""
        from src.steps.step1_ingestion import generate_slug

        title = "Test Article Title"
        slug1 = generate_slug(title, set())
        slug2 = generate_slug(title, set())

        assert slug1 == slug2

    def test_generate_slug_different_for_different_titles(self) -> None:
        """Test that different titles produce different slugs."""
        from src.steps.step1_ingestion import generate_slug

        slug1 = generate_slug("Article One", set())
        slug2 = generate_slug("Article Two", set())

        assert slug1 != slug2

    @pytest.mark.parametrize(
        "title,expected_prefix",
        [
            ("AI Model", "ai-model-"),
            ("Deep Learning Breakthrough Discovery", "deep-learning-breakthrough-discovery-"),
            ("   Spaces   Around   Title   ", "spaces-around-title-"),
            ("Multiple---Dashes", "multiple-dashes-"),
        ],
    )
    def test_generate_slug_various_titles(self, title: str, expected_prefix: str) -> None:
        """Test slug generation with various title formats."""
        from src.steps.step1_ingestion import generate_slug

        slug = generate_slug(title, set())
        assert slug.startswith(expected_prefix)


class TestFeedFiltering:
    """Test feed article filtering functionality."""

    def test_filter_specialized_feed_no_filtering(self) -> None:
        """Test that specialized feeds accept all articles."""
        from src.steps.step1_ingestion import apply_filters

        article = RawArticle(
            title="Crypto and Blockchain News",
            url=HttpUrl("https://example.com"),
            feed_name="Test Feed",
            feed_priority=5,
        )

        # Specialized feeds have no filter
        results = apply_filters([article], None)

        assert len(results) == 1
        assert results[0] == (article, True, None)

    def test_filter_whitelist_keywords_match(self) -> None:
        """Test whitelist keywords matching."""
        from src.steps.step1_ingestion import apply_filters

        article = RawArticle(
            title="New AI Model Released",
            url=HttpUrl("https://example.com"),
            feed_name="Test",
            feed_priority=5,
        )

        filter_config = FeedFilter(whitelist_keywords=["AI", "machine learning"])

        results = apply_filters([article], filter_config)

        assert len(results) == 1
        assert results[0][1] is True  # passed
        assert results[0][2] is None  # no reason

    def test_filter_whitelist_keywords_no_match(self) -> None:
        """Test whitelist keywords not matching."""
        from src.steps.step1_ingestion import apply_filters

        article = RawArticle(
            title="Latest Smartphone News",
            url=HttpUrl("https://example.com"),
            feed_name="Test",
            feed_priority=5,
        )

        filter_config = FeedFilter(whitelist_keywords=["AI", "machine learning"])

        results = apply_filters([article], filter_config)

        assert len(results) == 1
        assert results[0][1] is False
        assert "whitelist" in results[0][2].lower()

    def test_filter_blacklist_keywords_match(self) -> None:
        """Test blacklist keywords blocking."""
        from src.steps.step1_ingestion import apply_filters

        article = RawArticle(
            title="AI and Crypto Together",
            url=HttpUrl("https://example.com"),
            feed_name="Test",
            feed_priority=5,
        )

        filter_config = FeedFilter(
            whitelist_keywords=["AI"], blacklist_keywords=["crypto", "blockchain"]
        )

        results = apply_filters([article], filter_config)

        assert len(results) == 1
        assert results[0][1] is False
        assert "blacklist" in results[0][2].lower()

    def test_filter_case_insensitive(self) -> None:
        """Test that filtering is case insensitive."""
        from src.steps.step1_ingestion import apply_filters

        article = RawArticle(
            title="ai model",  # lowercase
            url=HttpUrl("https://example.com"),
            feed_name="Test",
            feed_priority=5,
        )

        filter_config = FeedFilter(whitelist_keywords=["AI"])  # uppercase

        results = apply_filters([article], filter_config)

        assert results[0][1] is True

    def test_filter_checks_content_field(self) -> None:
        """Test that filtering checks content field when configured."""
        from src.steps.step1_ingestion import apply_filters

        article = RawArticle(
            title="Generic Title",
            url=HttpUrl("https://example.com"),
            content="This article discusses machine learning advances",
            feed_name="Test",
            feed_priority=5,
        )

        # Configure filter to check content field
        filter_config = FeedFilter(
            whitelist_keywords=["machine learning"], apply_to_fields=["title", "content"]
        )

        results = apply_filters([article], filter_config)

        assert results[0][1] is True

    def test_filter_whitelist_categories(self) -> None:
        """Test filtering by RSS categories."""
        from src.steps.step1_ingestion import apply_filters_with_categories

        article = RawArticle(
            title="Test Article",
            url=HttpUrl("https://example.com"),
            feed_name="Test",
            feed_priority=5,
        )

        categories = ["Technology", "AI", "Science"]
        filter_config = FeedFilter(whitelist_categories=["AI"])

        passed = apply_filters_with_categories(article, filter_config, categories)

        assert passed is True

    def test_filter_whitelist_categories_no_match(self) -> None:
        """Test category filtering with no match."""
        from src.steps.step1_ingestion import apply_filters_with_categories

        article = RawArticle(
            title="Test Article",
            url=HttpUrl("https://example.com"),
            feed_name="Test",
            feed_priority=5,
        )

        categories = ["Sports", "Gaming"]
        filter_config = FeedFilter(whitelist_categories=["AI"])

        passed = apply_filters_with_categories(article, filter_config, categories)

        assert passed is False

    def test_filter_whitelist_regex_match(self) -> None:
        """Test whitelist regex matching."""
        from src.steps.step1_ingestion import apply_filters

        article = RawArticle(
            title="New GPT-4 Model Released",
            url=HttpUrl("https://example.com"),
            feed_name="Test",
            feed_priority=5,
        )

        # Regex pattern to match GPT, Claude, Gemini
        filter_config = FeedFilter(
            whitelist_regex=r"\b(GPT|Claude|Gemini)\b",
        )

        results = apply_filters([article], filter_config)

        assert results[0][1] is True

    def test_filter_whitelist_regex_no_match(self) -> None:
        """Test whitelist regex not matching."""
        from src.steps.step1_ingestion import apply_filters

        article = RawArticle(
            title="Latest smartphone news",
            url=HttpUrl("https://example.com"),
            feed_name="Test",
            feed_priority=5,
        )

        filter_config = FeedFilter(
            whitelist_regex=r"\b(AI|GPT|Claude)\b",
        )

        results = apply_filters([article], filter_config)

        assert results[0][1] is False
        assert results[0][2] == "Whitelist regex not matched"

    def test_filter_blacklist_regex_match(self) -> None:
        """Test blacklist regex matching."""
        from src.steps.step1_ingestion import apply_filters

        article = RawArticle(
            title="Crypto trading bot announcement",
            url=HttpUrl("https://example.com"),
            feed_name="Test",
            feed_priority=5,
        )

        filter_config = FeedFilter(
            blacklist_regex=r"\b(crypto|blockchain|bitcoin)\b",
        )

        results = apply_filters([article], filter_config)

        assert results[0][1] is False
        assert results[0][2] == "Blacklist regex matched"

    def test_filter_regex_case_insensitive(self) -> None:
        """Test regex filters are case-insensitive."""
        from src.steps.step1_ingestion import apply_filters

        article = RawArticle(
            title="New ai model released",
            url=HttpUrl("https://example.com"),
            feed_name="Test",
            feed_priority=5,
        )

        filter_config = FeedFilter(
            whitelist_regex=r"\bAI\b",  # Uppercase pattern
        )

        results = apply_filters([article], filter_config)

        # Should match despite case difference
        assert results[0][1] is True

    def test_filter_combined_keywords_and_regex(self) -> None:
        """Test combining keyword and regex filters."""
        from src.steps.step1_ingestion import apply_filters

        article = RawArticle(
            title="Machine learning with GPT-4",
            url=HttpUrl("https://example.com"),
            feed_name="Test",
            feed_priority=5,
        )

        filter_config = FeedFilter(
            whitelist_keywords=["machine learning"],
            whitelist_regex=r"\bGPT-\d+\b",
        )

        results = apply_filters([article], filter_config)

        # Must pass both keyword and regex filters
        assert results[0][1] is True


class TestRSSFetching:
    """Test RSS feed fetching functionality."""

    @pytest.mark.asyncio
    async def test_fetch_single_feed_success(self) -> None:
        """Test successful feed fetching."""
        from datetime import datetime

        from src.steps.step1_ingestion import fetch_single_feed

        feed_config = FeedConfig(
            name="Test Feed",
            url="https://example.com/feed.rss",
            feed_type="specialized",
            priority=5,
        )

        # Use today's date to pass the 2-day filter
        today = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")

        mock_rss_content = f"""<?xml version="1.0"?>
<rss version="2.0">
    <channel>
        <title>Test Feed</title>
        <item>
            <title>Test Article</title>
            <link>https://example.com/article1</link>
            <description>Test description</description>
            <pubDate>{today}</pubDate>
        </item>
    </channel>
</rss>"""

        with aioresponses() as m:
            m.get("https://example.com/feed.rss", status=200, body=mock_rss_content)

            articles = await fetch_single_feed(feed_config)

            assert len(articles) == 1
            assert articles[0].title == "Test Article"
            assert str(articles[0].url) == "https://example.com/article1"
            assert articles[0].feed_name == "Test Feed"

    @pytest.mark.asyncio
    async def test_fetch_single_feed_timeout(self) -> None:
        """Test feed fetch timeout handling."""
        from src.steps.step1_ingestion import fetch_single_feed

        feed_config = FeedConfig(
            name="Slow Feed",
            url="https://example.com/feed.rss",
            feed_type="specialized",
            priority=5,
        )

        with aioresponses() as m:
            m.get("https://example.com/feed.rss", exception=TimeoutError())

            with pytest.raises(asyncio.TimeoutError):
                await fetch_single_feed(feed_config)

    @pytest.mark.asyncio
    async def test_fetch_single_feed_malformed_xml(self) -> None:
        """Test handling of malformed RSS feed."""
        from src.steps.step1_ingestion import fetch_single_feed

        feed_config = FeedConfig(
            name="Bad Feed",
            url="https://example.com/feed.rss",
            feed_type="specialized",
            priority=5,
        )

        bad_xml = "{ this is not XML }"

        with aioresponses() as m:
            m.get("https://example.com/feed.rss", status=200, body=bad_xml)

            with pytest.raises(ValueError, match="Malformed feed"):
                await fetch_single_feed(feed_config)

    @pytest.mark.asyncio
    async def test_fetch_single_feed_http_error(self) -> None:
        """Test HTTP error handling."""
        from src.steps.step1_ingestion import fetch_single_feed

        feed_config = FeedConfig(
            name="Error Feed",
            url="https://example.com/feed.rss",
            feed_type="specialized",
            priority=5,
        )

        with aioresponses() as m:
            m.get("https://example.com/feed.rss", status=404)

            with pytest.raises(aiohttp.ClientResponseError):
                await fetch_single_feed(feed_config)

    @pytest.mark.asyncio
    async def test_fetch_feed_with_atom_format(self) -> None:
        """Test fetching Atom format feed."""
        from datetime import datetime

        from src.steps.step1_ingestion import fetch_single_feed

        feed_config = FeedConfig(
            name="Atom Feed",
            url="https://example.com/atom.xml",
            feed_type="specialized",
            priority=5,
        )

        # Use today's date in ISO format for Atom
        today_iso = datetime.now().isoformat() + "Z"

        mock_atom_content = f"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
    <title>Atom Feed</title>
    <entry>
        <title>Atom Article</title>
        <link href="https://example.com/atom-article"/>
        <summary>Atom description</summary>
        <updated>{today_iso}</updated>
    </entry>
</feed>"""

        with aioresponses() as m:
            m.get("https://example.com/atom.xml", status=200, body=mock_atom_content)

            articles = await fetch_single_feed(feed_config)

            assert len(articles) == 1
            assert articles[0].title == "Atom Article"


class TestStep1Execution:
    """Test complete Step 1 execution."""

    @pytest.mark.asyncio
    async def test_run_step1_basic(self) -> None:
        """Test basic Step 1 execution."""
        from datetime import datetime

        from src.models.config import FeedsConfig
        from src.steps.step1_ingestion import run_step1
        from src.utils.cache import CacheManager

        config = Step1Config(enabled=True, max_concurrent_feeds=5)
        feeds_config = FeedsConfig(
            feeds=[
                FeedConfig(
                    name="Test Feed",
                    url="https://example.com/feed.rss",
                    feed_type="specialized",
                    priority=10,
                    enabled=True,
                )
            ]
        )

        # Use today's date
        today = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")

        mock_rss = f"""<?xml version="1.0"?>
<rss version="2.0">
    <channel>
        <item>
            <title>AI News Article</title>
            <link>https://example.com/1</link>
            <description>Description</description>
            <pubDate>{today}</pubDate>
        </item>
    </channel>
</rss>"""

        with aioresponses() as m:
            m.get("https://example.com/feed.rss", status=200, body=mock_rss)

            cache_manager = CacheManager()
            result = await run_step1(config, feeds_config, cache_manager)

            assert result.success is True
            assert result.feeds_fetched == 1
            assert result.feeds_failed == 0
            assert len(result.articles) >= 1

    @pytest.mark.asyncio
    async def test_run_step1_disabled(self) -> None:
        """Test Step 1 when disabled."""
        from src.models.config import FeedsConfig
        from src.steps.step1_ingestion import run_step1
        from src.utils.cache import CacheManager

        config = Step1Config(enabled=False)
        feeds_config = FeedsConfig(feeds=[])
        cache_manager = CacheManager()

        result = await run_step1(config, feeds_config, cache_manager)

        assert result.success is True
        assert result.feeds_fetched == 0
        assert len(result.articles) == 0

    @pytest.mark.asyncio
    async def test_run_step1_mixed_success_failure(self) -> None:
        """Test Step 1 with some feeds succeeding and some failing."""
        from src.models.config import FeedsConfig
        from src.steps.step1_ingestion import run_step1
        from src.utils.cache import CacheManager

        config = Step1Config(enabled=True, max_concurrent_feeds=10)
        feeds_config = FeedsConfig(
            feeds=[
                FeedConfig(
                    name="Good Feed",
                    url="https://example.com/good.rss",
                    feed_type="specialized",
                    priority=10,
                    enabled=True,
                ),
                FeedConfig(
                    name="Bad Feed",
                    url="https://example.com/bad.rss",
                    feed_type="specialized",
                    priority=8,
                    enabled=True,
                ),
            ]
        )

        good_rss = """<?xml version="1.0"?>
<rss version="2.0">
    <channel>
        <item>
            <title>Good Article</title>
            <link>https://example.com/good</link>
        </item>
    </channel>
</rss>"""

        with aioresponses() as m:
            m.get("https://example.com/good.rss", status=200, body=good_rss)
            m.get("https://example.com/bad.rss", status=404)

            cache_manager = CacheManager()
            result = await run_step1(config, feeds_config, cache_manager)

            assert result.success is True
            assert result.feeds_fetched == 1
            assert result.feeds_failed == 1
