"""Integration tests for Step 1 with real RSS feeds and full pipeline flow."""

from datetime import datetime
from pathlib import Path

import pytest
from aioresponses import aioresponses

from src.models.config import FeedConfig, FeedsConfig, Step1Config
from src.steps.step1_ingestion import run_step1
from src.utils.cache import CacheManager


@pytest.fixture
def temp_cache_for_step1(tmp_path: Path) -> CacheManager:
    """Create temporary cache for Step 1 tests."""
    cache_dir = tmp_path / "step1_cache"
    cache_dir.mkdir()
    return CacheManager(cache_dir=cache_dir)


@pytest.fixture
def sample_rss_content() -> str:
    """Sample valid RSS 2.0 content."""
    today = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>AI News Feed</title>
        <link>https://example.com</link>
        <description>Latest AI news</description>
        <item>
            <title>OpenAI Releases New GPT Model</title>
            <link>https://example.com/news/gpt-release</link>
            <description>OpenAI announced a new GPT model with improved capabilities</description>
            <pubDate>{today}</pubDate>
            <author>John Doe</author>
        </item>
        <item>
            <title>Machine Learning Breakthrough in Healthcare</title>
            <link>https://example.com/news/ml-healthcare</link>
            <description>Researchers made breakthrough in medical diagnosis using AI</description>
            <pubDate>{today}</pubDate>
        </item>
        <item>
            <title>Crypto Markets Rally Despite Regulations</title>
            <link>https://example.com/news/crypto-rally</link>
            <description>Cryptocurrency markets see gains</description>
            <pubDate>{today}</pubDate>
        </item>
    </channel>
</rss>"""


@pytest.fixture
def sample_atom_content() -> str:
    """Sample valid Atom 1.0 content."""
    today_iso = datetime.now().isoformat() + "Z"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
    <title>AI Research Feed</title>
    <link href="https://research.example.com"/>
    <updated>{today_iso}</updated>
    <entry>
        <title>New Neural Network Architecture</title>
        <link href="https://research.example.com/neural-net"/>
        <summary>Novel architecture improves training efficiency</summary>
        <updated>{today_iso}</updated>
        <author>
            <name>Jane Smith</name>
        </author>
    </entry>
</feed>"""


class TestStep1IntegrationBasic:
    """Basic integration tests for Step 1."""

    @pytest.mark.asyncio
    async def test_step1_single_specialized_feed(
        self, temp_cache_for_step1: CacheManager, sample_rss_content: str
    ) -> None:
        """Test fetching a single specialized feed."""
        config = Step1Config(enabled=True, max_concurrent_feeds=10, timeout_seconds=10)

        feeds_config = FeedsConfig(
            feeds=[
                FeedConfig(
                    name="AI News",
                    url="https://example.com/feed.rss",
                    feed_type="specialized",
                    priority=10,
                    enabled=True,
                )
            ]
        )

        with aioresponses() as m:
            m.get("https://example.com/feed.rss", status=200, body=sample_rss_content)

            result = await run_step1(config, feeds_config, temp_cache_for_step1)

            assert result.success is True
            assert result.feeds_fetched == 1
            assert result.feeds_failed == 0
            assert len(result.articles) == 3  # All articles from specialized feed
            assert result.total_articles_raw == 3

            # Check articles have slugs
            for article in result.articles:
                assert article.slug is not None
                assert len(article.slug) > 0
                assert "-" in article.slug

    @pytest.mark.asyncio
    async def test_step1_generalist_feed_with_filters(
        self, temp_cache_for_step1: CacheManager, sample_rss_content: str
    ) -> None:
        """Test generalist feed with keyword filtering."""
        config = Step1Config(enabled=True)

        feeds_config = FeedsConfig(
            feeds=[
                FeedConfig(
                    name="Tech News",
                    url="https://example.com/feed.rss",
                    feed_type="generalist",
                    priority=7,
                    enabled=True,
                    filter={
                        "whitelist_keywords": ["AI", "machine learning", "GPT"],
                        "blacklist_keywords": ["crypto", "cryptocurrency"],
                    },
                )
            ]
        )

        with aioresponses() as m:
            m.get("https://example.com/feed.rss", status=200, body=sample_rss_content)

            result = await run_step1(config, feeds_config, temp_cache_for_step1)

            assert result.success is True
            assert result.total_articles_raw == 3
            # Should filter out crypto article
            assert result.articles_after_filter == 2
            assert len(result.articles) == 2

            # Verify filtered articles
            titles = [a.title for a in result.articles]
            assert "Crypto Markets" not in " ".join(titles)

    @pytest.mark.asyncio
    async def test_step1_multiple_feeds_parallel(
        self, temp_cache_for_step1: CacheManager, sample_rss_content: str, sample_atom_content: str
    ) -> None:
        """Test fetching multiple feeds in parallel."""
        config = Step1Config(enabled=True, max_concurrent_feeds=10)

        feeds_config = FeedsConfig(
            feeds=[
                FeedConfig(
                    name="RSS Feed",
                    url="https://example.com/rss.xml",
                    feed_type="specialized",
                    priority=10,
                    enabled=True,
                ),
                FeedConfig(
                    name="Atom Feed",
                    url="https://example.com/atom.xml",
                    feed_type="specialized",
                    priority=9,
                    enabled=True,
                ),
            ]
        )

        with aioresponses() as m:
            m.get("https://example.com/rss.xml", status=200, body=sample_rss_content)
            m.get("https://example.com/atom.xml", status=200, body=sample_atom_content)

            result = await run_step1(config, feeds_config, temp_cache_for_step1)

            assert result.success is True
            assert result.feeds_fetched == 2
            assert result.feeds_failed == 0
            assert len(result.articles) == 4  # 3 from RSS + 1 from Atom

    @pytest.mark.asyncio
    async def test_step1_feed_failure_continues_processing(
        self, temp_cache_for_step1: CacheManager, sample_rss_content: str
    ) -> None:
        """Test that failed feeds don't stop processing of other feeds."""
        config = Step1Config(enabled=True)

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
                    priority=9,
                    enabled=True,
                ),
            ]
        )

        with aioresponses() as m:
            m.get("https://example.com/good.rss", status=200, body=sample_rss_content)
            m.get("https://example.com/bad.rss", status=404)

            result = await run_step1(config, feeds_config, temp_cache_for_step1)

            assert result.success is True
            assert result.feeds_fetched == 1
            assert result.feeds_failed == 1
            assert len(result.articles) == 3  # Only from good feed

    @pytest.mark.asyncio
    async def test_step1_disabled_config(self, temp_cache_for_step1: CacheManager) -> None:
        """Test Step 1 when disabled in config."""
        config = Step1Config(enabled=False)
        feeds_config = FeedsConfig(feeds=[])

        result = await run_step1(config, feeds_config, temp_cache_for_step1)

        assert result.success is True
        assert result.feeds_fetched == 0
        assert len(result.articles) == 0

    @pytest.mark.asyncio
    async def test_step1_caches_results(
        self, temp_cache_for_step1: CacheManager, sample_rss_content: str
    ) -> None:
        """Test that Step 1 saves results to cache."""
        from src.models.articles import ProcessedArticle

        config = Step1Config(enabled=True)

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

        with aioresponses() as m:
            m.get("https://example.com/feed.rss", status=200, body=sample_rss_content)

            result = await run_step1(config, feeds_config, temp_cache_for_step1)

            assert result.success is True

            # Check cache was populated - use ProcessedArticle, not list
            cached_articles = temp_cache_for_step1.load("articles", ProcessedArticle)
            assert cached_articles is not None
            assert len(cached_articles) > 0
            # Verify articles are ProcessedArticle instances
            assert all(isinstance(a, ProcessedArticle) for a in cached_articles)


class TestStep1IntegrationEdgeCases:
    """Integration tests for Step 1 edge cases."""

    @pytest.mark.asyncio
    async def test_step1_no_enabled_feeds(self, temp_cache_for_step1: CacheManager) -> None:
        """Test with all feeds disabled."""
        config = Step1Config(enabled=True)

        feeds_config = FeedsConfig(
            feeds=[
                FeedConfig(
                    name="Disabled Feed",
                    url="https://example.com/feed.rss",
                    feed_type="specialized",
                    priority=10,
                    enabled=False,  # Disabled
                )
            ]
        )

        result = await run_step1(config, feeds_config, temp_cache_for_step1)

        assert result.success is True
        assert result.feeds_fetched == 0
        assert len(result.articles) == 0

    @pytest.mark.asyncio
    async def test_step1_slug_uniqueness(self, temp_cache_for_step1: CacheManager) -> None:
        """Test that all generated slugs are unique."""
        config = Step1Config(enabled=True)

        today = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
        rss_with_similar_titles = f"""<?xml version="1.0"?>
<rss version="2.0">
    <channel>
        <item>
            <title>AI Model Released</title>
            <link>https://example.com/1</link>
            <pubDate>{today}</pubDate>
        </item>
        <item>
            <title>AI Model Released!</title>
            <link>https://example.com/2</link>
            <pubDate>{today}</pubDate>
        </item>
        <item>
            <title>AI Model Released Today</title>
            <link>https://example.com/3</link>
            <pubDate>{today}</pubDate>
        </item>
    </channel>
</rss>"""

        feeds_config = FeedsConfig(
            feeds=[
                FeedConfig(
                    name="Test",
                    url="https://example.com/feed.rss",
                    feed_type="specialized",
                    priority=10,
                    enabled=True,
                )
            ]
        )

        with aioresponses() as m:
            m.get("https://example.com/feed.rss", status=200, body=rss_with_similar_titles)

            result = await run_step1(config, feeds_config, temp_cache_for_step1)

            assert result.success is True
            assert len(result.articles) == 3

            # Check all slugs are unique
            slugs = [a.slug for a in result.articles]
            assert len(slugs) == len(set(slugs))

    @pytest.mark.asyncio
    async def test_step1_articles_sorted_by_date(
        self, temp_cache_for_step1: CacheManager, sample_rss_content: str
    ) -> None:
        """Test that articles are sorted by published date descending."""
        config = Step1Config(enabled=True)

        feeds_config = FeedsConfig(
            feeds=[
                FeedConfig(
                    name="Test",
                    url="https://example.com/feed.rss",
                    feed_type="specialized",
                    priority=10,
                    enabled=True,
                )
            ]
        )

        with aioresponses() as m:
            m.get("https://example.com/feed.rss", status=200, body=sample_rss_content)

            result = await run_step1(config, feeds_config, temp_cache_for_step1)

            assert len(result.articles) > 1

            # Check dates are in descending order (newest first)
            dates = [a.published_date for a in result.articles if a.published_date]
            if len(dates) > 1:
                for i in range(len(dates) - 1):
                    assert dates[i] >= dates[i + 1]
