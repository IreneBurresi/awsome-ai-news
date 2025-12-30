"""Unit tests for Step 3: News Clustering."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models.articles import ProcessedArticle
from src.models.news import NewsCluster, Step3Result
from src.steps.step3_clustering import (
    _create_singleton_clusters,
    _format_articles_for_prompt,
    _generate_news_id,
    _prepare_articles_for_prompt,
)


class TestNewsIdGeneration:
    """Test news ID generation logic."""

    def test_generate_news_id_deterministic(self) -> None:
        """Test that news ID generation is deterministic."""
        title = "AI Model Released"
        slugs = ["article-1-abc", "article-2-def"]

        id1 = _generate_news_id(title, slugs)
        id2 = _generate_news_id(title, slugs)

        assert id1 == id2

    def test_generate_news_id_format(self) -> None:
        """Test news ID format."""
        title = "Breaking News"
        slugs = ["breaking-news-123"]

        news_id = _generate_news_id(title, slugs)

        assert news_id.startswith("news-")
        assert len(news_id) == 17  # "news-" + 12 hex chars

    def test_generate_news_id_different_titles(self) -> None:
        """Test different titles produce different IDs."""
        slugs = ["article-1"]

        id1 = _generate_news_id("Title 1", slugs)
        id2 = _generate_news_id("Title 2", slugs)

        assert id1 != id2

    def test_generate_news_id_different_slugs(self) -> None:
        """Test different slugs produce different IDs."""
        title = "Same Title"

        id1 = _generate_news_id(title, ["slug-1"])
        id2 = _generate_news_id(title, ["slug-2"])

        assert id1 != id2

    def test_generate_news_id_sorted_slugs(self) -> None:
        """Test that slug order doesn't matter."""
        title = "News"

        id1 = _generate_news_id(title, ["b-slug", "a-slug", "c-slug"])
        id2 = _generate_news_id(title, ["a-slug", "c-slug", "b-slug"])

        assert id1 == id2


class TestArticlePreparation:
    """Test article preparation for LLM prompt."""

    def test_prepare_articles_basic(self) -> None:
        """Test basic article preparation."""
        articles = [
            ProcessedArticle(
                title="AI Breakthrough",
                url="https://example.com/article",
                published_date=datetime.now(),
                content="Full content here...",
                author="Author",
                feed_name="AI News",
                feed_priority=8,
                slug="ai-breakthrough-abc123",
                content_hash="hash123",
            )
        ]

        result = _prepare_articles_for_prompt(articles)

        assert len(result) == 1
        assert result[0]["slug"] == "ai-breakthrough-abc123"
        assert result[0]["title"] == "AI Breakthrough"
        assert result[0]["url"] == "https://example.com/article"
        assert result[0]["feed"] == "AI News"

    def test_prepare_articles_content_preview(self) -> None:
        """Test content is truncated to 200 chars."""
        long_content = "a" * 500
        articles = [
            ProcessedArticle(
                title="Article",
                url="https://example.com/article",
                published_date=datetime.now(),
                content=long_content,
                author="Author",
                feed_name="Feed",
                feed_priority=5,
                slug="article-slug",
                content_hash="hash",
            )
        ]

        result = _prepare_articles_for_prompt(articles)

        assert len(result[0]["content_preview"]) == 200

    def test_prepare_articles_no_content(self) -> None:
        """Test handling of articles without content."""
        articles = [
            ProcessedArticle(
                title="No Content Article",
                url="https://example.com/article",
                published_date=datetime.now(),
                content=None,
                author="Author",
                feed_name="Feed",
                feed_priority=5,
                slug="no-content-slug",
                content_hash="hash",
            )
        ]

        result = _prepare_articles_for_prompt(articles)

        assert result[0]["content_preview"] == ""

    def test_format_articles_for_prompt(self) -> None:
        """Test formatting articles for inclusion in prompt."""
        articles_data = [
            {
                "slug": "article-1",
                "title": "First Article",
                "url": "https://example.com/1",
                "content_preview": "Preview 1",
                "feed": "Feed 1",
                "published": "2025-12-24T10:00:00",
            },
            {
                "slug": "article-2",
                "title": "Second Article",
                "url": "https://example.com/2",
                "content_preview": "Preview 2",
                "feed": "Feed 2",
                "published": "2025-12-24T11:00:00",
            },
        ]

        result = _format_articles_for_prompt(articles_data)

        assert "1. [article-1] First Article" in result
        assert "2. [article-2] Second Article" in result
        assert "Preview 1" in result
        assert "Preview 2" in result


class TestSingletonClusters:
    """Test singleton cluster fallback logic."""

    def test_create_singleton_basic(self) -> None:
        """Test basic singleton cluster creation."""
        article = ProcessedArticle(
            title="Standalone Article",
            url="https://example.com/article",
            published_date=datetime.now(),
            content="Article content here, more than 50 characters for summary validation.",
            author="Author",
            feed_name="Feed",
            feed_priority=5,
            slug="standalone-article-abc",
            content_hash="hash",
        )

        clusters = _create_singleton_clusters([article])

        assert len(clusters) == 1
        assert clusters[0].title == "Standalone Article"
        assert clusters[0].article_count == 1
        assert clusters[0].article_slugs == ["standalone-article-abc"]
        assert clusters[0].main_topic == "singleton"

    def test_create_singleton_short_title(self) -> None:
        """Test singleton with title < 10 chars."""
        article = ProcessedArticle(
            title="Short",  # 5 chars
            url="https://example.com/article",
            published_date=datetime.now(),
            content="Content here, more than 50 characters for summary validation.",
            author="Author",
            feed_name="Feed",
            feed_priority=5,
            slug="short-slug",
            content_hash="hash",
        )

        clusters = _create_singleton_clusters([article])

        assert len(clusters[0].title) >= 10
        assert "News: Short" in clusters[0].title

    def test_create_singleton_short_content(self) -> None:
        """Test singleton with content < 50 chars."""
        article = ProcessedArticle(
            title="Article With Short Content",
            url="https://example.com/article",
            published_date=datetime.now(),
            content="Short",  # < 50 chars
            author="Author",
            feed_name="Feed Name",
            feed_priority=5,
            slug="article-slug",
            content_hash="hash",
        )

        clusters = _create_singleton_clusters([article])

        # Should generate summary from title + feed name
        assert len(clusters[0].summary) >= 50
        assert "Article With Short Content" in clusters[0].summary
        assert "Feed Name" in clusters[0].summary

    def test_create_singleton_multiple_articles(self) -> None:
        """Test creating multiple singleton clusters."""
        articles = [
            ProcessedArticle(
                title=f"Article {i}",
                url=f"https://example.com/article-{i}",
                published_date=datetime.now(),
                content=f"Content for article {i}, with enough characters to pass validation.",
                author="Author",
                feed_name="Feed",
                feed_priority=5,
                slug=f"article-{i}-slug",
                content_hash=f"hash_{i}",
            )
            for i in range(5)
        ]

        clusters = _create_singleton_clusters(articles)

        assert len(clusters) == 5
        for i, cluster in enumerate(clusters):
            assert cluster.article_count == 1
            assert cluster.article_slugs == [f"article-{i}-slug"]


class TestNewsClusterValidation:
    """Test NewsCluster Pydantic model validation."""

    def test_valid_news_cluster(self) -> None:
        """Test creating a valid news cluster."""
        cluster = NewsCluster(
            news_id="news-abc123def456",
            title="AI Model Released by OpenAI",
            summary="OpenAI has released a new AI model with enhanced capabilities. This is a summary of the news.",
            article_slugs=["article-1-abc", "article-2-def"],
            article_count=2,
            main_topic="model release",
            keywords=["AI", "OpenAI", "model", "release"],
            created_at=datetime.now(),
        )

        assert cluster.news_id == "news-abc123def456"
        assert cluster.article_count == 2
        assert len(cluster.article_slugs) == 2

    def test_title_too_short(self) -> None:
        """Test title minimum length validation."""
        with pytest.raises(ValidationError):
            NewsCluster(
                news_id="news-123",
                title="Short",  # < 10 chars
                summary="Valid summary with more than 50 characters to pass validation.",
                article_slugs=["slug-1"],
                article_count=1,
                main_topic="topic",
                keywords=["key"],
                created_at=datetime.now(),
            )

    def test_summary_too_short(self) -> None:
        """Test summary minimum length validation."""
        with pytest.raises(ValidationError):
            NewsCluster(
                news_id="news-123",
                title="Valid Title Here",
                summary="Short",  # < 50 chars
                article_slugs=["slug-1"],
                article_count=1,
                main_topic="topic",
                keywords=["key"],
                created_at=datetime.now(),
            )

    def test_article_count_mismatch(self) -> None:
        """Test article_count must match article_slugs length."""
        with pytest.raises(ValidationError):
            NewsCluster(
                news_id="news-123",
                title="Valid Title",
                summary="Valid summary with enough characters to pass validation.",
                article_slugs=["slug-1", "slug-2"],  # 2 slugs
                article_count=3,  # But count says 3
                main_topic="topic",
                keywords=["key"],
                created_at=datetime.now(),
            )

    def test_empty_article_slugs(self) -> None:
        """Test article_slugs cannot be empty."""
        # This should fail because article_count must be >= 1
        # and count must match slugs length
        with pytest.raises(ValidationError):
            NewsCluster(
                news_id="news-123",
                title="Valid Title",
                summary="Valid summary with enough characters to pass validation.",
                article_slugs=[],  # Empty list
                article_count=0,
                main_topic="topic",
                keywords=["key"],
                created_at=datetime.now(),
            )


class TestStep3Result:
    """Test Step3Result model."""

    def test_successful_result(self) -> None:
        """Test creating a successful result."""
        clusters = [
            NewsCluster(
                news_id="news-1",
                title="Cluster Title 1",
                summary="Summary for cluster 1 with enough characters to pass validation.",
                article_slugs=["slug-1", "slug-2"],
                article_count=2,
                main_topic="topic1",
                keywords=["key1"],
                created_at=datetime.now(),
            )
        ]

        result = Step3Result(
            success=True,
            news_clusters=clusters,
            total_clusters=1,
            singleton_clusters=0,
            multi_article_clusters=1,
            articles_clustered=2,
            api_calls=1,
            api_failures=0,
        )

        assert result.success is True
        assert len(result.news_clusters) == 1
        assert result.total_clusters == 1
        assert result.articles_clustered == 2

    def test_failed_result_with_errors(self) -> None:
        """Test creating a failed result with errors."""
        result = Step3Result(
            success=False,
            news_clusters=[],
            total_clusters=0,
            singleton_clusters=0,
            multi_article_clusters=0,
            articles_clustered=0,
            api_calls=1,
            api_failures=1,
            errors=["API call failed: timeout"],
        )

        assert result.success is False
        assert len(result.errors) == 1
        assert result.api_failures == 1

    def test_fallback_result(self) -> None:
        """Test result with fallback used."""
        result = Step3Result(
            success=True,
            news_clusters=[],
            total_clusters=10,
            singleton_clusters=10,
            multi_article_clusters=0,
            articles_clustered=10,
            api_calls=1,
            api_failures=1,
            errors=["API failed, using fallback"],
            fallback_used=True,
        )

        assert result.success is True
        assert result.fallback_used is True
        assert result.singleton_clusters == 10
