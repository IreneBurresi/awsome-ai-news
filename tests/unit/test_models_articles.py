"""Unit tests for article models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models.articles import (
    ArticleCluster,
    ClusteredArticle,
    ProcessedArticle,
    RawArticle,
    SelectedArticle,
)


class TestRawArticle:
    """Test RawArticle model."""

    def test_raw_article_valid(self) -> None:
        """Test valid raw article creation."""
        article = RawArticle(
            title="Test Article",
            url="https://example.com/article",
            published_date=datetime(2024, 1, 1),
            content="Article content",
            author="John Doe",
            feed_name="Test Feed",
            feed_priority=8,
        )
        assert article.title == "Test Article"
        assert article.feed_priority == 8
        assert article.author == "John Doe"

    def test_raw_article_minimal(self) -> None:
        """Test raw article with minimal required fields."""
        article = RawArticle(
            title="Test",
            url="https://example.com",
            feed_name="Test Feed",
            feed_priority=5,
        )
        assert article.title == "Test"
        assert article.published_date is None
        assert article.content is None
        assert article.author is None

    def test_raw_article_invalid_url(self) -> None:
        """Test raw article with invalid URL."""
        with pytest.raises(ValidationError):
            RawArticle(
                title="Test",
                url="not-a-url",
                feed_name="Test Feed",
                feed_priority=5,
            )

    def test_raw_article_invalid_priority(self) -> None:
        """Test raw article with invalid priority."""
        with pytest.raises(ValidationError):
            RawArticle(
                title="Test",
                url="https://example.com",
                feed_name="Test Feed",
                feed_priority=11,  # Out of range
            )


class TestProcessedArticle:
    """Test ProcessedArticle model."""

    def test_processed_article_valid(self) -> None:
        """Test valid processed article creation."""
        article = ProcessedArticle(
            title="Test Article",
            url="https://example.com/article",
            published_date=datetime(2024, 1, 1),
            content="Article content",
            author="John Doe",
            feed_name="Test Feed",
            feed_priority=8,
            slug="test-article",
            content_hash="abc123",
        )
        assert article.slug == "test-article"
        assert article.content_hash == "abc123"


class TestClusteredArticle:
    """Test ClusteredArticle model."""

    def test_clustered_article_valid(self) -> None:
        """Test valid clustered article creation."""
        article = ClusteredArticle(
            title="Test Article",
            url="https://example.com/article",
            feed_name="Test Feed",
            feed_priority=8,
            slug="test-article",
            content_hash="abc123",
            cluster_id=1,
        )
        assert article.cluster_id == 1


class TestArticleCluster:
    """Test ArticleCluster model."""

    def test_article_cluster_valid(self) -> None:
        """Test valid article cluster creation."""
        article1 = ClusteredArticle(
            title="Article 1",
            url="https://example.com/1",
            feed_name="Feed",
            feed_priority=8,
            slug="article-1",
            content_hash="hash1",
            cluster_id=1,
        )
        article2 = ClusteredArticle(
            title="Article 2",
            url="https://example.com/2",
            feed_name="Feed",
            feed_priority=7,
            slug="article-2",
            content_hash="hash2",
            cluster_id=1,
        )

        cluster = ArticleCluster(
            cluster_id=1,
            topic="AI Testing",
            articles=[article1, article2],
            representative_article=article1,
        )

        assert cluster.cluster_id == 1
        assert cluster.topic == "AI Testing"
        assert len(cluster.articles) == 2
        assert cluster.representative_article == article1

    def test_article_cluster_without_representative(self) -> None:
        """Test article cluster without representative article."""
        article = ClusteredArticle(
            title="Article",
            url="https://example.com",
            feed_name="Feed",
            feed_priority=8,
            slug="article",
            content_hash="hash",
            cluster_id=1,
        )

        cluster = ArticleCluster(
            cluster_id=1,
            topic="Topic",
            articles=[article],
        )

        assert cluster.representative_article is None


class TestSelectedArticle:
    """Test SelectedArticle model."""

    def test_selected_article_valid(self) -> None:
        """Test valid selected article creation."""
        article = SelectedArticle(
            title="Selected Article",
            url="https://example.com/article",
            feed_name="Test Feed",
            feed_priority=9,
            slug="selected-article",
            content_hash="abc123",
            cluster_id=1,
            cluster_topic="AI News",
            quality_score=0.85,
            score_breakdown={
                "recency": 0.9,
                "source_priority": 0.9,
                "content_quality": 0.8,
                "engagement_potential": 0.8,
            },
        )

        assert article.quality_score == 0.85
        assert article.cluster_topic == "AI News"
        assert len(article.score_breakdown) == 4

    def test_selected_article_invalid_quality_score(self) -> None:
        """Test selected article with invalid quality score."""
        with pytest.raises(ValidationError):
            SelectedArticle(
                title="Article",
                url="https://example.com",
                feed_name="Feed",
                feed_priority=8,
                slug="article",
                content_hash="hash",
                cluster_id=1,
                cluster_topic="Topic",
                quality_score=1.5,  # Out of range
            )
