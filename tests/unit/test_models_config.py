"""Unit tests for configuration models."""

import pytest
from pydantic import ValidationError

from src.models.config import (
    FeedConfig,
    FeedFilter,
    LoggingConfig,
    Step1Config,
    Step3Config,
)


class TestFeedFilter:
    """Test FeedFilter model."""

    def test_feed_filter_valid(self) -> None:
        """Test valid feed filter creation."""
        filter_config = FeedFilter(
            whitelist_keywords=["AI", "ML"],
            blacklist_keywords=["crypto", "NFT"],
        )
        assert filter_config.whitelist_keywords == ["AI", "ML"]
        assert filter_config.blacklist_keywords == ["crypto", "NFT"]

    def test_feed_filter_empty(self) -> None:
        """Test feed filter with empty lists."""
        filter_config = FeedFilter()
        assert filter_config.whitelist_keywords == []
        assert filter_config.blacklist_keywords == []


class TestFeedConfig:
    """Test FeedConfig model."""

    def test_feed_config_specialized(self) -> None:
        """Test specialized feed configuration."""
        feed = FeedConfig(
            name="Test Feed",
            url="https://example.com/feed.xml",
            feed_type="specialized",
            priority=10,
        )
        assert feed.name == "Test Feed"
        assert feed.feed_type == "specialized"
        assert feed.priority == 10
        assert feed.enabled is True
        assert feed.filter is None

    def test_feed_config_generalist_with_filter(self) -> None:
        """Test generalist feed with filter."""
        feed = FeedConfig(
            name="Tech News",
            url="https://example.com/feed.xml",
            feed_type="generalist",
            priority=7,
            filter=FeedFilter(whitelist_keywords=["AI"]),
        )
        assert feed.feed_type == "generalist"
        assert feed.filter is not None
        assert feed.filter.whitelist_keywords == ["AI"]

    def test_feed_config_invalid_priority(self) -> None:
        """Test invalid priority values."""
        with pytest.raises(ValidationError):
            FeedConfig(
                name="Test",
                url="https://example.com/feed.xml",
                feed_type="specialized",
                priority=11,  # Out of range
            )

        with pytest.raises(ValidationError):
            FeedConfig(
                name="Test",
                url="https://example.com/feed.xml",
                feed_type="specialized",
                priority=0,  # Out of range
            )

    def test_feed_config_invalid_type(self) -> None:
        """Test invalid feed type."""
        with pytest.raises(ValidationError):
            FeedConfig(
                name="Test",
                url="https://example.com/feed.xml",
                feed_type="invalid",  # type: ignore
                priority=5,
            )


class TestStep1Config:
    """Test Step1Config model."""

    def test_step1_config_defaults(self) -> None:
        """Test Step 1 config with defaults."""
        config = Step1Config()
        assert config.enabled is True
        assert config.timeout_seconds == 30
        assert config.max_articles_per_feed == 50
        assert config.parallel_fetch is True

    def test_step1_config_custom(self) -> None:
        """Test Step 1 config with custom values."""
        config = Step1Config(
            enabled=False,
            timeout_seconds=60,
            max_articles_per_feed=100,
            parallel_fetch=False,
            max_concurrent_feeds=10,
        )
        assert config.enabled is False
        assert config.timeout_seconds == 60
        assert config.max_concurrent_feeds == 10


class TestStep3Config:
    """Test Step3Config model."""

    def test_step3_config_defaults(self) -> None:
        """Test Step 3 config with defaults."""
        config = Step3Config()
        assert config.llm_model == "gemini-2.5-flash-lite"
        assert config.max_clusters == 20
        assert config.temperature == 0.3
        assert config.fallback_to_singleton is True

    def test_step3_config_invalid_temperature(self) -> None:
        """Test invalid temperature values."""
        with pytest.raises(ValidationError):
            Step3Config(temperature=3.0)  # Out of range

        with pytest.raises(ValidationError):
            Step3Config(temperature=-0.1)  # Out of range


class TestLoggingConfig:
    """Test LoggingConfig model."""

    def test_logging_config_defaults(self) -> None:
        """Test logging config with defaults."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.serialize is True
        assert config.colorize is True
        assert config.rotation == "500 MB"
        assert config.retention == "30 days"
        assert config.compression == "zip"

    def test_logging_config_custom(self) -> None:
        """Test logging config with custom values."""
        config = LoggingConfig(
            level="DEBUG",
            serialize=False,
            colorize=False,
            rotation="1 GB",
            retention="60 days",
        )
        assert config.level == "DEBUG"
        assert config.serialize is False
        assert config.colorize is False
