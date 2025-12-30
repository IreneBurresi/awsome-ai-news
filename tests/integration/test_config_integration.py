"""Integration tests for configuration loading."""

from pathlib import Path

import pytest
import yaml

from src.models.config import FeedsConfig, PipelineConfig
from src.utils.config_loader import load_feeds_config, load_pipeline_config


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Create temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def sample_feeds_yaml(temp_config_dir: Path) -> Path:
    """Create sample feeds.yaml file."""
    feeds_config = {
        "feeds": [
            {
                "name": "Test Specialized Feed",
                "url": "https://example.com/feed.xml",
                "feed_type": "specialized",
                "enabled": True,
                "priority": 10,
            },
            {
                "name": "Test Generalist Feed",
                "url": "https://example.com/general.xml",
                "feed_type": "generalist",
                "enabled": True,
                "priority": 7,
                "filter": {
                    "whitelist_keywords": ["AI", "ML"],
                    "blacklist_keywords": ["crypto"],
                },
            },
        ]
    }

    feeds_path = temp_config_dir / "feeds.yaml"
    feeds_path.write_text(yaml.dump(feeds_config))
    return feeds_path


@pytest.fixture
def sample_pipeline_yaml(temp_config_dir: Path) -> Path:
    """Create sample pipeline.yaml file."""
    pipeline_config = {
        "pipeline": {
            "name": "awesome-ai-news",
            "version": "1.0.0",
            "execution_mode": "production",
        },
        "step0_cache": {
            "enabled": True,
            "retention": {"articles_days": 10, "news_days": 3},
            "backup_on_error": True,
            "cleanup_on_start": True,
        },
        "step1_ingestion": {
            "enabled": True,
            "timeout_seconds": 30,
            "max_articles_per_feed": 50,
            "user_agent": "test-agent",
            "parallel_fetch": True,
            "max_concurrent_feeds": 5,
        },
        "step2_dedup": {
            "enabled": True,
        },
        "step3_clustering": {
            "enabled": True,
            "llm_model": "gemini-2.5-flash-lite",
            "max_clusters": 20,
            "min_cluster_size": 1,
            "timeout_seconds": 30,
            "retry_attempts": 3,
            "retry_delay_seconds": 2,
            "fallback_to_singleton": True,
            "temperature": 0.3,
        },
        "step4_multi_dedup": {
            "enabled": True,
            "strategy": "keep_highest_priority",
            "similarity_threshold": 0.9,
        },
        "step5_selection": {
            "enabled": True,
            "target_count": 10,
            "min_quality_score": 0.6,
            "scoring_weights": {
                "recency": 0.3,
                "source_priority": 0.3,
                "content_quality": 0.2,
                "engagement_potential": 0.2,
            },
        },
        "step6_enhancement": {
            "enabled": True,
            "llm_model": "gemini-2.5-flash-lite",
            "use_grounding": True,
            "timeout_seconds": 15,
            "retry_attempts": 3,
            "temperature": 0.5,
            "max_summary_length": 300,
        },
        "step7_repo": {
            "enabled": True,
            "output_file": "README.md",
            "archive_enabled": True,
            "archive_dir": "archive",
            "commit_message_template": "Update AI news - {date}",
            "git_push": True,
        },
        "step8_rss": {
            "enabled": True,
            "output_file": "feed.xml",
            "feed_title": "Test Feed",
            "feed_description": "Test Description",
            "feed_link": "https://example.com",
            "max_items": 50,
        },
        "logging": {
            "level": "INFO",
            "serialize": True,
            "colorize": True,
            "file_path": "logs/test.log",
            "rotation": "500 MB",
            "retention": "30 days",
            "compression": "zip",
        },
        "error_handling": {
            "stop_on_critical": True,
            "continue_on_recoverable": True,
            "max_consecutive_failures": 3,
            "notification_on_failure": False,
        },
    }

    pipeline_path = temp_config_dir / "pipeline.yaml"
    pipeline_path.write_text(yaml.dump(pipeline_config))
    return pipeline_path


class TestConfigIntegration:
    """Integration tests for configuration loading."""

    def test_load_feeds_config(self, sample_feeds_yaml: Path) -> None:
        """Test loading and validating feeds configuration."""
        config = load_feeds_config(sample_feeds_yaml)

        assert isinstance(config, FeedsConfig)
        assert len(config.feeds) == 2

        # Check specialized feed
        specialized = config.feeds[0]
        assert specialized.name == "Test Specialized Feed"
        assert specialized.feed_type == "specialized"
        assert specialized.priority == 10
        assert specialized.filter is None

        # Check generalist feed
        generalist = config.feeds[1]
        assert generalist.name == "Test Generalist Feed"
        assert generalist.feed_type == "generalist"
        assert generalist.priority == 7
        assert generalist.filter is not None
        assert "AI" in generalist.filter.whitelist_keywords
        assert "crypto" in generalist.filter.blacklist_keywords

    def test_load_pipeline_config(self, sample_pipeline_yaml: Path) -> None:
        """Test loading and validating pipeline configuration."""
        config = load_pipeline_config(sample_pipeline_yaml)

        assert isinstance(config, PipelineConfig)

        # Check pipeline metadata
        assert config.pipeline.name == "awesome-ai-news"
        assert config.pipeline.version == "1.0.0"
        assert config.pipeline.execution_mode == "production"

        # Check Step 0 config
        assert config.step0_cache.enabled is True
        assert config.step0_cache.retention["articles_days"] == 10
        assert config.step0_cache.retention["news_days"] == 3

        # Check Step 1 config
        assert config.step1_ingestion.timeout_seconds == 30
        assert config.step1_ingestion.parallel_fetch is True

        # Check Step 3 config
        assert config.step3_clustering.llm_model == "gemini-2.5-flash-lite"
        assert config.step3_clustering.temperature == 0.3

        # Check logging config
        assert config.logging.level == "INFO"
        assert config.logging.serialize is True

        # Check error handling
        assert config.error_handling.stop_on_critical is True

    def test_load_invalid_feeds_config(self, temp_config_dir: Path) -> None:
        """Test loading invalid feeds configuration."""
        invalid_config = {
            "feeds": [
                {
                    "name": "Invalid Feed",
                    "url": "https://example.com",
                    "feed_type": "specialized",
                    "priority": 15,  # Invalid priority (must be 1-10)
                }
            ]
        }

        config_path = temp_config_dir / "invalid_feeds.yaml"
        config_path.write_text(yaml.dump(invalid_config))

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            load_feeds_config(config_path)

    def test_load_nonexistent_config(self, temp_config_dir: Path) -> None:
        """Test loading non-existent configuration file."""
        nonexistent = temp_config_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            load_feeds_config(nonexistent)

    def test_config_with_all_steps_disabled(self, temp_config_dir: Path) -> None:
        """Test configuration with all steps disabled."""
        config = {
            "pipeline": {"name": "test", "version": "1.0.0", "execution_mode": "dry_run"},
            "step0_cache": {
                "enabled": False,
                "retention": {},
                "backup_on_error": False,
                "cleanup_on_start": False,
            },
            "step1_ingestion": {"enabled": False},
            "step2_dedup": {"enabled": False},
            "step3_clustering": {"enabled": False},
            "step4_multi_dedup": {"enabled": False},
            "step5_selection": {"enabled": False},
            "step6_enhancement": {"enabled": False},
            "step7_repo": {"enabled": False},
            "step8_rss": {"enabled": False},
            "logging": {"level": "DEBUG"},
            "error_handling": {},
        }

        config_path = temp_config_dir / "disabled.yaml"
        config_path.write_text(yaml.dump(config))

        loaded = load_pipeline_config(config_path)
        assert loaded.step0_cache.enabled is False
        assert loaded.step1_ingestion.enabled is False
        assert loaded.pipeline.execution_mode == "dry_run"
