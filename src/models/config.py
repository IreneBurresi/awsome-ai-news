"""Configuration models for the pipeline."""

from typing import Literal

from pydantic import BaseModel, Field


class FeedFilter(BaseModel):
    """Filter configuration for generalist feeds."""

    whitelist_keywords: list[str] = Field(default_factory=list)
    blacklist_keywords: list[str] = Field(default_factory=list)
    whitelist_categories: list[str] = Field(
        default_factory=list, description="RSS category tags to whitelist"
    )
    whitelist_regex: str | None = Field(
        default=None, description="Regex pattern for whitelist matching"
    )
    blacklist_regex: str | None = Field(
        default=None, description="Regex pattern for blacklist matching"
    )
    apply_to_fields: list[str] = Field(
        default_factory=lambda: ["title", "description"],
        description="Fields to apply filters to (title, description, content)",
    )


class FeedConfig(BaseModel):
    """Configuration for a single RSS feed."""

    name: str = Field(description="Feed name")
    url: str = Field(description="Feed URL")
    feed_type: Literal["specialized", "generalist"] = Field(description="Feed type")
    enabled: bool = Field(default=True)
    priority: int = Field(ge=1, le=10, description="Feed priority (1-10)")
    filter: FeedFilter | None = Field(
        default=None, description="Optional filter for generalist feeds"
    )


class FeedsConfig(BaseModel):
    """Configuration for all RSS feeds."""

    feeds: list[FeedConfig] = Field(description="List of RSS feeds")


class StepConfig(BaseModel):
    """Base configuration for a pipeline step."""

    enabled: bool = Field(default=True)


class Step0Config(StepConfig):
    """Step 0: Cache management configuration."""

    retention: dict[str, int] = Field(description="Retention periods in days")
    backup_on_error: bool = Field(default=True)
    cleanup_on_start: bool = Field(default=True)


class Step1Config(StepConfig):
    """Step 1: RSS ingestion configuration."""

    timeout_seconds: int = Field(default=30)
    max_articles_per_feed: int = Field(default=50)
    user_agent: str = Field(default="awesome-ai-news/1.0")
    parallel_fetch: bool = Field(default=True)
    max_concurrent_feeds: int = Field(default=5)


class Step2Config(StepConfig):
    """Step 2: Single-feed deduplication configuration.

    Uses exact slug matching for O(1) deduplication performance.
    Lookback window: 10 days (hardcoded in step2_dedup.py).
    """

    pass  # Only uses 'enabled' field from StepConfig base class


class Step3Config(StepConfig):
    """Step 3: AI clustering configuration."""

    llm_model: str = Field(default="gemini-2.5-flash-lite")
    max_clusters: int = Field(default=20)
    min_cluster_size: int = Field(default=1)
    timeout_seconds: int = Field(default=30)
    retry_attempts: int = Field(default=3)
    retry_delay_seconds: int = Field(default=2)
    fallback_to_singleton: bool = Field(default=True)
    temperature: float = Field(ge=0.0, le=2.0, default=0.3)


class Step4Config(StepConfig):
    """Step 4: Multi-day news deduplication configuration.

    Uses Gemini API for semantic comparison of news clusters across 3 days.
    """

    llm_model: str = Field(default="gemini-2.5-flash-lite")
    lookback_days: int = Field(default=3, ge=1, le=7)
    similarity_threshold: float = Field(ge=0.0, le=1.0, default=0.85)
    timeout_seconds: int = Field(default=30)
    retry_attempts: int = Field(default=3)
    temperature: float = Field(ge=0.0, le=2.0, default=0.3)
    fallback_to_no_merge: bool = Field(
        default=True, description="If API fails, don't merge (keep all news)"
    )


class Step5Config(StepConfig):
    """Step 5: Selection configuration."""

    target_count: int = Field(default=10)
    min_quality_score: float = Field(ge=0.0, le=1.0, default=0.6)
    scoring_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "recency": 0.3,
            "source_priority": 0.3,
            "content_quality": 0.2,
            "engagement_potential": 0.2,
        }
    )


class Step6Config(StepConfig):
    """Step 6: Enhancement configuration."""

    llm_model: str = Field(default="gemini-2.5-flash-lite")
    use_grounding: bool = Field(default=True)
    timeout_seconds: int = Field(default=15)
    retry_attempts: int = Field(default=3)
    temperature: float = Field(ge=0.0, le=2.0, default=0.5)
    max_summary_length: int = Field(default=300)


class Step7Config(StepConfig):
    """Step 7: Repository update configuration."""

    output_file: str = Field(default="README.md")
    archive_enabled: bool = Field(default=True)
    archive_dir: str = Field(default="archive")
    commit_message_template: str = Field(default="Update AI news - {date}")
    git_push: bool = Field(default=True)


class Step8Config(StepConfig):
    """Step 8: RSS generation configuration."""

    output_file: str = Field(default="feed.xml")
    feed_title: str = Field(default="Awesome AI News")
    feed_description: str = Field(default="Curated AI news aggregated and enhanced by AI")
    feed_link: str = Field(default="https://github.com/yourusername/awesome-ai-news")
    max_items: int = Field(default=50)


class LoggingConfig(BaseModel):
    """Logging configuration for loguru."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    serialize: bool = Field(default=True, description="Serialize logs to JSON")
    colorize: bool = Field(default=True, description="Colorize console output")
    file_path: str = Field(default="logs/pipeline.log")
    rotation: str = Field(default="500 MB", description="Log rotation size/time")
    retention: str = Field(default="30 days", description="Log retention period")
    compression: str = Field(default="zip", description="Compression format for rotated logs")


class ErrorHandlingConfig(BaseModel):
    """Error handling configuration."""

    stop_on_critical: bool = Field(default=True)
    continue_on_recoverable: bool = Field(default=True)
    max_consecutive_failures: int = Field(default=3)
    notification_on_failure: bool = Field(default=False)


class PipelineMetadata(BaseModel):
    """Pipeline metadata."""

    name: str = Field(default="awesome-ai-news")
    version: str = Field(default="1.0.0")
    execution_mode: Literal["production", "development", "dry_run"] = Field(default="production")


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    pipeline: PipelineMetadata
    step0_cache: Step0Config
    step1_ingestion: Step1Config
    step2_dedup: Step2Config
    step3_clustering: Step3Config
    step4_multi_dedup: Step4Config
    step5_selection: Step5Config
    step6_enhancement: Step6Config
    step7_repo: Step7Config
    step8_rss: Step8Config
    logging: LoggingConfig
    error_handling: ErrorHandlingConfig
