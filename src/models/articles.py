"""Article data models for the pipeline."""

from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class RawArticle(BaseModel):
    """Raw article from RSS feed (Step 1)."""

    title: str = Field(description="Article title")
    url: HttpUrl = Field(description="Article URL")
    published_date: datetime | None = Field(default=None, description="Publication date")
    content: str | None = Field(default=None, description="Article content/summary")
    author: str | None = Field(default=None, description="Article author")
    feed_name: str = Field(description="Source feed name")
    feed_priority: int = Field(ge=1, le=10, description="Source feed priority")


class ProcessedArticle(BaseModel):
    """Processed article after deduplication (Step 2)."""

    title: str = Field(description="Article title")
    url: HttpUrl = Field(description="Article URL")
    published_date: datetime | None = Field(default=None, description="Publication date")
    content: str | None = Field(default=None, description="Article content/summary")
    author: str | None = Field(default=None, description="Article author")
    feed_name: str = Field(description="Source feed name")
    feed_priority: int = Field(ge=1, le=10, description="Source feed priority")
    slug: str = Field(description="URL slug for the article")
    content_hash: str = Field(description="Hash for deduplication")


class ClusteredArticle(BaseModel):
    """Article assigned to a cluster (Step 3)."""

    title: str = Field(description="Article title")
    url: HttpUrl = Field(description="Article URL")
    published_date: datetime | None = Field(default=None, description="Publication date")
    content: str | None = Field(default=None, description="Article content/summary")
    author: str | None = Field(default=None, description="Article author")
    feed_name: str = Field(description="Source feed name")
    feed_priority: int = Field(ge=1, le=10, description="Source feed priority")
    slug: str = Field(description="URL slug for the article")
    content_hash: str = Field(description="Hash for deduplication")
    cluster_id: int = Field(description="Assigned cluster ID")


class ArticleCluster(BaseModel):
    """A cluster of related articles (Step 3)."""

    cluster_id: int = Field(description="Cluster ID")
    topic: str = Field(description="Main topic of the cluster")
    articles: list[ClusteredArticle] = Field(description="Articles in this cluster")
    representative_article: ClusteredArticle | None = Field(
        default=None, description="Most representative article in cluster"
    )


class SelectedArticle(BaseModel):
    """Article selected for final output (Step 5)."""

    title: str = Field(description="Article title")
    url: HttpUrl = Field(description="Article URL")
    published_date: datetime | None = Field(default=None, description="Publication date")
    content: str | None = Field(default=None, description="Article content/summary")
    author: str | None = Field(default=None, description="Article author")
    feed_name: str = Field(description="Source feed name")
    feed_priority: int = Field(ge=1, le=10, description="Source feed priority")
    slug: str = Field(description="URL slug for the article")
    content_hash: str = Field(description="Hash for deduplication")
    cluster_id: int = Field(description="Assigned cluster ID")
    cluster_topic: str = Field(description="Cluster topic")
    quality_score: float = Field(ge=0.0, le=1.0, description="Quality score")
    score_breakdown: dict[str, float] = Field(
        default_factory=dict, description="Breakdown of quality score components"
    )


class Step1Result(BaseModel):
    """Result from Step 1 execution."""

    success: bool = Field(description="Whether step completed successfully")
    articles: list[ProcessedArticle] = Field(
        default_factory=list, description="Processed articles with slugs"
    )
    feeds_fetched: int = Field(ge=0, description="Number of feeds successfully fetched")
    feeds_failed: int = Field(ge=0, description="Number of feeds that failed")
    total_articles_raw: int = Field(ge=0, description="Total articles before filtering")
    articles_after_filter: int = Field(ge=0, description="Articles after filtering")
    slug_collisions: int = Field(default=0, ge=0, description="Number of slug collisions detected")
    errors: list[str] = Field(default_factory=list, description="Error messages if any")


class DeduplicationStats(BaseModel):
    """Statistics from Step 2 deduplication."""

    input_articles: int = Field(ge=0, description="Articles from Step 1")
    cache_articles: int = Field(ge=0, description="Articles loaded from cache")
    duplicates_found: int = Field(ge=0, description="Duplicate articles detected")
    unique_articles: int = Field(ge=0, description="Unique articles after dedup")
    deduplication_rate: float = Field(ge=0.0, le=1.0, description="Percentage of duplicates")
    cache_files_loaded: int = Field(ge=0, description="Cache files successfully loaded")
    cache_files_corrupted: int = Field(ge=0, description="Cache files that were corrupted")


class Step2Result(BaseModel):
    """Result from Step 2 execution."""

    success: bool = Field(description="Whether step completed successfully")
    unique_articles: list[ProcessedArticle] = Field(
        default_factory=list, description="Deduplicated articles"
    )
    stats: DeduplicationStats = Field(description="Deduplication statistics")
    cache_updated: bool = Field(description="Whether cache was updated with new articles")
    errors: list[str] = Field(default_factory=list, description="Error messages if any")
