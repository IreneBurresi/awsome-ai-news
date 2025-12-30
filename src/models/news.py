"""Enhanced news data models for final output."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl, field_validator


class NewsCluster(BaseModel):
    """Cluster of articles grouped by topic (Step 3).

    Can be updated/merged in Step 4 multi-day deduplication.
    """

    news_id: str = Field(description="Unique news ID")
    title: str = Field(min_length=10, description="News title (10-150 chars)")
    summary: str = Field(min_length=50, description="News summary (50-500 chars)")
    article_slugs: list[str] = Field(description="List of article slugs (min 1)")
    article_count: int = Field(ge=1, description="Number of articles in cluster")
    main_topic: str = Field(description="Main topic (e.g., 'model release', 'research')")
    keywords: list[str] = Field(description="Extracted keywords (max 10)")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime | None = Field(
        default=None, description="Last update timestamp (Step 4 merge)"
    )

    @field_validator("article_count")
    @classmethod
    def validate_count_matches_slugs(cls, v: int, info) -> int:
        """Validate that article_count matches article_slugs length."""
        if "article_slugs" in info.data and v != len(info.data["article_slugs"]):
            raise ValueError("article_count must match article_slugs length")
        return v


class Step3Result(BaseModel):
    """Result from Step 3: Clustering."""

    success: bool = Field(description="Whether step completed successfully")
    news_clusters: list[NewsCluster] = Field(
        default_factory=list, description="News clusters generated"
    )
    total_clusters: int = Field(ge=0, description="Total number of clusters")
    singleton_clusters: int = Field(ge=0, description="Clusters with 1 article")
    multi_article_clusters: int = Field(ge=0, description="Clusters with 2+ articles")
    articles_clustered: int = Field(ge=0, description="Total articles clustered")
    api_calls: int = Field(ge=0, description="Gemini API calls made")
    api_failures: int = Field(default=0, ge=0, description="Failed API calls")
    errors: list[str] = Field(default_factory=list, description="Error messages if any")
    fallback_used: bool = Field(default=False, description="Whether fallback was used")


class NewsDeduplicationPair(BaseModel):
    """Pair of news clusters identified as duplicates (Step 4).

    Only includes pairs that should be merged (LLM decided they're the same story).
    """

    news_today_id: str = Field(description="News ID from today's clustering")
    news_cached_id: str = Field(description="News ID from cache (last 3 days)")
    merge_reason: str = Field(description="Explanation why they're the same story (max 150 chars)")


class Step4Result(BaseModel):
    """Result from Step 4: Multi-day news deduplication."""

    success: bool = Field(description="Whether step completed successfully")
    unique_news: list[NewsCluster] = Field(
        default_factory=list, description="Deduplicated news clusters"
    )
    news_before_dedup: int = Field(ge=0, description="News count before deduplication")
    news_after_dedup: int = Field(ge=0, description="News count after deduplication")
    duplicates_found: int = Field(ge=0, description="Number of duplicate pairs found")
    news_merged: int = Field(ge=0, description="Number of news actually merged")
    api_calls: int = Field(ge=0, description="Gemini API calls made")
    api_failures: int = Field(default=0, ge=0, description="Failed API calls")
    errors: list[str] = Field(default_factory=list, description="Error messages if any")
    fallback_used: bool = Field(default=False, description="Whether fallback (no merge) was used")


class NewsCategory(str, Enum):
    """Standard categories for AI news classification."""

    MODEL_RELEASE = "model_release"
    RESEARCH = "research"
    POLICY_REGULATION = "policy_regulation"
    FUNDING_ACQUISITION = "funding_acquisition"
    PRODUCT_LAUNCH = "product_launch"
    PARTNERSHIP = "partnership"
    ETHICS_SAFETY = "ethics_safety"
    INDUSTRY_NEWS = "industry_news"
    OTHER = "other"


class CategorizedNews(BaseModel):
    """News cluster with category and importance score (Step 5)."""

    news_cluster: NewsCluster = Field(description="Original news cluster from Step 4")
    category: NewsCategory = Field(description="Assigned news category")
    importance_score: float = Field(ge=0.0, le=10.0, description="Importance score (0-10)")
    reasoning: str | None = Field(
        default=None, max_length=300, description="Brief explanation for score/category"
    )


class Step5Result(BaseModel):
    """Result from Step 5: Top news selection and categorization."""

    success: bool = Field(description="Whether step completed successfully")
    top_news: list[CategorizedNews] = Field(
        default_factory=list, max_length=10, description="Top 10 selected news"
    )
    all_categorized_news: list[CategorizedNews] = Field(
        default_factory=list, description="All news with categories and scores"
    )
    categories_distribution: dict[NewsCategory, int] = Field(
        default_factory=dict, description="Count of news per category"
    )
    api_calls: int = Field(ge=0, description="Gemini API calls made")
    api_failures: int = Field(default=0, ge=0, description="Failed API calls")
    errors: list[str] = Field(default_factory=list, description="Error messages if any")


class Citation(BaseModel):
    """Citation quote from a source (nested in ExternalLink)."""

    text: str = Field(min_length=10, max_length=500, description="Citation text")
    author: str | None = Field(default=None, description="Citation author")
    source: str | None = Field(default=None, description="Citation source/outlet")
    url: HttpUrl | None = Field(default=None, description="Citation source URL")


class ExternalLink(BaseModel):
    """External link for news enhancement with nested citations."""

    url: HttpUrl = Field(description="External link URL")
    title: str = Field(min_length=5, max_length=200, description="Link title")
    source: str = Field(description="Source domain (e.g., techcrunch.com)")
    citations: list[Citation] = Field(
        default_factory=list, description="Citations from this source"
    )
    relevance_score: float = Field(ge=0.0, le=1.0, default=1.0, description="Relevance score")
    snippet: str | None = Field(
        default=None, max_length=300, description="Text snippet from source"
    )


class EnhancedNews(BaseModel):
    """Enhanced news with additional content from web grounding (Step 6)."""

    news: CategorizedNews = Field(description="Original categorized news from Step 5")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Flattened list of citations for quick access",
    )
    abstract: str = Field(
        min_length=50,
        max_length=300,
        description="Brief abstract/summary (50-300 chars)",
    )
    extended_summary: str = Field(
        min_length=200,
        max_length=4000,
        description="Extended summary with grounding (200-4000 chars)",
    )
    external_links: list[ExternalLink] = Field(
        default_factory=list,
        min_length=0,
        max_length=10,
        description="External authoritative links with nested citations",
    )
    key_points: list[str] = Field(default_factory=list, max_length=7, description="Key takeaways")
    enhanced_at: datetime = Field(
        default_factory=datetime.utcnow, description="Enhancement timestamp"
    )
    grounded: bool = Field(default=False, description="Whether grounding was used")


class Step6Result(BaseModel):
    """Result from Step 6: Content enhancement with grounding."""

    success: bool = Field(description="Whether step completed successfully")
    enhanced_news: list[EnhancedNews] = Field(
        default_factory=list, max_length=10, description="Enhanced news items"
    )
    total_external_links: int = Field(ge=0, description="Total external links found")
    avg_links_per_news: float = Field(ge=0.0, description="Average links per news item")
    enhancement_failures: int = Field(default=0, ge=0, description="Number of enhancement failures")
    api_calls: int = Field(ge=0, description="Gemini API calls made")
    api_failures: int = Field(default=0, ge=0, description="Failed API calls")
    errors: list[str] = Field(default_factory=list, description="Error messages if any")


class NewsMetadata(BaseModel):
    """Metadata for the news collection."""

    generated_at: datetime = Field(description="Generation timestamp")
    total_feeds_processed: int = Field(description="Number of feeds processed")
    total_articles_ingested: int = Field(description="Total articles ingested")
    total_articles_after_dedup: int = Field(description="Articles after deduplication")
    total_clusters: int = Field(description="Number of clusters created")
    total_news_selected: int = Field(description="Number of news items selected")
    pipeline_version: str = Field(description="Pipeline version")
    llm_model: str = Field(description="LLM model used")


class NewsCollection(BaseModel):
    """Complete collection of enhanced news with metadata."""

    news: list[EnhancedNews] = Field(description="List of enhanced news items")
    metadata: NewsMetadata = Field(description="Collection metadata")
