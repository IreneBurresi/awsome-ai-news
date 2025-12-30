"""Pydantic data models for the pipeline."""

from src.models.articles import (
    ArticleCluster,
    ClusteredArticle,
    ProcessedArticle,
    RawArticle,
    SelectedArticle,
    Step1Result,
)
from src.models.config import (
    ErrorHandlingConfig,
    FeedConfig,
    FeedFilter,
    FeedsConfig,
    LoggingConfig,
    PipelineConfig,
    PipelineMetadata,
    Step0Config,
    Step1Config,
    Step2Config,
    Step3Config,
    Step4Config,
    Step5Config,
    Step6Config,
    Step7Config,
    Step8Config,
    StepConfig,
)
from src.models.news import (
    CategorizedNews,
    Citation,
    EnhancedNews,
    ExternalLink,
    NewsCategory,
    NewsCluster,
    NewsCollection,
    NewsDeduplicationPair,
    NewsMetadata,
    Step3Result,
    Step4Result,
    Step5Result,
    Step6Result,
)
from src.models.repository import (
    CommitInfo,
    Step7Result,
)
from src.models.rss import (
    RSSFeed,
    RSSItem,
    Step8Result,
)

__all__ = [
    # Articles
    "RawArticle",
    "ProcessedArticle",
    "ClusteredArticle",
    "ArticleCluster",
    "SelectedArticle",
    "Step1Result",
    # News
    "NewsCluster",
    "Step3Result",
    "NewsDeduplicationPair",
    "Step4Result",
    "NewsCategory",
    "CategorizedNews",
    "Step5Result",
    "ExternalLink",
    "Citation",
    "EnhancedNews",
    "Step6Result",
    "NewsMetadata",
    "NewsCollection",
    # Repository
    "CommitInfo",
    "Step7Result",
    # RSS
    "RSSItem",
    "RSSFeed",
    "Step8Result",
    # Config
    "FeedFilter",
    "FeedConfig",
    "FeedsConfig",
    "StepConfig",
    "Step0Config",
    "Step1Config",
    "Step2Config",
    "Step3Config",
    "Step4Config",
    "Step5Config",
    "Step6Config",
    "Step7Config",
    "Step8Config",
    "PipelineMetadata",
    "PipelineConfig",
    "LoggingConfig",
    "ErrorHandlingConfig",
]
