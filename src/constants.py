"""Application-wide constants.

Contains configuration constants used across the codebase to avoid magic numbers
and maintain consistency.
"""

# Slug Generation
SLUG_WORD_COUNT = 4  # Number of words from title to include in slug
SLUG_HASH_LENGTH = 8  # Length of hash suffix in slug

# Content Validation
ABSTRACT_MIN_LENGTH = 50  # Minimum length for abstract text
ABSTRACT_MAX_LENGTH = 300  # Maximum length for abstract text
SUMMARY_MIN_LENGTH = 100  # Minimum length for summary text

# Cache Retention
DEFAULT_ARTICLES_RETENTION_DAYS = 10  # Days to keep articles in cache
DEFAULT_NEWS_RETENTION_DAYS = 3  # Days to keep news in cache

# Repository Management
README_OLD_NEWS_CUTOFF_DAYS = 30  # Days before news sections are archived

# Git Operations
GIT_OPERATION_TIMEOUT_SECONDS = 30  # Timeout for git commands

# LLM Configuration
DEFAULT_LLM_TEMPERATURE = 0.3  # Default temperature for LLM calls
DEFAULT_LLM_MAX_RETRIES = 3  # Default max retries for LLM calls
DEFAULT_RETRY_MIN_WAIT = 2  # Minimum wait time between retries (seconds)
DEFAULT_RETRY_MAX_WAIT = 10  # Maximum wait time between retries (seconds)

# Clustering
DEFAULT_MAX_CLUSTERS = 100  # Maximum number of clusters to create
DEFAULT_SIMILARITY_THRESHOLD = 0.8  # Similarity threshold for deduplication

# Selection
DEFAULT_TARGET_NEWS_COUNT = 10  # Target number of top news to select
DEFAULT_MIN_QUALITY_SCORE = 6.0  # Minimum quality score for news selection

# Enhancement
ENHANCEMENT_TIMEOUT_PER_NEWS = 15  # Seconds per news item for enhancement

# Feed Processing
MAX_ERROR_DISPLAY = 5  # Maximum number of errors to display in logs
