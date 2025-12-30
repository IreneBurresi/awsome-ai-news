"""Unit tests for Step 5: Top News Selection and Categorization."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.models.config import Step5Config
from src.models.news import CategorizedNews, NewsCategory, NewsCluster
from src.steps.step5_selection import (
    CategorizedNewsItem,
    _calculate_category_distribution,
    _get_category_description,
    _parse_categorized_news,
    _prepare_news_for_prompt,
    run_step5,
)


@pytest.fixture
def step5_config() -> Step5Config:
    """Step 5 configuration fixture."""
    return Step5Config(
        enabled=True,
        target_count=10,
        min_quality_score=0.6,
        scoring_weights={
            "recency": 0.3,
            "source_priority": 0.3,
            "content_quality": 0.2,
            "engagement_potential": 0.2,
        },
    )


@pytest.fixture
def sample_news_clusters() -> list[NewsCluster]:
    """Sample news clusters for testing."""
    return [
        NewsCluster(
            news_id="news-001",
            title="GPT-5 Released by OpenAI",
            summary=(
                "OpenAI releases GPT-5, their most advanced language model with "
                "breakthrough capabilities in reasoning and understanding."
            ),
            article_slugs=["gpt5-release"],
            article_count=1,
            main_topic="model release",
            keywords=["GPT-5", "OpenAI", "model"],
            created_at=datetime.utcnow(),
        ),
        NewsCluster(
            news_id="news-002",
            title="New AI Safety Research",
            summary=(
                "Researchers publish groundbreaking paper on AI alignment and safety "
                "mechanisms for large language models."
            ),
            article_slugs=["ai-safety-research"],
            article_count=1,
            main_topic="research",
            keywords=["AI safety", "research", "alignment"],
            created_at=datetime.utcnow(),
        ),
        NewsCluster(
            news_id="news-003",
            title="EU AI Act Passed",
            summary=(
                "European Union passes comprehensive AI regulation framework to govern "
                "development and deployment of AI systems."
            ),
            article_slugs=["eu-ai-act"],
            article_count=1,
            main_topic="policy",
            keywords=["EU", "regulation", "policy"],
            created_at=datetime.utcnow(),
        ),
    ]


def test_get_category_description() -> None:
    """Test category description retrieval."""
    desc = _get_category_description(NewsCategory.MODEL_RELEASE)
    assert "model" in desc.lower()
    assert len(desc) > 0

    desc_other = _get_category_description(NewsCategory.OTHER)
    assert len(desc_other) > 0


def test_prepare_news_for_prompt(sample_news_clusters: list[NewsCluster]) -> None:
    """Test news formatting for prompt."""
    prompt_text = _prepare_news_for_prompt(sample_news_clusters)

    assert len(prompt_text) > 0
    assert "news-001" in prompt_text
    assert "GPT-5" in prompt_text
    assert "news-002" in prompt_text
    assert "AI Safety" in prompt_text


def test_prepare_news_for_prompt_empty() -> None:
    """Test news formatting with empty list."""
    prompt_text = _prepare_news_for_prompt([])
    assert prompt_text == ""


def test_parse_categorized_news(sample_news_clusters: list[NewsCluster]) -> None:
    """Test parsing categorization response."""
    mock_response = MagicMock()
    mock_response.categorized_news = [
        CategorizedNewsItem(
            news_id="news-001",
            category="model_release",
            importance_score=9.5,
            reasoning="Major model release from leading company",
        ),
        CategorizedNewsItem(
            news_id="news-002",
            category="research",
            importance_score=7.0,
            reasoning="Important safety research",
        ),
        CategorizedNewsItem(
            news_id="news-003",
            category="policy_regulation",
            importance_score=8.5,
            reasoning="Significant policy development",
        ),
    ]
    mock_response.rationale = "Categorization complete"

    categorized = _parse_categorized_news(sample_news_clusters, mock_response)

    assert len(categorized) == 3
    assert categorized[0].category == NewsCategory.MODEL_RELEASE
    assert categorized[0].importance_score == 9.5
    assert categorized[1].category == NewsCategory.RESEARCH
    assert categorized[2].category == NewsCategory.POLICY_REGULATION


def test_parse_categorized_news_deduplicates(
    sample_news_clusters: list[NewsCluster],
) -> None:
    """Ensure duplicate news entries from the LLM are ignored."""
    mock_response = MagicMock()
    mock_response.categorized_news = [
        CategorizedNewsItem(
            news_id="news-001",
            category="model_release",
            importance_score=9.5,
            reasoning="Primary classification",
        ),
        CategorizedNewsItem(
            news_id="news-001",
            category="research",
            importance_score=4.0,
            reasoning="Duplicate that should be ignored",
        ),
        CategorizedNewsItem(
            news_id="news-002",
            category="research",
            importance_score=7.5,
            reasoning="Different news item",
        ),
    ]
    mock_response.rationale = "Duplicate test"

    categorized = _parse_categorized_news(sample_news_clusters, mock_response)
    ids = [item.news_cluster.news_id for item in categorized]

    assert ids.count("news-001") == 1, "Duplicate news IDs should be skipped"
    assert set(ids) == {"news-001", "news-002", "news-003"}

    news_one = next(item for item in categorized if item.news_cluster.news_id == "news-001")
    assert news_one.importance_score == 9.5


def test_parse_categorized_news_invalid_category(
    sample_news_clusters: list[NewsCluster],
) -> None:
    """Test parsing with invalid category defaults to OTHER."""
    mock_response = MagicMock()
    mock_response.categorized_news = [
        CategorizedNewsItem(
            news_id="news-001",
            category="invalid_category",
            importance_score=5.0,
            reasoning="Test",
        )
    ]
    mock_response.rationale = "Test"

    categorized = _parse_categorized_news(sample_news_clusters, mock_response)

    assert len(categorized) > 0
    assert categorized[0].category == NewsCategory.OTHER


def test_parse_categorized_news_missing_news(
    sample_news_clusters: list[NewsCluster],
) -> None:
    """Test parsing when some news are not categorized."""
    mock_response = MagicMock()
    mock_response.categorized_news = [
        CategorizedNewsItem(
            news_id="news-001",
            category="model_release",
            importance_score=9.0,
            reasoning="Test",
        )
        # news-002 and news-003 missing
    ]
    mock_response.rationale = "Test"

    categorized = _parse_categorized_news(sample_news_clusters, mock_response)

    # Should still have all 3 news (missing ones get defaults)
    assert len(categorized) == 3

    # Find the default-added news
    default_news = [c for c in categorized if c.news_cluster.news_id in ["news-002", "news-003"]]
    assert len(default_news) == 2
    assert all(c.category == NewsCategory.OTHER for c in default_news)
    assert all(c.importance_score == 5.0 for c in default_news)


def test_parse_categorized_news_invalid_news_id(
    sample_news_clusters: list[NewsCluster],
) -> None:
    """Test parsing with invalid news ID (should skip)."""
    mock_response = MagicMock()
    mock_response.categorized_news = [
        CategorizedNewsItem(
            news_id="news-999",  # Doesn't exist
            category="model_release",
            importance_score=9.0,
            reasoning="Test",
        )
    ]
    mock_response.rationale = "Test"

    categorized = _parse_categorized_news(sample_news_clusters, mock_response)

    # Should add all original news with defaults since none were categorized
    assert len(categorized) == 3
    assert all(c.category == NewsCategory.OTHER for c in categorized)


def test_parse_categorized_news_score_clamping(
    sample_news_clusters: list[NewsCluster],
) -> None:
    """Test that importance scores at boundary values work correctly.

    Note: With Pydantic schema validation (ge=0.0, le=10.0), out-of-range values
    are rejected at model creation time. This test verifies boundary values work.
    """
    mock_response = MagicMock()
    mock_response.categorized_news = [
        CategorizedNewsItem(
            news_id="news-001",
            category="model_release",
            importance_score=10.0,  # Max valid
            reasoning="Test",
        ),
        CategorizedNewsItem(
            news_id="news-002",
            category="research",
            importance_score=0.0,  # Min valid
            reasoning="Test",
        ),
        CategorizedNewsItem(
            news_id="news-003",
            category="policy_regulation",
            importance_score=5.0,  # Mid-range
            reasoning="Test",
        ),
    ]
    mock_response.rationale = "Test"

    categorized = _parse_categorized_news(sample_news_clusters, mock_response)

    assert categorized[0].importance_score == 10.0  # Max preserved
    assert categorized[1].importance_score == 0.0  # Min preserved
    assert categorized[2].importance_score == 5.0  # Mid preserved


def test_calculate_category_distribution() -> None:
    """Test category distribution calculation."""
    news_clusters = [
        NewsCluster(
            news_id=f"news-{i}",
            title=f"News Article Number {i}",
            summary=f"Summary for news {i} with enough characters to pass validation requirements.",
            article_slugs=[f"slug-{i}"],
            article_count=1,
            main_topic="test",
            keywords=["test"],
            created_at=datetime.utcnow(),
        )
        for i in range(5)
    ]

    categorized = [
        CategorizedNews(
            news_cluster=news_clusters[0],
            category=NewsCategory.MODEL_RELEASE,
            importance_score=9.0,
        ),
        CategorizedNews(
            news_cluster=news_clusters[1],
            category=NewsCategory.MODEL_RELEASE,
            importance_score=8.0,
        ),
        CategorizedNews(
            news_cluster=news_clusters[2],
            category=NewsCategory.RESEARCH,
            importance_score=7.0,
        ),
        CategorizedNews(
            news_cluster=news_clusters[3],
            category=NewsCategory.POLICY_REGULATION,
            importance_score=6.0,
        ),
        CategorizedNews(
            news_cluster=news_clusters[4],
            category=NewsCategory.RESEARCH,
            importance_score=5.0,
        ),
    ]

    distribution = _calculate_category_distribution(categorized)

    assert distribution[NewsCategory.MODEL_RELEASE] == 2
    assert distribution[NewsCategory.RESEARCH] == 2
    assert distribution[NewsCategory.POLICY_REGULATION] == 1


@pytest.mark.asyncio
async def test_run_step5_disabled(
    step5_config: Step5Config, sample_news_clusters: list[NewsCluster]
) -> None:
    """Test Step 5 when disabled."""
    step5_config.enabled = False

    result = await run_step5(step5_config, sample_news_clusters, api_key="test-key")

    assert result.success is True
    assert len(result.top_news) == 0
    assert len(result.all_categorized_news) == 0
    assert result.api_calls == 0


@pytest.mark.asyncio
async def test_run_step5_empty_input(step5_config: Step5Config) -> None:
    """Test Step 5 with empty news list."""
    result = await run_step5(step5_config, [], api_key="test-key")

    assert result.success is True
    assert len(result.top_news) == 0
    assert len(result.all_categorized_news) == 0
    assert result.api_calls == 0


@pytest.mark.asyncio
async def test_run_step5_no_api_key(
    step5_config: Step5Config, sample_news_clusters: list[NewsCluster]
) -> None:
    """Test Step 5 without API key fails."""
    result = await run_step5(step5_config, sample_news_clusters, api_key=None)

    assert result.success is False
    assert len(result.top_news) == 0
    assert len(result.errors) > 0
    assert "No API key" in result.errors[0]


@pytest.mark.asyncio
async def test_run_step5_successful_categorization(
    step5_config: Step5Config, sample_news_clusters: list[NewsCluster]
) -> None:
    """Test successful Step 5 execution."""
    mock_response = MagicMock()
    mock_response.text = """{
        "categorized_news": [
            {
                "news_id": "news-001",
                "category": "model_release",
                "importance_score": 9.5,
                "reasoning": "Major model release"
            },
            {
                "news_id": "news-002",
                "category": "research",
                "importance_score": 7.0,
                "reasoning": "Important research"
            },
            {
                "news_id": "news-003",
                "category": "policy_regulation",
                "importance_score": 8.5,
                "reasoning": "Significant policy"
            }
        ],
        "rationale": "Categorization complete"
    }"""

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step5(step5_config, sample_news_clusters, api_key="test-key")

    assert result.success is True
    assert len(result.all_categorized_news) == 3
    assert len(result.top_news) == 3  # All 3 since target is 10
    assert result.api_calls == 1

    # Check sorting by importance score
    assert result.top_news[0].importance_score == 9.5  # Highest
    assert result.top_news[1].importance_score == 8.5  # Second
    assert result.top_news[2].importance_score == 7.0  # Third

    # Check category distribution
    assert result.categories_distribution[NewsCategory.MODEL_RELEASE] == 1
    assert result.categories_distribution[NewsCategory.RESEARCH] == 1
    assert result.categories_distribution[NewsCategory.POLICY_REGULATION] == 1


@pytest.mark.asyncio
async def test_run_step5_selects_top_n(step5_config: Step5Config) -> None:
    """Test that Step 5 selects most interesting news (quality filtered, max N)."""
    # Create 15 news clusters
    news_clusters = [
        NewsCluster(
            news_id=f"news-{i:03d}",
            title=f"News Article Number {i}",
            summary=(
                f"Summary for news article {i} with enough characters to pass "
                "validation requirements."
            ),
            article_slugs=[f"slug-{i}"],
            article_count=1,
            main_topic="test",
            keywords=["test"],
            created_at=datetime.utcnow(),
        )
        for i in range(15)
    ]

    # Mock response with all 15 categorized, scores from 10.0 to 3.0
    categorized_items = [
        {
            "news_id": f"news-{i:03d}",
            "category": "industry_news",
            "importance_score": 10.0 - i * 0.5,  # Descending scores
            "reasoning": f"News {i}",
        }
        for i in range(15)
    ]

    mock_response = MagicMock()
    import json

    mock_response.text = json.dumps(
        {"categorized_news": categorized_items, "rationale": "All categorized"}
    )

    step5_config.target_count = 10

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step5(step5_config, news_clusters, api_key="test-key")

    assert result.success is True
    assert len(result.all_categorized_news) == 15  # All categorized

    # Only 9 news have score >= 6.0 (threshold), so only 9 selected (not 10)
    assert len(result.top_news) == 9
    assert all(news.importance_score >= 6.0 for news in result.top_news)

    # Verify scores are descending
    scores = [news.importance_score for news in result.top_news]
    assert scores == sorted(scores, reverse=True)
    assert min(scores) >= 6.0  # All above quality threshold


@pytest.mark.asyncio
async def test_run_step5_api_failure(
    step5_config: Step5Config, sample_news_clusters: list[NewsCluster]
) -> None:
    """Test Step 5 handles API failures."""
    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.side_effect = Exception("API failed")

        result = await run_step5(step5_config, sample_news_clusters, api_key="test-key")

    assert result.success is False
    assert len(result.top_news) == 0
    assert result.api_failures == 1
    assert len(result.errors) > 0


@pytest.mark.asyncio
async def test_run_step5_critical_error(
    step5_config: Step5Config, sample_news_clusters: list[NewsCluster]
) -> None:
    """Test Step 5 handles critical errors gracefully."""
    # This should trigger the outer exception handler
    with patch(
        "src.steps.step5_selection._call_gemini_categorization",
        side_effect=Exception("Critical error"),
    ):
        result = await run_step5(step5_config, sample_news_clusters, api_key="test-key")

    assert result.success is False
    assert len(result.errors) > 0
