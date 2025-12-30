"""Integration tests for Step 5: Top News Selection and Categorization."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.models.config import Step5Config
from src.models.news import NewsCategory, NewsCluster
from src.steps.step5_selection import run_step5


@pytest.fixture
def step5_config() -> Step5Config:
    """Standard Step 5 configuration."""
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


def create_sample_news(news_id: str, title: str, summary: str) -> NewsCluster:
    """Helper to create sample news cluster."""
    return NewsCluster(
        news_id=news_id,
        title=title,
        summary=summary,
        article_slugs=[news_id.lower()],
        article_count=1,
        main_topic="test",
        keywords=["test", "ai"],
        created_at=datetime.utcnow(),
    )


@pytest.mark.asyncio
async def test_step5_categorizes_all_news(step5_config: Step5Config) -> None:
    """Test that Step 5 categorizes all news clusters."""
    news_clusters = [
        create_sample_news(
            "news-001",
            "GPT-5 Released by OpenAI",
            "OpenAI releases GPT-5, their most advanced language model with breakthrough capabilities.",
        ),
        create_sample_news(
            "news-002",
            "AI Safety Research Published",
            "Researchers publish groundbreaking paper on AI alignment and safety mechanisms.",
        ),
        create_sample_news(
            "news-003",
            "EU AI Regulation Passed",
            "European Union passes comprehensive AI regulation framework to govern AI systems.",
        ),
    ]

    mock_response = MagicMock()
    mock_response.text = """{
        "categorized_news": [
            {
                "news_id": "news-001",
                "category": "model_release",
                "importance_score": 9.5,
                "reasoning": "Major model release from leading company"
            },
            {
                "news_id": "news-002",
                "category": "research",
                "importance_score": 7.0,
                "reasoning": "Important safety research"
            },
            {
                "news_id": "news-003",
                "category": "policy_regulation",
                "importance_score": 8.5,
                "reasoning": "Significant policy development"
            }
        ],
        "rationale": "Categorization based on content type and impact"
    }"""

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step5(step5_config, news_clusters, api_key="test-key")

    # Should succeed
    assert result.success is True
    assert len(result.all_categorized_news) == 3
    assert len(result.top_news) == 3  # All 3 since target is 10
    assert result.api_calls == 1

    # Verify categories
    assert result.all_categorized_news[0].category == NewsCategory.MODEL_RELEASE
    assert result.all_categorized_news[1].category == NewsCategory.RESEARCH
    assert result.all_categorized_news[2].category == NewsCategory.POLICY_REGULATION

    # Verify sorting by score
    assert result.top_news[0].importance_score == 9.5
    assert result.top_news[1].importance_score == 8.5
    assert result.top_news[2].importance_score == 7.0


@pytest.mark.asyncio
async def test_step5_selects_top_10_from_many(step5_config: Step5Config) -> None:
    """Test that Step 5 correctly selects top 10 from larger set."""
    # Create 20 news clusters
    news_clusters = [
        create_sample_news(
            f"news-{i:03d}",
            f"AI News Article Number {i}",
            f"This is an important AI news article about topic {i} with enough detail to pass validation.",
        )
        for i in range(20)
    ]

    # Mock response with descending importance scores
    import json

    categorized_items = [
        {
            "news_id": f"news-{i:03d}",
            "category": "industry_news",
            "importance_score": 10.0 - i * 0.3,
            "reasoning": f"News item {i}",
        }
        for i in range(20)
    ]

    mock_response = MagicMock()
    mock_response.text = json.dumps(
        {"categorized_news": categorized_items, "rationale": "Scored by importance"}
    )

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step5(step5_config, news_clusters, api_key="test-key")

    # Should succeed
    assert result.success is True
    assert len(result.all_categorized_news) == 20  # All categorized
    assert len(result.top_news) == 10  # Only top 10 selected

    # Verify top 10 are the highest scored
    top_scores = [news.importance_score for news in result.top_news]
    assert len(top_scores) == 10
    assert all(top_scores[i] >= top_scores[i + 1] for i in range(9))  # Descending


@pytest.mark.asyncio
async def test_step5_handles_partial_categorization(step5_config: Step5Config) -> None:
    """Test that Step 5 handles when API doesn't categorize all news."""
    news_clusters = [
        create_sample_news(
            "news-001",
            "First AI News Article",
            "This is the first news article about AI with enough detail to pass validation.",
        ),
        create_sample_news(
            "news-002",
            "Second AI News Article",
            "This is the second news article about AI with enough detail to pass validation.",
        ),
        create_sample_news(
            "news-003",
            "Third AI News Article",
            "This is the third news article about AI with enough detail to pass validation.",
        ),
    ]

    # Mock response only categorizes 2 out of 3
    mock_response = MagicMock()
    mock_response.text = """{
        "categorized_news": [
            {
                "news_id": "news-001",
                "category": "model_release",
                "importance_score": 8.0,
                "reasoning": "Test"
            },
            {
                "news_id": "news-002",
                "category": "research",
                "importance_score": 7.0,
                "reasoning": "Test"
            }
        ],
        "rationale": "Partial categorization"
    }"""

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step5(step5_config, news_clusters, api_key="test-key")

    # Should still succeed and include all 3 news
    assert result.success is True
    assert len(result.all_categorized_news) == 3

    # The missing one should have default category
    news_003 = next(
        (n for n in result.all_categorized_news if n.news_cluster.news_id == "news-003"),
        None,
    )
    assert news_003 is not None
    assert news_003.category == NewsCategory.OTHER
    assert news_003.importance_score == 5.0


@pytest.mark.asyncio
async def test_step5_category_distribution(step5_config: Step5Config) -> None:
    """Test category distribution calculation."""
    news_clusters = [
        create_sample_news(f"news-{i:03d}", f"News Article {i}", f"Summary for news {i}" * 10)
        for i in range(6)
    ]

    mock_response = MagicMock()
    import json

    mock_response.text = json.dumps(
        {
            "categorized_news": [
                {"news_id": "news-000", "category": "model_release", "importance_score": 9.0, "reasoning": "Test"},
                {"news_id": "news-001", "category": "model_release", "importance_score": 8.0, "reasoning": "Test"},
                {"news_id": "news-002", "category": "research", "importance_score": 7.5, "reasoning": "Test"},
                {"news_id": "news-003", "category": "research", "importance_score": 7.0, "reasoning": "Test"},
                {"news_id": "news-004", "category": "policy_regulation", "importance_score": 6.5, "reasoning": "Test"},
                {"news_id": "news-005", "category": "industry_news", "importance_score": 6.0, "reasoning": "Test"},
            ],
            "rationale": "Test",
        }
    )

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step5(step5_config, news_clusters, api_key="test-key")

    # Check category distribution
    assert result.success is True
    assert result.categories_distribution[NewsCategory.MODEL_RELEASE] == 2
    assert result.categories_distribution[NewsCategory.RESEARCH] == 2
    assert result.categories_distribution[NewsCategory.POLICY_REGULATION] == 1
    assert result.categories_distribution[NewsCategory.INDUSTRY_NEWS] == 1


@pytest.mark.asyncio
async def test_step5_invalid_category_defaults_to_other(step5_config: Step5Config) -> None:
    """Test that invalid categories default to OTHER."""
    news_clusters = [
        create_sample_news(
            "news-001",
            "Test News Article",
            "This is a test news article with enough detail to pass validation requirements.",
        )
    ]

    mock_response = MagicMock()
    mock_response.text = """{
        "categorized_news": [
            {
                "news_id": "news-001",
                "category": "invalid_category_name",
                "importance_score": 7.0,
                "reasoning": "Test"
            }
        ],
        "rationale": "Test"
    }"""

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step5(step5_config, news_clusters, api_key="test-key")

    assert result.success is True
    assert result.all_categorized_news[0].category == NewsCategory.OTHER


@pytest.mark.asyncio
async def test_step5_score_clamping(step5_config: Step5Config) -> None:
    """Test that importance scores outside 0-10 range are clamped."""
    news_clusters = [
        create_sample_news(f"news-{i:03d}", f"News Article {i}", f"Summary for news {i}" * 10)
        for i in range(3)
    ]

    mock_response = MagicMock()
    import json

    mock_response.text = json.dumps(
        {
            "categorized_news": [
                {"news_id": "news-000", "category": "model_release", "importance_score": 15.0, "reasoning": "Test"},  # Too high
                {"news_id": "news-001", "category": "research", "importance_score": -5.0, "reasoning": "Test"},  # Too low
                {"news_id": "news-002", "category": "industry_news", "importance_score": 7.0, "reasoning": "Test"},  # Valid
            ],
            "rationale": "Test",
        }
    )

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step5(step5_config, news_clusters, api_key="test-key")

    assert result.success is True
    # Verify clamping
    assert result.all_categorized_news[0].importance_score == 10.0  # Clamped to max
    assert result.all_categorized_news[1].importance_score == 0.0  # Clamped to min
    assert result.all_categorized_news[2].importance_score == 7.0  # Unchanged
