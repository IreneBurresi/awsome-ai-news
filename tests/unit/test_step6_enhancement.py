"""Unit tests for Step 6: Content Enhancement with Web Grounding."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.models.config import Step6Config
from src.models.news import (
    CategorizedNews,
    NewsCategory,
    NewsCluster,
)
from src.steps.step6_enhancement import (
    _parse_single_news_response,
    run_step6,
)


@pytest.fixture
def step6_config() -> Step6Config:
    """Step 6 configuration fixture."""
    return Step6Config(
        enabled=True,
        llm_model="gemini-2.5-flash-lite",
        use_grounding=True,
        timeout_seconds=15,
        retry_attempts=3,
        temperature=0.3,
        max_summary_length=300,
    )


@pytest.fixture
def sample_categorized_news() -> list[CategorizedNews]:
    """Sample categorized news for testing."""
    return [
        CategorizedNews(
            news_cluster=NewsCluster(
                news_id="news-001",
                title="GPT-5 Released by OpenAI",
                summary="OpenAI releases GPT-5, their most advanced language model with breakthrough capabilities in reasoning and understanding.",
                article_slugs=["gpt5-release"],
                article_count=1,
                main_topic="model release",
                keywords=["GPT-5", "OpenAI", "model"],
                created_at=datetime.utcnow(),
            ),
            category=NewsCategory.MODEL_RELEASE,
            importance_score=9.5,
            reasoning="Major model release from leading company",
        ),
        CategorizedNews(
            news_cluster=NewsCluster(
                news_id="news-002",
                title="New AI Safety Research",
                summary="Researchers publish groundbreaking paper on AI alignment and safety mechanisms for large language models.",
                article_slugs=["ai-safety-research"],
                article_count=1,
                main_topic="research",
                keywords=["AI safety", "research", "alignment"],
                created_at=datetime.utcnow(),
            ),
            category=NewsCategory.RESEARCH,
            importance_score=7.0,
            reasoning="Important safety research",
        ),
    ]


# Removed tests for _extract_external_links (async function tested via integration tests)


@pytest.mark.asyncio
async def test_parse_single_news_response_valid(
    sample_categorized_news: list[CategorizedNews],
) -> None:
    """Test parsing valid single news response."""
    response_text = """
=== NEWS START ===
NEWS_ID: news-001
TITLE: GPT-5 Released by OpenAI
ABSTRACT:
OpenAI releases GPT-5 with unprecedented capabilities in reasoning, understanding, and factual accuracy.

EXTENDED SUMMARY:
OpenAI has officially released GPT-5, marking a significant milestone in artificial intelligence development. The new model demonstrates unprecedented capabilities in reasoning, understanding complex queries, and generating human-like responses. According to OpenAI's announcement, GPT-5 features improvements in factual accuracy, reduced hallucinations, and better alignment with human values. The model has been trained on a larger and more diverse dataset, incorporating feedback from millions of users worldwide. Industry experts are calling this release a game-changer for AI applications in healthcare, education, and scientific research.

KEY POINTS:
- GPT-5 represents a major advancement in language model capabilities
- Improved factual accuracy and reduced hallucinations
- Better alignment with human values and ethics

CITATIONS:
- "This is the most capable model we've ever released" - OpenAI
- "GPT-5 shows remarkable improvements in reasoning" - TechCrunch
=== NEWS END ===
"""

    # Mock grounding metadata with external links
    mock_chunk = MagicMock()
    mock_chunk.web.uri = "https://openai.com/blog/gpt5"
    mock_chunk.web.title = "GPT-5 Release"

    grounding_metadata = {
        "grounding_chunks": [mock_chunk],
        "grounding_supports": [],
        "web_search_queries": [],
    }

    enhanced_news = await _parse_single_news_response(
        response_text, grounding_metadata, sample_categorized_news[0]
    )

    # Check enhanced news
    assert enhanced_news.news.news_cluster.news_id == "news-001"
    assert len(enhanced_news.abstract) >= 50
    assert "GPT-5" in enhanced_news.abstract
    assert len(enhanced_news.extended_summary) >= 200
    assert "GPT-5" in enhanced_news.extended_summary
    assert len(enhanced_news.key_points) >= 3
    assert len(enhanced_news.citations) == 2
    assert enhanced_news.grounded is True


# Removed old batch processing tests - no longer applicable with one call per news


@pytest.mark.asyncio
async def test_run_step6_disabled(
    step6_config: Step6Config, sample_categorized_news: list[CategorizedNews]
) -> None:
    """Test Step 6 when disabled."""
    step6_config.enabled = False

    result = await run_step6(step6_config, sample_categorized_news, api_key="test-key")

    assert result.success is True
    assert len(result.enhanced_news) == 0
    assert result.total_external_links == 0
    assert result.avg_links_per_news == 0.0
    assert result.api_calls == 0


@pytest.mark.asyncio
async def test_run_step6_empty_input(step6_config: Step6Config) -> None:
    """Test Step 6 with empty news list."""
    result = await run_step6(step6_config, [], api_key="test-key")

    assert result.success is True
    assert len(result.enhanced_news) == 0
    assert result.total_external_links == 0
    assert result.api_calls == 0


@pytest.mark.asyncio
async def test_run_step6_no_api_key(
    step6_config: Step6Config, sample_categorized_news: list[CategorizedNews]
) -> None:
    """Test Step 6 without API key fails."""
    result = await run_step6(step6_config, sample_categorized_news, api_key=None)

    assert result.success is False
    assert len(result.enhanced_news) == 0
    assert len(result.errors) > 0
    assert "No API key" in result.errors[0]


@pytest.mark.asyncio
async def test_run_step6_successful_enhancement(
    step6_config: Step6Config, sample_categorized_news: list[CategorizedNews]
) -> None:
    """Test successful Step 6 execution with grounding (one call per news)."""
    # Mock response text for single news
    mock_response_text = """
=== NEWS START ===
NEWS_ID: news-001
TITLE: GPT-5 Released by OpenAI
ABSTRACT:
OpenAI releases GPT-5 with unprecedented improvements in reasoning and factual accuracy.

EXTENDED SUMMARY:
OpenAI has officially released GPT-5, marking a significant milestone in artificial intelligence development. The new model demonstrates unprecedented capabilities in reasoning, understanding complex queries, and generating human-like responses. According to OpenAI's announcement, GPT-5 features improvements in factual accuracy, reduced hallucinations, and better alignment with human values. The model has been trained on a larger and more diverse dataset.

KEY POINTS:
- GPT-5 represents a major advancement
- Improved factual accuracy
- Better alignment with human values

CITATIONS:
- "Most capable model ever" - OpenAI
=== NEWS END ===
"""

    # Mock grounding response
    mock_response = MagicMock()
    mock_response.text = mock_response_text

    # Mock grounding metadata
    mock_candidate = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.web.uri = "https://openai.com/blog/gpt5"
    mock_chunk.web.title = "GPT-5 Release"

    mock_grounding_metadata = MagicMock()
    mock_grounding_metadata.grounding_chunks = [mock_chunk]
    mock_grounding_metadata.web_search_queries = ["GPT-5 release"]
    mock_grounding_metadata.grounding_supports = []

    mock_candidate.grounding_metadata = mock_grounding_metadata
    mock_response.candidates = [mock_candidate]

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step6(step6_config, sample_categorized_news, api_key="test-key")

    assert result.success is True
    assert len(result.enhanced_news) == len(sample_categorized_news)
    assert result.total_external_links >= 0
    # One call per news item
    assert result.api_calls == len(sample_categorized_news)
    assert result.enhancement_failures == 0

    # Check enhanced news properties
    assert len(result.enhanced_news[0].abstract) >= 50
    assert len(result.enhanced_news[0].extended_summary) >= 200
    assert len(result.enhanced_news[0].key_points) > 0
    assert result.enhanced_news[0].grounded is True


@pytest.mark.asyncio
async def test_run_step6_api_failure(
    step6_config: Step6Config, sample_categorized_news: list[CategorizedNews]
) -> None:
    """Test Step 6 handles API failures (one call per news)."""
    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.side_effect = Exception("API failed")

        result = await run_step6(step6_config, sample_categorized_news, api_key="test-key")

    assert result.success is False
    assert len(result.enhanced_news) == 0
    # Should fail all news items (one call per news)
    assert result.api_failures == len(sample_categorized_news)
    assert len(result.errors) > 0


@pytest.mark.asyncio
async def test_run_step6_partial_failure(
    step6_config: Step6Config, sample_categorized_news: list[CategorizedNews]
) -> None:
    """Test Step 6 with partial enhancement (some news fail parsing)."""
    # Create a mock that fails for the second news
    mock_response_text_valid = """
=== NEWS START ===
NEWS_ID: news-001
TITLE: GPT-5 Released
ABSTRACT:
OpenAI has released GPT-5 with significant improvements in AI capabilities.

EXTENDED SUMMARY:
OpenAI has released GPT-5 with significant improvements. The model shows better reasoning and reduced hallucinations. This is a major milestone in AI development that will impact various industries. Experts predict widespread adoption soon.

KEY POINTS:
- Major advancement
- Better reasoning
=== NEWS END ===
"""

    mock_response_valid = MagicMock()
    mock_response_valid.text = mock_response_text_valid
    mock_response_valid.candidates = []

    call_count = [0]

    def mock_generate(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return mock_response_valid
        else:
            raise Exception("API failed for second news")

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.side_effect = mock_generate

        result = await run_step6(step6_config, sample_categorized_news, api_key="test-key")

    assert result.success is True  # Partial success
    assert len(result.enhanced_news) == 1  # Only first news succeeded
    assert result.enhancement_failures == 1  # Second news failed
    assert result.api_failures == 1  # One API call failed


@pytest.mark.asyncio
async def test_run_step6_calculates_statistics(
    step6_config: Step6Config, sample_categorized_news: list[CategorizedNews]
) -> None:
    """Test that Step 6 correctly calculates statistics (one call per news)."""
    mock_response_text = """
=== NEWS START ===
NEWS_ID: news-001
TITLE: Test
ABSTRACT:
This is a test abstract with enough words to meet the minimum requirement.

EXTENDED SUMMARY:
This is a test summary with enough words to meet the minimum requirement of 200 characters. This summary discusses various aspects of the test topic and provides detailed information about the subject matter. It includes multiple sentences to ensure adequate length.

KEY POINTS:
- Point 1
- Point 2
=== NEWS END ===
"""

    # Mock with external links
    mock_response = MagicMock()
    mock_response.text = mock_response_text

    mock_candidate = MagicMock()
    mock_chunk1 = MagicMock()
    mock_chunk1.web.uri = "https://example1.com/article"
    mock_chunk1.web.title = "Article 1"

    mock_chunk2 = MagicMock()
    mock_chunk2.web.uri = "https://example2.com/article"
    mock_chunk2.web.title = "Article 2"

    mock_grounding_metadata = MagicMock()
    mock_grounding_metadata.grounding_chunks = [mock_chunk1, mock_chunk2]
    mock_grounding_metadata.web_search_queries = []
    mock_grounding_metadata.grounding_supports = []

    mock_candidate.grounding_metadata = mock_grounding_metadata
    mock_response.candidates = [mock_candidate]

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step6(step6_config, sample_categorized_news, api_key="test-key")

    assert result.success is True
    assert result.total_external_links >= 0
    assert result.avg_links_per_news >= 0.0
    assert result.api_calls == len(sample_categorized_news)  # One call per news
    if result.enhanced_news:
        assert result.avg_links_per_news == result.total_external_links / len(result.enhanced_news)
