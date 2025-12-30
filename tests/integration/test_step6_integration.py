"""Integration tests for Step 6: Content Enhancement with Web Grounding."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.models.config import Step6Config
from src.models.news import CategorizedNews, NewsCategory, NewsCluster
from src.steps.step6_enhancement import run_step6


@pytest.fixture
def step6_config() -> Step6Config:
    """Standard Step 6 configuration."""
    return Step6Config(
        enabled=True,
        llm_model="gemini-2.5-flash-lite",
        use_grounding=True,
        timeout_seconds=15,
        retry_attempts=3,
        temperature=0.3,
        max_summary_length=300,
    )


def create_sample_categorized_news(
    news_id: str, title: str, summary: str, category: NewsCategory, score: float
) -> CategorizedNews:
    """Helper to create sample categorized news."""
    return CategorizedNews(
        news_cluster=NewsCluster(
            news_id=news_id,
            title=title,
            summary=summary,
            article_slugs=[news_id.lower()],
            article_count=1,
            main_topic="test",
            keywords=["test", "ai"],
            created_at=datetime.utcnow(),
        ),
        category=category,
        importance_score=score,
        reasoning="Test reasoning",
    )


@pytest.mark.asyncio
async def test_step6_enhances_all_news(step6_config: Step6Config) -> None:
    """Test that Step 6 enhances all top news items."""
    top_news = [
        create_sample_categorized_news(
            "news-001",
            "GPT-5 Released by OpenAI",
            "OpenAI releases GPT-5, their most advanced language model with breakthrough capabilities in reasoning and understanding.",
            NewsCategory.MODEL_RELEASE,
            9.5,
        ),
        create_sample_categorized_news(
            "news-002",
            "AI Safety Research Published",
            "Researchers publish groundbreaking paper on AI alignment and safety mechanisms for large language models.",
            NewsCategory.RESEARCH,
            7.0,
        ),
    ]

    # Mock Gemini API response
    mock_response_text = """
---NEWS 1---
TITLE: GPT-5 Released by OpenAI

EXTENDED SUMMARY:
OpenAI has officially released GPT-5, marking a significant milestone in artificial intelligence development. The new model demonstrates unprecedented capabilities in reasoning, understanding complex queries, and generating human-like responses. According to OpenAI's announcement, GPT-5 features improvements in factual accuracy, reduced hallucinations, and better alignment with human values. The model has been trained on a larger and more diverse dataset, incorporating feedback from millions of users worldwide. Industry experts are calling this release a game-changer for AI applications in healthcare, education, and scientific research.

KEY POINTS:
- GPT-5 represents a major advancement in language model capabilities
- Improved factual accuracy and reduced hallucinations
- Better alignment with human values and ethics
- Trained on larger and more diverse dataset

CITATIONS:
- "This is the most capable model we've ever released" - Sam Altman, OpenAI CEO

---END NEWS 1---

---NEWS 2---
TITLE: AI Safety Research Published

EXTENDED SUMMARY:
A team of researchers from leading universities has published a groundbreaking paper on AI alignment and safety mechanisms for large language models. The research introduces novel techniques for ensuring AI systems remain aligned with human intentions and values, even as they scale to unprecedented capabilities. The paper presents empirical evidence that current safety measures may be insufficient for next-generation models and proposes a comprehensive framework for evaluating and improving AI safety.

KEY POINTS:
- Novel techniques for AI alignment introduced
- Current safety measures may be insufficient
- Comprehensive evaluation framework proposed

CITATIONS:
- "Safety must be built into AI systems from the ground up" - Lead Researcher

---END NEWS 2---
"""

    mock_response = MagicMock()
    mock_response.text = mock_response_text

    # Mock grounding metadata
    mock_candidate = MagicMock()
    mock_chunk1 = MagicMock()
    mock_chunk1.web.uri = "https://openai.com/blog/gpt5"
    mock_chunk1.web.title = "GPT-5 Release Announcement"

    mock_chunk2 = MagicMock()
    mock_chunk2.web.uri = "https://arxiv.org/ai-safety-paper"
    mock_chunk2.web.title = "AI Safety Research Paper"

    mock_grounding_metadata = MagicMock()
    mock_grounding_metadata.grounding_chunks = [mock_chunk1, mock_chunk2]
    mock_grounding_metadata.web_search_queries = ["GPT-5", "AI safety"]
    mock_grounding_metadata.grounding_supports = []

    mock_candidate.grounding_metadata = mock_grounding_metadata
    mock_response.candidates = [mock_candidate]

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step6(step6_config, top_news, api_key="test-key")

    # Should succeed
    assert result.success is True
    assert len(result.enhanced_news) == 2
    assert result.api_calls == 1
    assert result.enhancement_failures == 0

    # Verify first enhanced news
    enhanced1 = result.enhanced_news[0]
    assert enhanced1.news.news_cluster.news_id == "news-001"
    assert len(enhanced1.extended_summary) >= 200
    assert "GPT-5" in enhanced1.extended_summary
    assert len(enhanced1.key_points) == 4
    assert len(enhanced1.citations) == 1
    assert enhanced1.grounded is True

    # Verify second enhanced news
    enhanced2 = result.enhanced_news[1]
    assert enhanced2.news.news_cluster.news_id == "news-002"
    assert len(enhanced2.extended_summary) >= 200
    assert "AI alignment" in enhanced2.extended_summary
    assert len(enhanced2.key_points) == 3
    assert len(enhanced2.citations) == 1


@pytest.mark.asyncio
async def test_step6_with_external_links(step6_config: Step6Config) -> None:
    """Test that Step 6 extracts external links from grounding metadata."""
    top_news = [
        create_sample_categorized_news(
            "news-001",
            "AI Breakthrough Announced",
            "Major AI breakthrough announced by research team with significant implications for the field and future applications.",
            NewsCategory.RESEARCH,
            8.0,
        ),
    ]

    mock_response_text = """
---NEWS 1---
TITLE: AI Breakthrough Announced

EXTENDED SUMMARY:
A major AI breakthrough has been announced by a leading research team, with significant implications for the field. The discovery involves a novel approach to neural network architecture that dramatically improves performance on complex reasoning tasks. The research team published their findings in a prestigious journal, and the work has already garnered significant attention from the AI community. Experts predict this breakthrough will accelerate progress in areas such as natural language understanding and scientific discovery.

KEY POINTS:
- Novel neural network architecture developed
- Dramatic performance improvements on reasoning tasks
- Published in prestigious journal
- Expected to accelerate AI progress

---END NEWS 1---
"""

    mock_response = MagicMock()
    mock_response.text = mock_response_text

    # Mock grounding with multiple external links
    mock_candidate = MagicMock()
    mock_chunks = []
    for i in range(5):
        chunk = MagicMock()
        chunk.web.uri = f"https://example{i}.com/article"
        chunk.web.title = f"Research Article {i}"
        mock_chunks.append(chunk)

    mock_grounding_metadata = MagicMock()
    mock_grounding_metadata.grounding_chunks = mock_chunks
    mock_grounding_metadata.web_search_queries = ["AI breakthrough"]
    mock_grounding_metadata.grounding_supports = []

    mock_candidate.grounding_metadata = mock_grounding_metadata
    mock_response.candidates = [mock_candidate]

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step6(step6_config, top_news, api_key="test-key")

    assert result.success is True
    assert result.total_external_links == 5
    assert result.avg_links_per_news == 5.0
    assert len(result.enhanced_news[0].external_links) == 5


@pytest.mark.asyncio
async def test_step6_handles_multiple_news(step6_config: Step6Config) -> None:
    """Test Step 6 with multiple news items (realistic scenario)."""
    # Create 5 news items
    top_news = [
        create_sample_categorized_news(
            f"news-{i:03d}",
            f"AI News Article {i}",
            f"This is AI news article number {i} with important information about developments in the field that require detailed coverage.",
            NewsCategory.INDUSTRY_NEWS,
            10.0 - i,
        )
        for i in range(5)
    ]

    # Mock response with all 5 news sections
    news_sections = []
    for i in range(5):
        section = f"""
---NEWS {i+1}---
TITLE: AI News Article {i}

EXTENDED SUMMARY:
This is an enhanced summary for AI news article {i}. The article discusses important developments in artificial intelligence and machine learning. The content has been expanded with additional context from web sources to provide readers with a comprehensive understanding of the topic. This summary includes recent developments, expert opinions, and potential implications for the industry and society at large.

KEY POINTS:
- Important development in AI field
- Expert opinions included
- Industry implications discussed

CITATIONS:
- "This is significant" - Expert {i}

---END NEWS {i+1}---
"""
        news_sections.append(section)

    mock_response_text = "\n".join(news_sections)

    mock_response = MagicMock()
    mock_response.text = mock_response_text
    mock_response.candidates = []

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step6(step6_config, top_news, api_key="test-key")

    assert result.success is True
    assert len(result.enhanced_news) == 5
    assert result.api_calls == 1
    assert result.enhancement_failures == 0

    # Verify all news were enhanced
    for i, enhanced in enumerate(result.enhanced_news):
        assert enhanced.news.news_cluster.news_id == f"news-{i:03d}"
        assert len(enhanced.extended_summary) >= 200


@pytest.mark.asyncio
async def test_step6_partial_enhancement_failure(step6_config: Step6Config) -> None:
    """Test Step 6 when some news fail enhancement (short summaries)."""
    top_news = [
        create_sample_categorized_news(
            "news-001",
            "Good News Article",
            "This is a good news article with enough content to create a proper summary when enhanced.",
            NewsCategory.RESEARCH,
            9.0,
        ),
        create_sample_categorized_news(
            "news-002",
            "Bad News Article",
            "This is a bad news article that will get a short summary that fails validation.",
            NewsCategory.INDUSTRY_NEWS,
            8.0,
        ),
    ]

    # First news has good summary, second has too short
    mock_response_text = """
---NEWS 1---
TITLE: Good News Article

EXTENDED SUMMARY:
This is a comprehensive enhanced summary for the good news article. It includes detailed information from multiple sources and provides readers with in-depth context about the topic. The summary discusses various aspects of the news, including background information, current developments, and future implications for the field and industry.

KEY POINTS:
- Comprehensive coverage provided
- Multiple sources consulted
- Future implications discussed

---END NEWS 1---

---NEWS 2---
TITLE: Bad News Article

EXTENDED SUMMARY:
Too short summary.

KEY POINTS:
- Brief point

---END NEWS 2---
"""

    mock_response = MagicMock()
    mock_response.text = mock_response_text
    mock_response.candidates = []

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step6(step6_config, top_news, api_key="test-key")

    assert result.success is True
    assert len(result.enhanced_news) == 1  # Only first one succeeded
    assert result.enhancement_failures == 1  # Second one failed


@pytest.mark.asyncio
async def test_step6_no_grounding_metadata(step6_config: Step6Config) -> None:
    """Test Step 6 when API returns no grounding metadata."""
    top_news = [
        create_sample_categorized_news(
            "news-001",
            "Test News Article",
            "Test news article with sufficient content for validation purposes and to ensure proper functioning of the system.",
            NewsCategory.OTHER,
            7.0,
        ),
    ]

    mock_response_text = """
---NEWS 1---
TITLE: Test News Article

EXTENDED SUMMARY:
This is a test enhanced summary without grounding metadata. The summary still provides comprehensive coverage of the topic and includes sufficient detail to meet the minimum length requirements. It discusses the main points and provides context for readers to understand the significance of the news.

KEY POINTS:
- Test point 1
- Test point 2
- Test point 3

---END NEWS 1---
"""

    mock_response = MagicMock()
    mock_response.text = mock_response_text
    mock_response.candidates = []  # No grounding metadata

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step6(step6_config, top_news, api_key="test-key")

    assert result.success is True
    assert len(result.enhanced_news) == 1
    assert result.total_external_links == 0
    assert result.avg_links_per_news == 0.0
    # Should still be marked as grounded=False when no external links
    assert result.enhanced_news[0].grounded is False


@pytest.mark.asyncio
async def test_step6_statistics_calculation(step6_config: Step6Config) -> None:
    """Test that Step 6 correctly calculates all statistics."""
    top_news = [
        create_sample_categorized_news(
            f"news-{i:03d}",
            f"News Article {i}",
            f"News article {i} with comprehensive content covering important developments in the AI field and their implications.",
            NewsCategory.RESEARCH,
            9.0 - i,
        )
        for i in range(3)
    ]

    # Create response with varying numbers of links per news
    mock_response_text = """
---NEWS 1---
TITLE: News Article 0

EXTENDED SUMMARY:
Enhanced summary for news 0 with comprehensive coverage of the topic including background information and current developments. The summary provides readers with detailed context and discusses implications for the field and industry. Additional research and expert opinions have been incorporated.

KEY POINTS:
- Point 1
- Point 2

---END NEWS 1---

---NEWS 2---
TITLE: News Article 1

EXTENDED SUMMARY:
Enhanced summary for news 1 with comprehensive coverage of the topic including background information and current developments. The summary provides readers with detailed context and discusses implications for the field and industry. Additional research and expert opinions have been incorporated.

KEY POINTS:
- Point 1

---END NEWS 2---

---NEWS 3---
TITLE: News Article 2

EXTENDED SUMMARY:
Enhanced summary for news 2 with comprehensive coverage of the topic including background information and current developments. The summary provides readers with detailed context and discusses implications for the field and industry. Additional research and expert opinions have been incorporated.

KEY POINTS:
- Point 1
- Point 2
- Point 3

---END NEWS 3---
"""

    mock_response = MagicMock()
    mock_response.text = mock_response_text

    # Mock grounding with 6 total links
    mock_candidate = MagicMock()
    mock_chunks = []
    for i in range(6):
        chunk = MagicMock()
        chunk.web.uri = f"https://example{i}.com/article"
        chunk.web.title = f"Article {i}"
        mock_chunks.append(chunk)

    mock_grounding_metadata = MagicMock()
    mock_grounding_metadata.grounding_chunks = mock_chunks
    mock_grounding_metadata.web_search_queries = []
    mock_grounding_metadata.grounding_supports = []

    mock_candidate.grounding_metadata = mock_grounding_metadata
    mock_response.candidates = [mock_candidate]

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.models.generate_content.return_value = mock_response

        result = await run_step6(step6_config, top_news, api_key="test-key")

    assert result.success is True
    assert len(result.enhanced_news) == 3
    # Implementation distributes all links to all news (6 links × 3 news = 18 total)
    assert result.total_external_links == 18
    assert result.avg_links_per_news == 6.0  # 18 links / 3 news
    assert result.api_calls == 1
    assert result.api_failures == 0
    assert result.enhancement_failures == 0
