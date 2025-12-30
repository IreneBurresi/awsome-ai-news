"""BDD tests for Step 5: Top News Selection and Categorization."""

import asyncio
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from src.models.config import Step5Config
from src.models.news import NewsCategory, NewsCluster, Step5Result
from src.steps.step5_selection import run_step5

# Load scenarios from feature file
scenarios("features/step5_selection.feature")


# Fixtures


@pytest.fixture
def step5_config() -> dict:
    """Step 5 configuration fixture."""
    return {
        "config": Step5Config(
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
    }


@pytest.fixture
def news_clusters() -> dict:
    """News clusters storage."""
    return {"clusters": []}


@pytest.fixture
def api_mock_config() -> dict:
    """API mock configuration."""
    return {"categorized": [], "should_fail": False}


@pytest.fixture
def step5_result() -> dict:
    """Step 5 result storage."""
    return {"result": None}


# Given Steps - Configuration


@given("Step 5 is enabled")
def step5_enabled(step5_config: dict) -> None:
    """Ensure Step 5 is enabled."""
    step5_config["config"].enabled = True


@given(parsers.parse("the target count is {count:d}"))
def set_target_count(step5_config: dict, count: int) -> None:
    """Set target count for top news selection."""
    step5_config["config"].target_count = count


@given(parsers.parse("the minimum quality score is {score:f}"))
def set_min_quality_score(step5_config: dict, score: float) -> None:
    """Set minimum quality score."""
    step5_config["config"].min_quality_score = score


# Given Steps - News Clusters


@given(parsers.parse("we have {count:d} news cluster"))
@given(parsers.parse("we have {count:d} news clusters"))
def create_news_clusters(news_clusters: dict, count: int) -> None:
    """Create news clusters."""
    clusters = []
    for i in range(count):
        cluster = NewsCluster(
            news_id=f"news-{i:03d}",
            title=f"AI News Article Number {i}",
            summary=f"This is an important AI news article about topic {i} with enough detail to pass validation.",
            article_slugs=[f"article-{i}"],
            article_count=1,
            main_topic="test",
            keywords=["ai", "test"],
            created_at=datetime.utcnow(),
        )
        clusters.append(cluster)
    news_clusters["clusters"] = clusters


# Given Steps - API Mocking


@given("the Gemini API categorizes them successfully")
def api_categorizes_successfully(api_mock_config: dict, news_clusters: dict) -> None:
    """Configure API to successfully categorize all news."""
    categorized = []
    categories = [
        "model_release",
        "research",
        "policy_regulation",
        "industry_news",
        "product_launch",
    ]

    for i, cluster in enumerate(news_clusters["clusters"]):
        categorized.append(
            {
                "news_id": cluster.news_id,
                "category": categories[i % len(categories)],
                "importance_score": 9.0 - i * 0.5,
                "reasoning": f"Test categorization for news {i}",
            }
        )

    api_mock_config["categorized"] = categorized


@given(parsers.parse("the Gemini API categorizes them with scores from {high:f} to {low:f}"))
def api_categorizes_with_scores(
    api_mock_config: dict, news_clusters: dict, high: float, low: float
) -> None:
    """Configure API to categorize with specific score range."""
    count = len(news_clusters["clusters"])
    categorized = []

    for i, cluster in enumerate(news_clusters["clusters"]):
        # Linear interpolation from high to low
        score = high - (high - low) * i / (count - 1) if count > 1 else high
        categorized.append(
            {
                "news_id": cluster.news_id,
                "category": "industry_news",
                "importance_score": score,
                "reasoning": f"Scored {score}",
            }
        )

    api_mock_config["categorized"] = categorized


@given(parsers.parse("the Gemini API only categorizes {count:d} of them"))
def api_categorizes_partial(api_mock_config: dict, news_clusters: dict, count: int) -> None:
    """Configure API to only categorize some news."""
    categorized = []

    for i in range(min(count, len(news_clusters["clusters"]))):
        cluster = news_clusters["clusters"][i]
        categorized.append(
            {
                "news_id": cluster.news_id,
                "category": "model_release",
                "importance_score": 8.0,
                "reasoning": "Test",
            }
        )

    api_mock_config["categorized"] = categorized


@given(parsers.parse('{count:d} is categorized as "{category}"'))
@given(parsers.parse('{count:d} are categorized as "{category}"'))
def categorize_as_category(
    api_mock_config: dict, news_clusters: dict, count: int, category: str
) -> None:
    """Categorize specific number of news with given category."""
    # This step accumulates categories
    existing_count = len(api_mock_config["categorized"])

    for i in range(count):
        if existing_count + i < len(news_clusters["clusters"]):
            cluster = news_clusters["clusters"][existing_count + i]
            api_mock_config["categorized"].append(
                {
                    "news_id": cluster.news_id,
                    "category": category,
                    "importance_score": 8.0 - i * 0.5,
                    "reasoning": f"Categorized as {category}",
                }
            )


@given(parsers.parse("one has importance score {score:f}"))
def set_importance_score(api_mock_config: dict, news_clusters: dict, score: float) -> None:
    """Set importance score for next news."""
    idx = len(api_mock_config["categorized"])
    if idx < len(news_clusters["clusters"]):
        cluster = news_clusters["clusters"][idx]
        api_mock_config["categorized"].append(
            {
                "news_id": cluster.news_id,
                "category": "industry_news",
                "importance_score": score,
                "reasoning": f"Test score {score}",
            }
        )


# When Steps


@when("I run Step 5")
def run_step5_action(
    step5_config: dict,
    news_clusters: dict,
    api_mock_config: dict,
    step5_result: dict,
) -> None:
    """Execute Step 5."""
    # Setup API mock
    if api_mock_config.get("should_fail"):
        with patch("google.genai.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.models.generate_content.side_effect = Exception("API failed")

            result = asyncio.run(
                run_step5(
                    step5_config["config"],
                    news_clusters["clusters"],
                    api_key="test-key",
                )
            )
    else:
        # Mock API to return configured categorizations
        mock_response = MagicMock()
        categorized = api_mock_config.get("categorized", [])
        response_dict = {
            "categorized_news": categorized,
            "rationale": "Mock categorization result",
        }
        mock_response.text = json.dumps(response_dict)

        with patch("google.genai.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.models.generate_content.return_value = mock_response

            result = asyncio.run(
                run_step5(
                    step5_config["config"],
                    news_clusters["clusters"],
                    api_key="test-key",
                )
            )

    step5_result["result"] = result


# Then Steps - Success


@then("Step 5 should succeed")
def check_step5_success(step5_result: dict) -> None:
    """Verify Step 5 succeeded."""
    result: Step5Result = step5_result["result"]
    assert result.success is True


# Then Steps - Categorization


@then(parsers.parse("all {count:d} news should be categorized"))
def check_all_categorized(step5_result: dict, count: int) -> None:
    """Verify all news are categorized."""
    result: Step5Result = step5_result["result"]
    assert len(result.all_categorized_news) == count


@then(parsers.parse("only the top {count:d} news should be selected"))
def check_top_selected(step5_result: dict, count: int) -> None:
    """Verify only top N news are selected."""
    result: Step5Result = step5_result["result"]
    assert len(result.top_news) == count


@then("the top news should be sorted by importance score")
def check_sorted_by_score(step5_result: dict) -> None:
    """Verify top news are sorted descending by score."""
    result: Step5Result = step5_result["result"]
    if len(result.top_news) > 1:
        scores = [news.importance_score for news in result.top_news]
        assert scores == sorted(scores, reverse=True)


@then("the top news scores should be descending")
def check_scores_descending(step5_result: dict) -> None:
    """Verify scores are in descending order."""
    result: Step5Result = step5_result["result"]
    scores = [news.importance_score for news in result.top_news]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


# Then Steps - Default Values


@then(parsers.parse('{count:d} news should have default category "{category}"'))
def check_default_category(step5_result: dict, count: int, category: str) -> None:
    """Verify some news have default category."""
    result: Step5Result = step5_result["result"]
    default_cat = NewsCategory(category)
    count_with_default = sum(
        1 for news in result.all_categorized_news if news.category == default_cat
    )
    assert count_with_default >= count


@then(parsers.parse("{count:d} news should have default score {score:f}"))
def check_default_score(step5_result: dict, count: int, score: float) -> None:
    """Verify some news have default score."""
    result: Step5Result = step5_result["result"]
    count_with_default = sum(
        1 for news in result.all_categorized_news if news.importance_score == score
    )
    assert count_with_default >= count


# Then Steps - Category Distribution


@then(parsers.parse("the category distribution should show {count:d} {category}"))
def check_category_distribution(step5_result: dict, count: int, category: str) -> None:
    """Verify category distribution."""
    result: Step5Result = step5_result["result"]
    cat = NewsCategory(category)
    assert result.categories_distribution.get(cat, 0) == count


# Then Steps - Score Clamping


@then(parsers.parse("the first score should be clamped to {score:f}"))
def check_first_score_clamped(step5_result: dict, score: float) -> None:
    """Verify first score is clamped."""
    result: Step5Result = step5_result["result"]
    # Find news with clamped score (was 15.0, clamped to 10.0)
    assert any(news.importance_score == score for news in result.all_categorized_news)


@then(parsers.parse("the second score should be clamped to {score:f}"))
def check_second_score_clamped(step5_result: dict, score: float) -> None:
    """Verify second score is clamped."""
    result: Step5Result = step5_result["result"]
    # Find news with clamped score (was -5.0, clamped to 0.0)
    assert any(news.importance_score == score for news in result.all_categorized_news)


@then(parsers.parse("the third score should remain {score:f}"))
def check_third_score_unchanged(step5_result: dict, score: float) -> None:
    """Verify third score is unchanged."""
    result: Step5Result = step5_result["result"]
    assert any(news.importance_score == score for news in result.all_categorized_news)


# Then Steps - Empty Results


@then("no news should be selected")
def check_no_news_selected(step5_result: dict) -> None:
    """Verify no news were selected."""
    result: Step5Result = step5_result["result"]
    assert len(result.top_news) == 0
    assert len(result.all_categorized_news) == 0


# Then Steps - API Calls


@then(parsers.parse("the API should be called {count:d} time"))
@then(parsers.parse("the API should be called {count:d} times"))
def check_api_calls(step5_result: dict, count: int) -> None:
    """Verify API call count."""
    result: Step5Result = step5_result["result"]
    assert result.api_calls == count


@then("the API should not be called")
def check_no_api_calls(step5_result: dict) -> None:
    """Verify API was not called."""
    result: Step5Result = step5_result["result"]
    assert result.api_calls == 0
