"""BDD tests for Step 3: News Clustering."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from src.models.articles import ProcessedArticle
from src.models.config import Step3Config
from src.models.news import NewsCluster, Step3Result
from src.steps.step3_clustering import GeminiClusteringResponse, run_step3

# Load all scenarios from feature file
scenarios("features/step3_clustering.feature")


# Fixtures


@pytest.fixture
def step3_config() -> dict:
    """Shared Step 3 configuration storage."""
    return {
        "config": Step3Config(
            enabled=True,
            llm_model="gemini-2.5-flash-lite",
            temperature=0.3,
            max_clusters=20,
            min_cluster_size=1,
            fallback_to_singleton=True,
        )
    }


@pytest.fixture
def input_articles() -> dict:
    """Storage for input articles from Step 2."""
    return {"articles": []}


@pytest.fixture
def step3_result() -> dict:
    """Shared result storage."""
    return {"result": None, "execution_time": None, "mock_call": None}


@pytest.fixture
def api_mock_config() -> dict:
    """Configuration for API mocking."""
    return {"fail_count": 0, "should_fail": False, "response": None}


# Background Steps


@given("the Step 3 configuration is enabled")
def step3_enabled(step3_config: dict) -> None:
    """Ensure Step 3 is enabled."""
    step3_config["config"].enabled = True


@given("the Gemini API is available")
def api_available() -> None:
    """Gemini API is available (mocked)."""
    pass


# Given Steps - Configuration


@given("the Step 3 configuration is disabled")
def step3_disabled(step3_config: dict) -> None:
    """Disable Step 3."""
    step3_config["config"].enabled = False


@given("fallback to singleton is enabled")
def fallback_enabled(step3_config: dict) -> None:
    """Enable fallback to singleton clusters."""
    step3_config["config"].fallback_to_singleton = True


@given("fallback to singleton is disabled")
def fallback_disabled(step3_config: dict) -> None:
    """Disable fallback to singleton clusters."""
    step3_config["config"].fallback_to_singleton = False


@given("no API key is configured")
def no_api_key(api_mock_config: dict) -> None:
    """No API key is configured."""
    api_mock_config["no_api_key"] = True


# Given Steps - Articles


@given(parsers.parse("I have {count:d} deduplicated article"))
@given(parsers.parse("I have {count:d} deduplicated articles"))
def have_articles(input_articles: dict, count: int) -> None:
    """Create deduplicated articles."""
    articles = [
        ProcessedArticle(
            title=f"Article {i}: AI News Topic",
            url=f"https://example.com/article-{i}",
            published_date=datetime.now(),
            content=f"Content for article {i}. This discusses AI developments and trends in the industry.",
            author=f"Author {i}",
            feed_name=f"Feed {i % 3}",
            feed_priority=5 + (i % 5),
            slug=f"article-{i}-ai-news-topic-hash{i:04d}",
            content_hash=f"hash_{i}",
        )
        for i in range(count)
    ]
    input_articles["articles"] = articles


@given("I have these deduplicated articles:")
def have_specific_articles(datatable, input_articles: dict) -> None:
    """Create specific deduplicated articles from table."""
    headers = datatable[0] if datatable else []
    title_idx = headers.index("title") if "title" in headers else 0
    slug_idx = headers.index("slug") if "slug" in headers else 1
    content_idx = headers.index("content") if "content" in headers else -1

    articles = []
    for row in datatable[1:]:
        title = row[title_idx]
        slug = row[slug_idx]
        content = (
            row[content_idx]
            if content_idx >= 0 and len(row) > content_idx
            else f"Content for {title}. This article discusses AI developments and industry trends."
        )

        article = ProcessedArticle(
            title=title,
            url=f"https://example.com/{slug}",
            published_date=datetime.now(),
            content=content,
            author="Test Author",
            feed_name="Test Feed",
            feed_priority=8,
            slug=slug,
            content_hash=f"hash_{slug}",
        )
        articles.append(article)

    input_articles["articles"] = articles


# Given Steps - API Mocking


@given("the Gemini API will fail")
def api_will_fail_simple(api_mock_config: dict) -> None:
    """Configure API to fail (simple version)."""
    api_mock_config["should_fail"] = True
    api_mock_config["fail_count"] = 999  # Always fail


@given("the Gemini API will fail with timeout")
def api_will_fail_timeout(api_mock_config: dict) -> None:
    """Configure API to fail with timeout."""
    api_mock_config["should_fail"] = True
    api_mock_config["fail_count"] = 999  # Always fail


@given(parsers.parse("the Gemini API will fail {count:d} times then succeed"))
def api_will_fail_then_succeed(api_mock_config: dict, count: int) -> None:
    """Configure API to fail N times then succeed."""
    api_mock_config["should_fail"] = True
    api_mock_config["fail_count"] = count


# When Steps


@when("I execute Step 3 clustering")
def execute_step3(
    step3_config: dict,
    input_articles: dict,
    step3_result: dict,
    api_mock_config: dict,
) -> None:
    """Execute Step 3 clustering with mocked API."""
    import time

    def create_mock_response(articles: list[ProcessedArticle]) -> GeminiClusteringResponse:
        """Create realistic mock response based on articles."""
        from src.steps.step3_clustering import _generate_news_id

        # Enhanced clustering logic for testing:
        # Group articles by shared meaningful keywords (e.g., "GPT-5", "OpenAI")
        clusters_dict = {}

        for article in articles:
            title_lower = article.title.lower()

            # Check for specific topics to cluster together
            cluster_key = None

            # GPT-5 related articles
            if "gpt-5" in title_lower or "gpt5" in title_lower:
                cluster_key = "gpt-5-release"
            # AI Regulation articles
            elif "regulation" in title_lower or "ai reg" in title_lower:
                cluster_key = "ai-regulation"
            # Tesla/Autopilot articles
            elif "tesla" in title_lower or "autopilot" in title_lower:
                cluster_key = "tesla-autopilot"
            # Quantum computing articles
            elif "quantum" in title_lower:
                cluster_key = "quantum-computing"
            # DeepMind articles
            elif "deepmind" in title_lower:
                cluster_key = "deepmind-research"
            # Meta AI articles
            elif "meta" in title_lower:
                cluster_key = "meta-ai"
            # AI Article (generic test articles)
            elif "ai article" in title_lower:
                cluster_key = "ai-article-general"
            # OpenAI articles (general)
            elif "openai" in title_lower:
                cluster_key = "openai-news"
            # Industry reaction articles
            elif "industry" in title_lower and "react" in title_lower:
                cluster_key = "industry-reaction"
            else:
                # Default: use article slug as unique cluster
                cluster_key = f"singleton-{article.slug}"

            if cluster_key not in clusters_dict:
                clusters_dict[cluster_key] = []
            clusters_dict[cluster_key].append(article)

        # Create NewsCluster objects
        clusters = []
        for key, arts in clusters_dict.items():
            # Generate cluster title from first article or combine titles
            if len(arts) > 1:
                # Multi-article cluster: use first article title (cleaned)
                base_title = arts[0].title.split(":")[0] if ":" in arts[0].title else arts[0].title
                title = base_title[:150]
            else:
                # Singleton: use article title
                title = arts[0].title[:150]

            # Ensure title is at least 10 chars
            if len(title) < 10:
                title = f"News: {title}"

            # Generate summary
            titles_combined = " ".join([a.title for a in arts[:3]])
            summary = f"This cluster discusses {key.replace('-', ' ')}. Articles: {titles_combined}"
            summary = summary[:500]

            # Ensure minimum length
            while len(summary) < 50:
                summary += " Additional context about the news topic and its implications."

            # Generate proper news_id using the helper function
            slugs = [a.slug for a in arts]
            news_id = _generate_news_id(title, slugs)

            # Determine main topic
            main_topic = "model release" if "gpt" in key.lower() or "model" in key.lower() else "general"

            # Extract keywords from cluster key
            keywords = [w for w in key.split("-") if w and w != "singleton"][:10]
            if not keywords:
                keywords = ["news", "topic"]

            cluster = NewsCluster(
                news_id=news_id,
                title=title,
                summary=summary,
                article_slugs=slugs,
                article_count=len(arts),
                main_topic=main_topic,
                keywords=keywords,
                created_at=datetime.now(),
            )
            clusters.append(cluster)

        return GeminiClusteringResponse(
            clusters=clusters,
            total_articles_processed=len(articles),
            clustering_rationale=f"Grouped {len(articles)} articles into {len(clusters)} clusters.",
        )

    # Setup API mock
    call_count = [0]  # Use list for mutable counter in closure

    # Determine API key first
    api_key = None if api_mock_config.get("no_api_key") else "test-api-key"

    # If no API key, don't mock - let the real code handle the failure
    if api_mock_config.get("no_api_key"):
        # No mocking - let real code fail due to missing API key
        start_time = time.time()
        result = asyncio.run(
            run_step3(step3_config["config"], input_articles["articles"], api_key=api_key)
        )
        execution_time = time.time() - start_time

        step3_result["result"] = result
        step3_result["execution_time"] = execution_time
        step3_result["mock_call"] = None

    # For retry test, we need to mock at a lower level to preserve retry decorator
    elif api_mock_config.get("fail_count", 999) < 999:
        # Retry test: mock the Gemini client to let retry decorator work
        class MockResponse:
            def __init__(self, text):
                self.text = text

        def mock_generate_content(*args, **kwargs):
            call_count[0] += 1
            fail_limit = api_mock_config.get("fail_count", 0)

            if call_count[0] <= fail_limit:
                raise Exception("API timeout (mocked)")

            # After fail_count failures, succeed
            response = create_mock_response(input_articles["articles"])
            return MockResponse(text=response.model_dump_json())

        with patch("google.genai.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.models.generate_content = mock_generate_content

            start_time = time.time()
            result = asyncio.run(
                run_step3(step3_config["config"], input_articles["articles"], api_key=api_key)
            )
            execution_time = time.time() - start_time

            step3_result["result"] = result
            step3_result["execution_time"] = execution_time
            step3_result["mock_call"] = None
    else:
        # Normal test: mock _call_gemini_clustering

        async def mock_api_call(*args, **kwargs):
            call_count[0] += 1

            if api_mock_config.get("should_fail", False):
                raise Exception("API timeout (mocked)")

            # Return mock response
            return create_mock_response(input_articles["articles"])

        with patch(
            "src.steps.step3_clustering._call_gemini_clustering",
            new_callable=AsyncMock,
        ) as mock_call:
            mock_call.side_effect = mock_api_call

            start_time = time.time()
            result = asyncio.run(
                run_step3(step3_config["config"], input_articles["articles"], api_key=api_key)
            )
            execution_time = time.time() - start_time

            step3_result["result"] = result
            step3_result["execution_time"] = execution_time
            step3_result["mock_call"] = mock_call


# Then Steps - Success and Basic Stats


@then("Step 3 should succeed")
def step3_succeeds(step3_result: dict) -> None:
    """Verify Step 3 succeeded."""
    result: Step3Result = step3_result["result"]
    assert result.success is True


@then("Step 3 should fail")
def step3_fails(step3_result: dict) -> None:
    """Verify Step 3 failed."""
    result: Step3Result = step3_result["result"]
    assert result.success is False


@then(parsers.parse("{count:d} news cluster should be created"))
@then(parsers.parse("{count:d} news clusters should be created"))
def clusters_created(step3_result: dict, count: int) -> None:
    """Verify number of clusters created."""
    result: Step3Result = step3_result["result"]
    assert result.total_clusters == count
    assert len(result.news_clusters) == count


@then(parsers.parse("at least {count:d} news cluster should be created"))
@then(parsers.parse("at least {count:d} news clusters should be created"))
def at_least_clusters(step3_result: dict, count: int) -> None:
    """Verify at least N clusters were created."""
    result: Step3Result = step3_result["result"]
    assert result.total_clusters >= count


# Then Steps - Cluster Details


@then(parsers.parse("the cluster should contain {count:d} articles"))
def cluster_contains_articles(step3_result: dict, count: int) -> None:
    """Verify cluster article count."""
    result: Step3Result = step3_result["result"]
    assert len(result.news_clusters) > 0
    # Check first cluster (assuming single cluster scenario)
    assert result.news_clusters[0].article_count == count


@then(parsers.parse("the cluster title should mention {text}"))
def cluster_title_mentions(step3_result: dict, text: str) -> None:
    """Verify cluster title contains specific text."""
    result: Step3Result = step3_result["result"]
    assert len(result.news_clusters) > 0
    # Remove quotes if present
    text = text.strip('"')
    assert text in result.news_clusters[0].title


@then(parsers.parse('the cluster topic should be "{topic}"'))
def cluster_topic_is(step3_result: dict, topic: str) -> None:
    """Verify cluster main topic."""
    result: Step3Result = step3_result["result"]
    assert len(result.news_clusters) > 0
    assert result.news_clusters[0].main_topic == topic


@then("all clusters should be singletons")
@then(parsers.parse("{count:d} singleton clusters should be created"))
def all_singletons(step3_result: dict, count: int = None) -> None:
    """Verify all clusters are singletons."""
    result: Step3Result = step3_result["result"]
    if count is not None:
        assert result.singleton_clusters == count
    else:
        assert result.singleton_clusters == result.total_clusters
    assert result.multi_article_clusters == 0


@then(parsers.parse("each cluster should have {count:d} article"))
def each_cluster_has_n_articles(step3_result: dict, count: int) -> None:
    """Verify each cluster has N articles."""
    result: Step3Result = step3_result["result"]
    for cluster in result.news_clusters:
        assert cluster.article_count == count


@then("the cluster should be a singleton")
def cluster_is_singleton(step3_result: dict) -> None:
    """Verify cluster is a singleton."""
    result: Step3Result = step3_result["result"]
    assert result.singleton_clusters == 1
    assert result.multi_article_clusters == 0


@then(parsers.parse("at least {count:d} cluster should have multiple articles"))
def at_least_multi_article(step3_result: dict, count: int) -> None:
    """Verify at least N multi-article clusters."""
    result: Step3Result = step3_result["result"]
    assert result.multi_article_clusters >= count


@then(parsers.parse("at least {count:d} cluster should be a singleton"))
def at_least_singleton(step3_result: dict, count: int) -> None:
    """Verify at least N singleton clusters."""
    result: Step3Result = step3_result["result"]
    assert result.singleton_clusters >= count


# Then Steps - API and Errors


@then(parsers.parse("{count:d} API calls should be made"))
@then(parsers.parse("{count:d} API call should be made"))
def api_calls_made(step3_result: dict, count: int) -> None:
    """Verify number of API calls."""
    result: Step3Result = step3_result["result"]
    assert result.api_calls == count


@then(parsers.parse("exactly {count:d} API call should be made"))
def exactly_api_calls(step3_result: dict, count: int) -> None:
    """Verify exact number of API calls."""
    result: Step3Result = step3_result["result"]
    assert result.api_calls == count


@then(parsers.parse("exactly {count:d} successful API call should be recorded"))
def exactly_successful_api_calls(step3_result: dict, count: int) -> None:
    """Verify exact number of successful API calls."""
    result: Step3Result = step3_result["result"]
    assert result.api_calls == count


@then(parsers.parse("{count:d} API failure should be recorded"))
@then(parsers.parse("{count:d} API failures should be recorded"))
def api_failures_recorded(step3_result: dict, count: int) -> None:
    """Verify number of API failures."""
    result: Step3Result = step3_result["result"]
    assert result.api_failures == count


@then("the fallback flag should be true")
def fallback_flag_true(step3_result: dict) -> None:
    """Verify fallback was used."""
    result: Step3Result = step3_result["result"]
    assert result.fallback_used is True


@then("an error message should be present")
def error_message_present(step3_result: dict) -> None:
    """Verify error messages exist."""
    result: Step3Result = step3_result["result"]
    assert len(result.errors) > 0


@then("news clusters should be created")
def news_clusters_created(step3_result: dict) -> None:
    """Verify news clusters were created."""
    result: Step3Result = step3_result["result"]
    assert len(result.news_clusters) > 0


# Then Steps - Validation


@then("each cluster should have a unique news_id")
def unique_news_ids(step3_result: dict) -> None:
    """Verify all news IDs are unique."""
    result: Step3Result = step3_result["result"]
    news_ids = [c.news_id for c in result.news_clusters]
    assert len(news_ids) == len(set(news_ids))


@then('each news_id should start with "news-"')
def news_ids_format(step3_result: dict) -> None:
    """Verify news ID format."""
    result: Step3Result = step3_result["result"]
    for cluster in result.news_clusters:
        assert cluster.news_id.startswith("news-")


@then(parsers.parse("each news_id should be {length:d} characters long"))
def news_id_length(step3_result: dict, length: int) -> None:
    """Verify news ID length."""
    result: Step3Result = step3_result["result"]
    for cluster in result.news_clusters:
        assert len(cluster.news_id) == length


@then(parsers.parse("each cluster title should be at least {length:d} characters"))
def title_min_length(step3_result: dict, length: int) -> None:
    """Verify cluster title minimum length."""
    result: Step3Result = step3_result["result"]
    for cluster in result.news_clusters:
        assert len(cluster.title) >= length


@then(parsers.parse("each cluster summary should be at least {length:d} characters"))
def summary_min_length(step3_result: dict, length: int) -> None:
    """Verify cluster summary minimum length."""
    result: Step3Result = step3_result["result"]
    for cluster in result.news_clusters:
        assert len(cluster.summary) >= length


@then(parsers.parse("cluster titles should not exceed {length:d} characters"))
def title_max_length(step3_result: dict, length: int) -> None:
    """Verify cluster title maximum length."""
    result: Step3Result = step3_result["result"]
    for cluster in result.news_clusters:
        assert len(cluster.title) <= length


@then(parsers.parse("cluster summaries should not exceed {length:d} characters"))
def summary_max_length(step3_result: dict, length: int) -> None:
    """Verify cluster summary maximum length."""
    result: Step3Result = step3_result["result"]
    for cluster in result.news_clusters:
        assert len(cluster.summary) <= length


# Then Steps - Performance


@then(parsers.parse("Step 3 should complete in less than {seconds:d} seconds"))
def completes_in_time(step3_result: dict, seconds: int) -> None:
    """Verify execution time."""
    execution_time = step3_result["execution_time"]
    assert execution_time < seconds


# Then Steps - Statistics


@then(parsers.parse("the total articles clustered should be {count:d}"))
def total_articles_clustered(step3_result: dict, count: int) -> None:
    """Verify total articles clustered."""
    result: Step3Result = step3_result["result"]
    assert result.articles_clustered == count


@then("the statistics should report:")
def statistics_report(datatable, step3_result: dict) -> None:
    """Verify detailed statistics."""
    result: Step3Result = step3_result["result"]

    headers = datatable[0] if datatable else []
    metric_idx = headers.index("metric") if "metric" in headers else 0
    value_idx = headers.index("value") if "value" in headers else 1

    for row in datatable[1:]:
        metric = row[metric_idx]
        expected_value = row[value_idx]

        if metric == "total_clusters":
            if expected_value.startswith(">="):
                assert result.total_clusters >= int(expected_value[2:].strip())
            else:
                assert result.total_clusters == int(expected_value)
        elif metric == "articles_clustered":
            assert result.articles_clustered == int(expected_value)
        elif metric == "api_calls":
            assert result.api_calls == int(expected_value)
        elif metric == "api_failures":
            assert result.api_failures == int(expected_value)


# Then Steps - Content Quality


@then("each cluster should have keywords")
def has_keywords(step3_result: dict) -> None:
    """Verify clusters have keywords."""
    result: Step3Result = step3_result["result"]
    for cluster in result.news_clusters:
        assert len(cluster.keywords) > 0


@then(parsers.parse("keywords should not exceed {count:d} per cluster"))
def keywords_limit(step3_result: dict, count: int) -> None:
    """Verify keyword count limit."""
    result: Step3Result = step3_result["result"]
    for cluster in result.news_clusters:
        assert len(cluster.keywords) <= count


@then("cluster article_slugs should match input slugs")
def slugs_match(step3_result: dict, input_articles: dict) -> None:
    """Verify cluster slugs match input."""
    result: Step3Result = step3_result["result"]
    input_slugs = {a.slug for a in input_articles["articles"]}
    cluster_slugs = set()
    for cluster in result.news_clusters:
        cluster_slugs.update(cluster.article_slugs)
    assert cluster_slugs == input_slugs


@then("cluster article_count should match slugs length")
def count_matches_slugs(step3_result: dict) -> None:
    """Verify article count matches slugs length."""
    result: Step3Result = step3_result["result"]
    for cluster in result.news_clusters:
        assert cluster.article_count == len(cluster.article_slugs)


@then("for each cluster article_count should equal slugs length")
def for_each_count_equals_slugs(step3_result: dict) -> None:
    """Verify article count equals slugs length for all clusters."""
    result: Step3Result = step3_result["result"]
    for cluster in result.news_clusters:
        assert cluster.article_count == len(cluster.article_slugs)


@then("the cluster should have a coherent topic")
def has_coherent_topic(step3_result: dict) -> None:
    """Verify cluster has a topic."""
    result: Step3Result = step3_result["result"]
    assert len(result.news_clusters) > 0
    assert result.news_clusters[0].main_topic is not None
    assert len(result.news_clusters[0].main_topic) > 0


@then("the cluster summary should synthesize all articles")
def summary_synthesizes(step3_result: dict) -> None:
    """Verify summary exists and is substantial."""
    result: Step3Result = step3_result["result"]
    assert len(result.news_clusters) > 0
    assert len(result.news_clusters[0].summary) >= 50


@then("the cluster should identify main topic")
def identifies_main_topic(step3_result: dict) -> None:
    """Verify main topic is identified."""
    result: Step3Result = step3_result["result"]
    assert len(result.news_clusters) > 0
    assert result.news_clusters[0].main_topic is not None


@then("all singleton titles should be at least 10 characters")
def singleton_titles_valid(step3_result: dict) -> None:
    """Verify singleton cluster titles meet minimum length."""
    result: Step3Result = step3_result["result"]
    for cluster in result.news_clusters:
        if cluster.article_count == 1:
            assert len(cluster.title) >= 10


@then("all singleton summaries should be at least 50 characters")
def singleton_summaries_valid(step3_result: dict) -> None:
    """Verify singleton cluster summaries meet minimum length."""
    result: Step3Result = step3_result["result"]
    for cluster in result.news_clusters:
        if cluster.article_count == 1:
            assert len(cluster.summary) >= 50
