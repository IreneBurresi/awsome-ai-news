"""Step 5: Top News Selection and Categorization.

Selects the most interesting news items (max 10) and categorizes them using Gemini API.
Only news with high importance scores are selected for the final output.
"""

from collections import defaultdict

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.config import Step5Config
from src.models.news import CategorizedNews, NewsCategory, NewsCluster, Step5Result


class CategorizedNewsItem(BaseModel):
    """Single categorized news item from Gemini."""

    news_id: str = Field(description="Unique news ID")
    category: str = Field(description="News category")
    importance_score: float = Field(ge=0.0, le=10.0, description="Importance score (0-10)")
    reasoning: str = Field(max_length=300, description="Brief reasoning for category and score")


class GeminiCategorizationResponse(BaseModel):
    """Structured response from Gemini for news categorization and scoring."""

    categorized_news: list[CategorizedNewsItem] = Field(
        description="List of news with category and importance score"
    )
    rationale: str = Field(description="Overall categorization strategy explanation")


async def run_step5(
    config: Step5Config,
    news_clusters: list[NewsCluster],
    api_key: str | None = None,
) -> Step5Result:
    """Execute Step 5: Top news selection and categorization.

    Selects the most interesting news (up to max specified in config) based on
    importance scores. Only news scoring above quality threshold are selected.

    Args:
        config: Step 5 configuration (includes target_count and scoring_weights)
        news_clusters: Deduplicated news clusters from Step 4
        api_key: Gemini API key (required if step enabled)

    Returns:
        Step5Result with selected top news (≤ target_count) and all categorized news

    Raises:
        ValueError: If config is invalid
    """
    try:
        logger.info("Starting Step 5: Top news selection and categorization")

        if not config.enabled:
            logger.info("Step 5 disabled, returning empty result")
            return Step5Result(
                success=True,
                top_news=[],
                all_categorized_news=[],
                categories_distribution={},
                api_calls=0,
                api_failures=0,
            )

        # Handle empty input
        if not news_clusters:
            logger.info("No news clusters to categorize")
            return Step5Result(
                success=True,
                top_news=[],
                all_categorized_news=[],
                categories_distribution={},
                api_calls=0,
                api_failures=0,
            )

        # Check for API key
        if not api_key:
            error_msg = "No API key provided for Step 5"
            logger.error(error_msg)
            return Step5Result(
                success=False,
                top_news=[],
                all_categorized_news=[],
                categories_distribution={},
                api_calls=0,
                api_failures=0,
                errors=[error_msg],
            )

        api_calls = 0
        api_failures = 0
        errors: list[str] = []

        try:
            # Call Gemini API for categorization and scoring
            logger.info(f"Categorizing and scoring {len(news_clusters)} news clusters")
            categorization_response = await _call_gemini_categorization(
                news_clusters, config, api_key
            )
            api_calls += 1

            # Parse and validate categorized news
            all_categorized_news = _parse_categorized_news(news_clusters, categorization_response)

            logger.info(
                f"Categorized {len(all_categorized_news)} news clusters",
                total=len(news_clusters),
            )

            # Sort by importance score and select most interesting
            sorted_news = sorted(
                all_categorized_news, key=lambda x: x.importance_score, reverse=True
            )

            # Filter by quality threshold (only include truly interesting news)
            quality_threshold = config.scoring_weights.get("quality_threshold", 6.0)
            interesting_news = [
                news for news in sorted_news if news.importance_score >= quality_threshold
            ]

            # Take top N, up to max target_count
            top_news = interesting_news[: config.target_count]

            logger.info(
                f"Selected {len(top_news)} most interesting news",
                total_candidates=len(all_categorized_news),
                above_threshold=len(interesting_news),
                quality_threshold=quality_threshold,
                max_count=config.target_count,
            )

            # Calculate category distribution
            categories_distribution = _calculate_category_distribution(all_categorized_news)

            logger.info("Step 5 completed successfully")

            return Step5Result(
                success=True,
                top_news=top_news,
                all_categorized_news=all_categorized_news,
                categories_distribution=categories_distribution,
                api_calls=api_calls,
                api_failures=api_failures,
                errors=errors,
            )

        except Exception as e:
            error_msg = f"Gemini API call failed: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            api_failures += 1

            return Step5Result(
                success=False,
                top_news=[],
                all_categorized_news=[],
                categories_distribution={},
                api_calls=api_calls,
                api_failures=api_failures,
                errors=errors,
            )

    except Exception as e:
        error_msg = f"Step 5 failed critically: {e}"
        logger.error(error_msg, exc_info=True)
        return Step5Result(
            success=False,
            top_news=[],
            all_categorized_news=[],
            categories_distribution={},
            api_calls=0,
            api_failures=0,
            errors=[error_msg],
        )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
async def _call_gemini_categorization(
    news_clusters: list[NewsCluster],
    config: Step5Config,
    api_key: str,
) -> GeminiCategorizationResponse:
    """Call Gemini API to categorize and score news.

    Args:
        news_clusters: News clusters to categorize
        config: Step 5 configuration
        api_key: Gemini API key

    Returns:
        GeminiCategorizationResponse with categorized news

    Raises:
        Exception: On API failures after retries
    """
    from google import genai

    from src.utils.prompt_loader import get_prompt_loader

    # Create client
    client = genai.Client(api_key=api_key)

    # Prepare news data for prompt
    news_data = _prepare_news_for_prompt(news_clusters)

    # Build categories description
    categories_desc = "\n".join(
        [f"- {category.value}: {_get_category_description(category)}" for category in NewsCategory]
    )

    # Load and format prompt from YAML
    prompt_loader = get_prompt_loader()
    prompt = prompt_loader.format_prompt(
        "step5_selection",
        num_news=len(news_clusters),
        news_formatted=news_data,
        categories_description=categories_desc,
        recency_weight=config.scoring_weights.get("recency", 0.3),
        source_priority_weight=config.scoring_weights.get("source_priority", 0.3),
        content_quality_weight=config.scoring_weights.get("content_quality", 0.2),
        engagement_weight=config.scoring_weights.get("engagement_potential", 0.2),
    )

    logger.debug("Calling Gemini API for categorization")

    # Make API call with structured output
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config={
            "temperature": 0.3,
            "response_mime_type": "application/json",
            "response_json_schema": GeminiCategorizationResponse.model_json_schema(),
        },
    )

    logger.debug("Gemini API response received", response_text=response.text[:200])  # type: ignore

    # Parse and validate with Pydantic
    categorization_response = GeminiCategorizationResponse.model_validate_json(response.text)  # type: ignore

    return categorization_response


def _prepare_news_for_prompt(news_clusters: list[NewsCluster]) -> str:
    """Format news clusters for inclusion in prompt.

    Args:
        news_clusters: List of news clusters

    Returns:
        Formatted string for prompt
    """
    lines = []
    for i, news in enumerate(news_clusters, 1):
        lines.append(
            f"{i}. [ID: {news.news_id}] {news.title}\n"
            f"   Summary: {news.summary[:200]}...\n"
            f"   Topic: {news.main_topic} | Articles: {news.article_count} | "
            f"Keywords: {', '.join(news.keywords[:5])}\n"
            f"   Created: {news.created_at.strftime('%Y-%m-%d %H:%M')}"
        )

    return "\n\n".join(lines)


def _get_category_description(category: NewsCategory) -> str:
    """Get human-readable description for a category.

    Args:
        category: News category

    Returns:
        Description string
    """
    descriptions = {
        NewsCategory.MODEL_RELEASE: "New AI model releases, major updates to existing models",
        NewsCategory.RESEARCH: "Research papers, academic findings, scientific breakthroughs",
        NewsCategory.POLICY_REGULATION: "AI policy, government regulations, legal frameworks",
        NewsCategory.FUNDING_ACQUISITION: "Funding rounds, investments, acquisitions",
        NewsCategory.PRODUCT_LAUNCH: "New AI products, features, services",
        NewsCategory.PARTNERSHIP: "Company partnerships, collaborations, joint ventures",
        NewsCategory.ETHICS_SAFETY: "AI safety, ethics, responsible AI, alignment",
        NewsCategory.INDUSTRY_NEWS: "Company announcements, industry trends, market news",
        NewsCategory.OTHER: "Other AI-related news that doesn't fit above categories",
    }
    return descriptions.get(category, "Other AI news")


def _parse_categorized_news(
    news_clusters: list[NewsCluster],
    categorization_response: GeminiCategorizationResponse,
) -> list[CategorizedNews]:
    """Parse categorization response and create CategorizedNews objects.

    Args:
        news_clusters: Original news clusters
        categorization_response: Response from Gemini

    Returns:
        List of CategorizedNews objects
    """
    # Create lookup map
    news_map = {news.news_id: news for news in news_clusters}

    categorized_news_list = []
    seen_news_ids: set[str] = set()

    for item in categorization_response.categorized_news:
        news_id = item.news_id
        category_str = item.category
        importance_score = item.importance_score
        reasoning = item.reasoning

        if not news_id:
            logger.warning("Skipping categorized item without news_id: %s", item)
            continue

        if news_id in seen_news_ids:
            logger.warning(
                "Duplicate categorization for news %s, keeping first entry only", news_id
            )
            continue

        seen_news_ids.add(news_id)

        # Find corresponding news cluster
        news_cluster = news_map.get(news_id)
        if not news_cluster:
            logger.warning(f"News cluster not found for ID: {news_id}")
            continue

        # Parse category
        try:
            category = NewsCategory(category_str)
        except ValueError:
            logger.warning(f"Invalid category '{category_str}' for news {news_id}, using OTHER")
            category = NewsCategory.OTHER

        # Validate and clamp importance score
        if importance_score < 0.0:
            importance_score = 0.0
        elif importance_score > 10.0:
            importance_score = 10.0

        categorized_news = CategorizedNews(
            news_cluster=news_cluster,
            category=category,
            importance_score=importance_score,
            reasoning=reasoning,
        )

        categorized_news_list.append(categorized_news)

    logger.info(
        f"Parsed {len(categorized_news_list)} categorized news from {len(news_clusters)} input"
    )

    # If some news were not categorized, add them with default values
    categorized_ids = {cn.news_cluster.news_id for cn in categorized_news_list}
    for news in news_clusters:
        if news.news_id not in categorized_ids:
            logger.warning(f"News {news.news_id} was not categorized, adding with defaults")
            categorized_news_list.append(
                CategorizedNews(
                    news_cluster=news,
                    category=NewsCategory.OTHER,
                    importance_score=5.0,
                    reasoning="Not categorized by LLM, using default values",
                )
            )

    return categorized_news_list


def _calculate_category_distribution(
    categorized_news: list[CategorizedNews],
) -> dict[NewsCategory, int]:
    """Calculate distribution of news across categories.

    Args:
        categorized_news: List of categorized news

    Returns:
        Dictionary mapping categories to counts
    """
    distribution = defaultdict(int)

    for news in categorized_news:
        distribution[news.category] += 1

    return dict(distribution)
