"""Step 6: Content Enhancement with Web Grounding.

Uses Gemini API with Google Search grounding to enhance top news with
extended summaries, external links, and citations. Makes ONE call per news.
"""

import re
from datetime import datetime
from urllib.parse import urlparse

import aiohttp
from loguru import logger
from pydantic import HttpUrl, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from src.constants import ABSTRACT_MAX_LENGTH, ABSTRACT_MIN_LENGTH
from src.models.config import Step6Config
from src.models.news import (
    CategorizedNews,
    Citation,
    EnhancedNews,
    ExternalLink,
    Step6Result,
)


async def run_step6(
    config: Step6Config,
    top_news: list[CategorizedNews],
    api_key: str | None = None,
) -> Step6Result:
    """Execute Step 6: Content enhancement with web grounding.

    Args:
        config: Step 6 configuration
        top_news: Top news from Step 5 (up to 10)
        api_key: Gemini API key (required if step enabled)

    Returns:
        Step6Result with enhanced news

    Raises:
        ValueError: If config is invalid
    """
    try:
        logger.info("Starting Step 6: Content enhancement with grounding")

        if not config.enabled:
            logger.info("Step 6 disabled, returning empty result")
            return Step6Result(
                success=True,
                enhanced_news=[],
                total_external_links=0,
                avg_links_per_news=0.0,
                enhancement_failures=0,
                api_calls=0,
                api_failures=0,
            )

        # Handle empty input
        if not top_news:
            logger.info("No news to enhance")
            return Step6Result(
                success=True,
                enhanced_news=[],
                total_external_links=0,
                avg_links_per_news=0.0,
                enhancement_failures=0,
                api_calls=0,
                api_failures=0,
            )

        # Check for API key
        if not api_key:
            error_msg = "No API key provided for Step 6"
            logger.error(error_msg)
            return Step6Result(
                success=False,
                enhanced_news=[],
                total_external_links=0,
                avg_links_per_news=0.0,
                enhancement_failures=0,
                api_calls=0,
                api_failures=0,
                errors=[error_msg],
            )

        api_calls = 0
        api_failures = 0
        errors: list[str] = []
        enhancement_failures = 0
        enhanced_news_list: list[EnhancedNews] = []

        logger.info(f"Enhancing {len(top_news)} news items with ONE Gemini call per news")

        # Process each news item individually
        for idx, cat_news in enumerate(top_news, 1):
            news_id = cat_news.news_cluster.news_id
            logger.info(f"Processing news {idx}/{len(top_news)}: {news_id}")

            try:
                enhanced = await _enhance_single_news(cat_news, config, api_key)
                api_calls += 1
                enhanced_news_list.append(enhanced)
                logger.info(
                    f"Successfully enhanced {news_id}: {len(enhanced.external_links)} links, "
                    f"{len(enhanced.citations)} citations"
                )
            except Exception as exc:
                api_failures += 1
                enhancement_failures += 1
                error_msg = f"Failed to enhance news {news_id}: {exc}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)

        logger.info(f"Enhanced {len(enhanced_news_list)}/{len(top_news)} news items")

        # Calculate statistics
        total_external_links = sum(len(news.external_links) for news in enhanced_news_list)
        avg_links_per_news = (
            total_external_links / len(enhanced_news_list) if enhanced_news_list else 0.0
        )

        success = len(enhanced_news_list) > 0

        logger.info("Step 6 completed successfully")

        return Step6Result(
            success=success,
            enhanced_news=enhanced_news_list,
            total_external_links=total_external_links,
            avg_links_per_news=avg_links_per_news,
            enhancement_failures=enhancement_failures,
            api_calls=api_calls,
            api_failures=api_failures,
            errors=errors,
        )

    except Exception as e:
        error_msg = f"Step 6 failed critically: {e}"
        logger.error(error_msg, exc_info=True)
        return Step6Result(
            success=False,
            enhanced_news=[],
            total_external_links=0,
            avg_links_per_news=0.0,
            enhancement_failures=len(top_news) if top_news else 0,
            api_calls=0,
            api_failures=0,
            errors=[error_msg],
        )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
async def _enhance_single_news(
    cat_news: CategorizedNews,
    config: Step6Config,
    api_key: str,
) -> EnhancedNews:
    """Enhance a single news item with Gemini grounding.

    Args:
        cat_news: Categorized news to enhance
        config: Step 6 configuration
        api_key: Gemini API key

    Returns:
        EnhancedNews with extended summary, links, and citations

    Raises:
        Exception: On API failures after retries
    """
    from google import genai
    from google.genai import types

    from src.utils.prompt_loader import get_prompt_loader

    client = genai.Client(api_key=api_key)
    news = cat_news.news_cluster

    keywords = ", ".join(news.keywords[:8]) if news.keywords else "n/a"

    # Load and format prompt from YAML
    prompt_loader = get_prompt_loader()
    prompt = prompt_loader.format_prompt(
        "step6_enhancement",
        news_id=news.news_id,
        title=news.title,
        category=cat_news.category.value,
        main_topic=news.main_topic,
        summary=news.summary,
        keywords=keywords,
        article_count=news.article_count,
    )

    logger.debug(f"Calling Gemini API with grounding for news {news.news_id}")

    grounding_tool = types.Tool(googleSearch=types.GoogleSearch())
    generation_config = types.GenerateContentConfig(
        temperature=getattr(config, "temperature", 0.3),
        tools=[grounding_tool],
    )

    response = client.models.generate_content(
        model=config.llm_model,
        contents=prompt,
        config=generation_config,
    )

    logger.debug(f"Gemini API response received for {news.news_id}")

    # Extract grounding metadata
    grounding_metadata: dict = {}
    if response.candidates:
        candidate = response.candidates[0]
        if getattr(candidate, "grounding_metadata", None):
            gm = candidate.grounding_metadata
            web_queries = (
                list(gm.web_search_queries)
                if getattr(gm, "web_search_queries", None) is not None
                else []
            )
            grounding_chunks = (
                list(gm.grounding_chunks)
                if getattr(gm, "grounding_chunks", None) is not None
                else []
            )
            grounding_supports = (
                list(gm.grounding_supports)
                if getattr(gm, "grounding_supports", None) is not None
                else []
            )
            logger.debug(
                "Grounding data for %s: %s queries, %s chunks, %s supports",
                news.news_id,
                len(web_queries),
                len(grounding_chunks),
                len(grounding_supports),
            )
            grounding_metadata = {
                "web_search_queries": web_queries,
                "grounding_chunks": grounding_chunks,
                "grounding_supports": grounding_supports,
            }
        else:
            logger.warning(f"Gemini response for {news.news_id} lacks grounding metadata")

    response_text = response.text or ""

    # Parse response
    enhanced = await _parse_single_news_response(
        response_text=response_text,
        grounding_metadata=grounding_metadata,
        cat_news=cat_news,
    )

    return enhanced


def _build_citations_from_grounding(
    grounding_metadata: dict,
    chunk_links: dict[int, ExternalLink],
) -> list[Citation]:
    """Build citations from grounding supports (automatic from Google Search API).

    Citations are extracted from groundingSupports which link text segments
    to source chunks. This is more reliable than parsing LLM-generated text.

    Args:
        grounding_metadata: Grounding metadata from Gemini API response
        chunk_links: External links indexed by chunk index

    Returns:
        List of Citation objects linking text segments to sources
    """
    citations: list[Citation] = []
    supports = grounding_metadata.get("grounding_supports") or []

    logger.debug(f"Building citations from {len(supports)} grounding supports")

    for support_idx, support in enumerate(supports):
        try:
            # Extract segment info
            segment = getattr(support, "segment", None)
            if not segment:
                continue

            segment_text = getattr(segment, "text", None)
            if not segment_text or len(segment_text.strip()) < 10:
                continue  # Skip very short segments

            # Get chunk indices that support this segment
            chunk_indices = getattr(support, "grounding_chunk_indices", [])
            if not chunk_indices:
                continue

            # Use the first (primary) chunk for this citation
            primary_chunk_idx = chunk_indices[0]
            chunk_link = chunk_links.get(primary_chunk_idx)

            if not chunk_link:
                logger.debug(f"Support {support_idx}: chunk {primary_chunk_idx} not found in links")
                continue

            # Create citation
            citation = Citation(
                text=segment_text.strip(),
                author=None,  # Not available in grounding metadata
                source=chunk_link.source or chunk_link.title,
                url=chunk_link.url,
            )
            citations.append(citation)

            logger.debug(
                f"Created citation from support {support_idx}: "
                f"{len(segment_text)} chars → {chunk_link.source}"
            )

        except Exception as exc:
            logger.warning(f"Failed to create citation from support {support_idx}: {exc}")
            continue

    logger.info(f"Built {len(citations)} citations from grounding supports")
    return citations


async def _parse_single_news_response(
    response_text: str,
    grounding_metadata: dict,
    cat_news: CategorizedNews,
) -> EnhancedNews:
    """Parse and validate Gemini API response into structured EnhancedNews."""
    news = cat_news.news_cluster
    news_id = news.news_id

    if not response_text.strip():
        raise ValueError(f"Gemini returned empty response for {news_id}")

    # Extract NEWS block
    pattern = re.compile(
        r"===\s*NEWS START\s*===(.*?)===\s*NEWS END\s*===",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(response_text)
    if not match:
        raise ValueError(f"No NEWS block found in response for {news_id}")

    section_text = match.group(1).strip()

    # Extract abstract
    abstract = _extract_section_text(
        section_text,
        "ABSTRACT",
        stop_headers=["EXTENDED SUMMARY"],
    )
    if not abstract:
        # Fallback: use first 150 chars of original summary
        abstract = news.summary[:150]
    abstract = abstract.strip()

    # Validate abstract length
    if len(abstract) < ABSTRACT_MIN_LENGTH:
        logger.warning(
            "Abstract for %s under %s chars (%s). Using original summary fallback.",
            news_id,
            ABSTRACT_MIN_LENGTH,
            len(abstract),
        )
        abstract = news.summary[:150]

    if len(abstract) > ABSTRACT_MAX_LENGTH:
        logger.warning(
            "Abstract for %s above %s chars (%s). Truncating.",
            news_id,
            ABSTRACT_MAX_LENGTH,
            len(abstract),
        )
        abstract = abstract[: ABSTRACT_MAX_LENGTH - 3] + "..."

    # Extract extended summary
    extended_summary = _extract_section_text(
        section_text,
        "EXTENDED SUMMARY",
        stop_headers=["KEY POINTS"],
    )
    if not extended_summary:
        extended_summary = news.summary
    extended_summary = extended_summary.strip()

    # Validate summary length
    if len(extended_summary) < 200:
        logger.warning(
            "Summary for %s under 200 chars (%s). Using original summary as fallback.",
            news_id,
            len(extended_summary),
        )
        extended_summary = f"{news.summary} {extended_summary}".strip()

    if len(extended_summary) > 4000:
        logger.warning(
            "Summary for %s above 4000 chars (%s). Truncating.",
            news_id,
            len(extended_summary),
        )
        extended_summary = extended_summary[:3997] + "..."

    # Extract key points
    key_points = _extract_key_points(section_text)

    # Extract external links from grounding chunks
    chunk_links = await _extract_external_links(grounding_metadata)

    # Build citations from grounding supports (automatic from API)
    flat_citations = _build_citations_from_grounding(grounding_metadata, chunk_links)

    # External links are all unique chunk links
    external_links = list(chunk_links.values())

    grounded = len(external_links) > 0

    enhanced_news = EnhancedNews(
        news=cat_news,
        citations=flat_citations,
        abstract=abstract,
        extended_summary=extended_summary,
        external_links=external_links[:10],
        key_points=key_points[:7],
        enhanced_at=datetime.utcnow(),
        grounded=grounded,
    )

    logger.debug(
        "Parsed news %s: %s chars, %s links, %s citations",
        news_id,
        len(extended_summary),
        len(enhanced_news.external_links),
        len(flat_citations),
    )

    return enhanced_news


def _extract_section_text(
    section_text: str,
    header: str,
    stop_headers: list[str] | None = None,
) -> str | None:
    """Extract multiline section text bounded by stop headers."""
    if stop_headers:
        stop_pattern = "|".join(rf"{re.escape(h)}\s*:" for h in stop_headers)
        lookahead = rf"(?=(?:{stop_pattern})|===\s*NEWS END\s*===|$)"
    else:
        lookahead = r"(?===\s*NEWS END\s*===|$)"

    pattern = rf"{header}\s*:\s*\n(.*?){lookahead}"
    match = re.search(pattern, section_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def _extract_key_points(section_text: str) -> list[str]:
    """Extract bullet key points from the section."""
    key_points_text = _extract_section_text(section_text, "KEY POINTS", None)
    if not key_points_text:
        return []

    points = re.findall(r"-\s*(.+?)(?=\n-|\n\n|$)", key_points_text, re.DOTALL)
    return [p.strip() for p in points if p.strip()]


async def _resolve_redirect_url(redirect_url: str, timeout: int = 5) -> str:
    """Resolve Google grounding redirect URL to final destination.

    Args:
        redirect_url: Google grounding redirect URL
        timeout: Request timeout in seconds

    Returns:
        Final resolved URL or original URL if resolution fails
    """
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.head(
                redirect_url,
                allow_redirects=True,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response,
        ):
            # Return the final URL after following redirects
            final_url = str(response.url)
            logger.debug(f"Resolved redirect: {redirect_url[:80]}... -> {final_url[:80]}...")
            return final_url
    except TimeoutError:
        logger.warning(f"Timeout resolving redirect URL: {redirect_url[:80]}...")
        return redirect_url
    except Exception as e:
        logger.warning(f"Failed to resolve redirect URL: {e}")
        return redirect_url


async def _extract_external_links(grounding_metadata: dict) -> dict[int, ExternalLink]:
    """Resolve grounding chunks into ExternalLink templates keyed by chunk index."""
    chunk_links: dict[int, ExternalLink] = {}
    grounding_chunks = grounding_metadata.get("grounding_chunks") or []
    logger.info("Processing %s grounding chunks for external links", len(grounding_chunks))

    for idx, chunk in enumerate(grounding_chunks):
        try:
            logger.debug("Chunk %s: type=%s", idx, type(chunk))
            if not hasattr(chunk, "web"):
                continue
            web = chunk.web
            uri = str(web.uri) if hasattr(web, "uri") else None
            title = str(web.title) if hasattr(web, "title") else None
            if not uri or not title:
                continue

            logger.info("Resolving URL %s/%s: %s", idx + 1, len(grounding_chunks), uri[:80])
            resolved_uri = await _resolve_redirect_url(uri)
            parsed = urlparse(resolved_uri)
            source = parsed.netloc or "unknown"

            try:
                link = ExternalLink(
                    url=HttpUrl(resolved_uri),
                    title=title[:200] if len(title) > 200 else title,
                    source=source,
                    relevance_score=1.0,
                    snippet=None,
                )
                chunk_links[idx] = link
            except ValidationError as exc:
                logger.warning("Invalid URL in chunk %s: %s", idx, str(exc)[:100])
        except Exception as exc:
            logger.warning("Failed to extract link from chunk %s: %s", idx, exc, exc_info=True)

    logger.info("Extracted %s external links from grounding", len(chunk_links))
    return chunk_links
