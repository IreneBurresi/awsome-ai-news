"""Step 7: Repository Update with Git operations.

Generates README.md, creates daily news YAML files, manages archive,
and creates Git commits with changes.
"""

import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from loguru import logger

from src.constants import GIT_OPERATION_TIMEOUT_SECONDS
from src.models.config import Step7Config
from src.models.news import Citation, EnhancedNews, ExternalLink
from src.models.repository import CommitInfo, Step7Result


async def run_step7(
    config: Step7Config,
    enhanced_news: list[EnhancedNews],
    dry_run: bool = False,
) -> Step7Result:
    """Execute Step 7: Repository update with Git operations.

    Args:
        config: Step 7 configuration
        enhanced_news: Enhanced news from Step 6
        dry_run: If True, skip actual Git operations

    Returns:
        Step7Result with operation status and commit info

    Raises:
        ValueError: If config is invalid
    """
    try:
        logger.info("Starting Step 7: Repository update")

        if not config.enabled:
            logger.info("Step 7 disabled, skipping repository update")
            return Step7Result(
                success=True,
                readme_updated=False,
                news_file_created=None,
                archive_updated=False,
                commit_created=False,
                pushed_to_remote=False,
                files_changed=0,
            )

        # Handle empty input
        if not enhanced_news:
            logger.info("No enhanced news to process")
            return Step7Result(
                success=True,
                readme_updated=False,
                news_file_created=None,
                archive_updated=False,
                commit_created=False,
                pushed_to_remote=False,
                files_changed=0,
            )

        errors: list[str] = []
        files_changed = 0

        # Create news YAML file
        news_file = None
        try:
            news_file = _create_daily_news_file(enhanced_news)
            files_changed += 1
            logger.info(f"Created daily news file: {news_file}")
        except Exception as e:
            error_msg = f"Failed to create news file: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)

        # Update README.md
        readme_updated = False
        try:
            readme_path = Path(config.output_file)
            _update_readme(readme_path, enhanced_news)
            readme_updated = True
            files_changed += 1
            logger.info(f"Updated README: {readme_path}")
        except Exception as e:
            error_msg = f"Failed to update README: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)

        # Update archive if enabled
        archive_updated = False
        if config.archive_enabled:
            try:
                archive_dir = Path(config.archive_dir)
                _update_archive(archive_dir, enhanced_news)
                archive_updated = True
                files_changed += 1
                logger.info(f"Updated archive: {archive_dir}")
            except Exception as e:
                error_msg = f"Failed to update archive: {e}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)

        # Create Git commit
        commit_info = None
        commit_created = False
        pushed = False

        if not dry_run:
            try:
                commit_info = _create_git_commit(config, files_changed)
                commit_created = True
                logger.info(f"Created Git commit: {commit_info.commit_hash}")

                # Push to remote if enabled
                if config.git_push:
                    _git_push()
                    pushed = True
                    logger.info("Pushed changes to remote")
            except Exception as e:
                error_msg = f"Git operations failed: {e}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
        else:
            logger.info("Dry run mode: skipping Git operations")

        success = len(errors) == 0 or (readme_updated and news_file is not None)

        logger.info(f"Step 7 completed: success={success}")

        return Step7Result(
            success=success,
            readme_updated=readme_updated,
            news_file_created=news_file,
            archive_updated=archive_updated,
            commit_created=commit_created,
            commit_info=commit_info,
            pushed_to_remote=pushed,
            files_changed=files_changed,
            errors=errors,
        )

    except Exception as e:
        error_msg = f"Step 7 failed critically: {e}"
        logger.error(error_msg, exc_info=True)
        return Step7Result(
            success=False,
            readme_updated=False,
            news_file_created=None,
            archive_updated=False,
            commit_created=False,
            pushed_to_remote=False,
            files_changed=0,
            errors=[error_msg],
        )


def _create_daily_news_file(enhanced_news: list[EnhancedNews]) -> Path:
    """Create daily YAML file with news data.

    Args:
        enhanced_news: List of enhanced news items

    Returns:
        Path to created YAML file

    Raises:
        IOError: If file creation fails
    """
    # Create news directory if it doesn't exist
    news_dir = Path("news")
    news_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with today's date
    today = datetime.now().strftime("%Y-%m-%d")
    news_file = news_dir / f"{today}.yaml"

    # Convert news to dict format
    news_data = {
        "date": today,
        "total_news": len(enhanced_news),
        "news": [
            {
                "id": news.news.news_cluster.news_id,
                "title": news.news.news_cluster.title,
                "abstract": news.abstract,
                "summary": news.extended_summary,
                "category": news.news.category.value,
                "importance_score": news.news.importance_score,
                "keywords": news.news.news_cluster.keywords,
                "article_count": news.news.news_cluster.article_count,
                "external_links": [
                    {
                        "title": link.title,
                        "url": str(link.url),
                        "source": link.source,
                        "citations": _format_citation_strings(
                            link, fallback_citations=news.citations
                        ),
                    }
                    for link in news.external_links[:5]  # Max 5 links per news
                ],
                "key_points": news.key_points,
                "grounded": news.grounded,
            }
            for news in enhanced_news
        ],
    }

    # Write YAML file
    with open(news_file, "w", encoding="utf-8") as f:
        yaml.dump(news_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    return news_file


def _update_readme(readme_path: Path, enhanced_news: list[EnhancedNews]) -> None:
    """Update README.md with new news in awesome list style.

    Args:
        readme_path: Path to README.md
        enhanced_news: List of enhanced news items

    Raises:
        IOError: If README update fails
    """
    # Generate new news section
    today = datetime.now().strftime("%Y-%m-%d")
    news_section = _generate_news_section(enhanced_news, today)

    # Read existing README or create template
    if readme_path.exists():
        with open(readme_path, encoding="utf-8") as f:
            content = f.read()
    else:
        content = _create_readme_template()

    # Find news section markers
    start_marker = "<!-- NEWS_START -->"
    end_marker = "<!-- NEWS_END -->"

    if start_marker in content and end_marker in content:
        # Replace existing news section
        before = content.split(start_marker)[0]
        after = content.split(end_marker)[1]
        new_content = f"{before}{start_marker}\n\n{news_section}\n{end_marker}{after}"
    else:
        # Append news section
        new_content = f"{content}\n\n{start_marker}\n\n{news_section}\n{end_marker}\n"

    # Clean old news (>30 days)
    new_content = _clean_old_news(new_content)

    # Write updated README
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def _generate_news_section(
    enhanced_news: list[EnhancedNews], date: str, truncate_summary: bool = True
) -> str:
    """Generate markdown section for news.

    Args:
        enhanced_news: List of enhanced news items
        date: Date string (YYYY-MM-DD)
        truncate_summary: If True, truncate summaries to 300 chars (default: True)

    Returns:
        Markdown formatted news section
    """
    lines = [f"## 📰 Latest AI News - {date}\n"]

    for news in enhanced_news:
        # Category emoji mapping
        category_emoji = {
            "model_release": "🚀",
            "research": "🔬",
            "policy_regulation": "📜",
            "funding_acquisition": "💰",
            "product_launch": "🎯",
            "partnership": "🤝",
            "ethics_safety": "🛡️",
            "industry_news": "📊",
            "other": "📌",
        }
        emoji = category_emoji.get(news.news.category.value, "📌")

        lines.append(f"\n### {emoji} {news.news.news_cluster.title}\n")
        lines.append(f"**Category**: {news.news.category.value.replace('_', ' ').title()}  ")
        lines.append(f"**Score**: {news.news.importance_score}/10  ")
        lines.append(f"**Articles**: {news.news.news_cluster.article_count}\n")

        # Truncate summary only if requested (README), keep full for archive
        if truncate_summary and len(news.extended_summary) > 300:
            lines.append(f"{news.extended_summary[:300]}...\n")
        else:
            lines.append(f"{news.extended_summary}\n")

        if news.key_points:
            lines.append("\n**Key Points:**\n")
            for point in news.key_points[:5]:
                lines.append(f"- {point}\n")

        if news.external_links:
            lines.append("\n**Sources:**\n")
            for link in news.external_links[:3]:
                lines.append(f"- [{link.title}]({link.url})\n")
                # Show citations from this source (indented)
                if link.citations:
                    for cit in link.citations[:2]:  # Max 2 citations per source
                        if cit.author:
                            lines.append(f'  - _"{cit.text}"_ — {cit.author}\n')
                        else:
                            lines.append(f'  - _"{cit.text}"_\n')

    return "".join(lines)


def _format_citation_strings(
    link: ExternalLink, fallback_citations: list[Citation] | None = None
) -> list[str]:
    """Convert structured citations into simple strings for serialization.

    If a link lacks direct citations, attempt to reuse flattened citations that
    match either the link URL or its source label.
    """
    formatted: list[str] = []
    seen: set[str] = set()

    link_citations = list(link.citations)
    if not link_citations and fallback_citations:
        link_citations = _match_fallback_citations(link, fallback_citations)

    for citation in link_citations:
        quote = citation.text.strip()
        if not quote:
            continue

        source_label = citation.source or citation.author or link.source
        entry = f'"{quote}" - {source_label}' if source_label else f'"{quote}"'

        if entry in seen:
            continue

        formatted.append(entry)
        seen.add(entry)

    return formatted


def _match_fallback_citations(
    link: ExternalLink, fallback_citations: list[Citation]
) -> list[Citation]:
    """Match flattened citations to a link using URL or source heuristics."""
    matched: list[Citation] = []
    link_url = str(link.url) if link.url else ""
    link_source_norm = _normalize_label(link.source)

    for citation in fallback_citations:
        if citation in matched:
            continue

        if citation.url and str(citation.url) == link_url:
            matched.append(citation)
            continue

        source_norm = _normalize_label(citation.source or citation.author or "")
        if (
            source_norm
            and link_source_norm
            and (source_norm in link_source_norm or link_source_norm in source_norm)
        ):
            matched.append(citation)

    return matched


def _normalize_label(label: str | None) -> str:
    """Normalize strings for fuzzy comparison."""
    if not label:
        return ""
    return re.sub(r"[^a-z0-9]", "", label.lower())


def _create_readme_template() -> str:
    """Create README template for new repository.

    Returns:
        README template string
    """
    return """# 🤖 Awesome AI News

> Curated AI news aggregated and enhanced by AI

This repository automatically aggregates, deduplicates, and curates the
latest AI news from multiple sources using Gemini AI.

## 📅 Daily Updates

Fresh AI news added daily at 08:00 UTC.

<!-- NEWS_START -->
<!-- NEWS_END -->

## 🗂️ Archive

Historical news available in the [archive](archive/) directory.

## 🔄 Automation

Powered by:
- **Gemini 2.5 Flash** for clustering and enhancement
- **Google Search Grounding** for authoritative sources
- **GitHub Actions** for daily automation

## 📝 License

CC0-1.0
"""


def _clean_old_news(content: str) -> str:
    """Remove news sections older than 30 days from README.

    Parses date headers in format `## YYYY-MM-DD` and removes sections
    where the date is older than 30 days from now.

    Args:
        content: README content with date sections

    Returns:
        Cleaned content with only recent sections (last 30 days)
    """
    cutoff_date = datetime.now() - timedelta(days=30)
    lines = content.split("\n")
    result_lines = []
    current_section_date = None
    keep_current_section = True

    # Pattern to match date headers: ## YYYY-MM-DD
    date_pattern = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})\s*$")

    for line in lines:
        match = date_pattern.match(line)
        if match:
            # Found a date header, parse it
            date_str = match.group(1)
            try:
                current_section_date = datetime.strptime(date_str, "%Y-%m-%d")
                keep_current_section = current_section_date >= cutoff_date
            except ValueError:
                # Invalid date format, keep section by default
                keep_current_section = True
                logger.warning(f"Invalid date format in README: {date_str}")

        if keep_current_section:
            result_lines.append(line)

    cleaned = "\n".join(result_lines)
    logger.info(f"Cleaned old news sections (cutoff: {cutoff_date.strftime('%Y-%m-%d')})")
    return cleaned


def _update_archive(archive_dir: Path, enhanced_news: list[EnhancedNews]) -> None:
    """Update monthly archive with news.

    Args:
        archive_dir: Path to archive directory
        enhanced_news: List of enhanced news items

    Raises:
        IOError: If archive update fails
    """
    # Create archive directory structure: archive/YYYY-MM/
    now = datetime.now()
    month_dir = archive_dir / f"{now.year}-{now.month:02d}"
    month_dir.mkdir(parents=True, exist_ok=True)

    # Create monthly summary file
    summary_file = month_dir / "README.md"

    # Generate summary content with FULL summaries (no truncation for archive)
    today = now.strftime("%Y-%m-%d")
    summary_content = f"# AI News Archive - {now.strftime('%B %Y')}\n\n"
    summary_content += f"## {today}\n\n"
    summary_content += _generate_news_section(enhanced_news, today, truncate_summary=False)

    # Append to existing or create new
    if summary_file.exists():
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{summary_content}")
    else:
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary_content)


def _create_git_commit(config: Step7Config, files_changed: int) -> CommitInfo:
    """Create Git commit with changes.

    Args:
        config: Step 7 configuration
        files_changed: Number of files changed

    Returns:
        CommitInfo with commit details

    Raises:
        subprocess.CalledProcessError: If Git operations fail
    """
    # Stage all changes
    subprocess.run(
        ["git", "add", "."],
        check=True,
        capture_output=True,
        text=True,
        timeout=GIT_OPERATION_TIMEOUT_SECONDS,
    )

    # Generate commit message
    today = datetime.now().strftime("%Y-%m-%d")
    commit_message = config.commit_message_template.format(date=today)

    # Create commit
    subprocess.run(
        ["git", "commit", "-m", commit_message],
        check=True,
        capture_output=True,
        text=True,
        timeout=GIT_OPERATION_TIMEOUT_SECONDS,
    )

    # Get commit hash
    hash_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=GIT_OPERATION_TIMEOUT_SECONDS,
    )
    commit_hash = hash_result.stdout.strip()

    return CommitInfo(
        commit_hash=commit_hash[:8],  # Short hash
        message=commit_message,
        timestamp=datetime.now(),
        author="awesome-ai-news-bot",
        files_changed=files_changed,
    )


def _git_push() -> None:
    """Push changes to remote repository.

    Raises:
        subprocess.CalledProcessError: If push fails
    """
    subprocess.run(
        ["git", "push"],
        check=True,
        capture_output=True,
        text=True,
        timeout=GIT_OPERATION_TIMEOUT_SECONDS,
    )
