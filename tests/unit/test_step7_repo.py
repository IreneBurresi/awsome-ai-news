"""Unit tests for Step 7: Repository Update."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.config import Step7Config
from src.models.news import (
    CategorizedNews,
    Citation,
    EnhancedNews,
    ExternalLink,
    NewsCategory,
    NewsCluster,
)
from src.steps.step7_repo import (
    _create_daily_news_file,
    _create_readme_template,
    _generate_news_section,
    _update_archive,
    _update_readme,
    run_step7,
)


@pytest.fixture
def step7_config() -> Step7Config:
    """Step 7 configuration fixture."""
    return Step7Config(
        enabled=True,
        output_file="README.md",
        archive_enabled=True,
        archive_dir="archive",
        commit_message_template="Update AI news - {date}",
        git_push=False,
    )


@pytest.fixture
def sample_enhanced_news() -> list[EnhancedNews]:
    """Sample enhanced news for testing."""
    return [
        EnhancedNews(
            news=CategorizedNews(
                news_cluster=NewsCluster(
                    news_id="test-001",
                    title="GPT-5 Released by OpenAI",
                    summary="OpenAI announces GPT-5 with unprecedented capabilities.",
                    article_slugs=["gpt5-test"],
                    article_count=1,
                    main_topic="model release",
                    keywords=["GPT-5", "OpenAI"],
                    created_at=datetime.utcnow(),
                ),
                category=NewsCategory.MODEL_RELEASE,
                importance_score=9.5,
                reasoning="Major model release",
            ),
            abstract="OpenAI announces GPT-5 with unprecedented capabilities in reasoning and accuracy.",
            extended_summary="OpenAI has officially released GPT-5, marking a significant milestone in artificial intelligence development. The new model demonstrates unprecedented capabilities in reasoning, understanding complex queries, and generating human-like responses with improved factual accuracy and reduced hallucinations compared to previous versions.",
            external_links=[
                ExternalLink(
                    url="https://openai.com/blog/gpt5",
                    title="GPT-5 Release",
                    source="openai.com",
                    citations=[
                        Citation(
                            text="Official release details published by OpenAI.",
                            source="OpenAI Blog",
                        )
                    ],
                )
            ],
            citations=[
                Citation(
                    text="Most capable model ever",
                    author="Sam Altman",
                    source="OpenAI CEO",
                )
            ],
            key_points=["Major advancement", "Better reasoning"],
            enhanced_at=datetime.utcnow(),
            grounded=True,
        ),
    ]


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create temporary test directory."""
    test_dir = tmp_path / "test_repo"
    test_dir.mkdir()
    return test_dir


def test_create_readme_template() -> None:
    """Test README template creation."""
    template = _create_readme_template()

    assert "# 🤖 Awesome AI News" in template
    assert "<!-- NEWS_START -->" in template
    assert "<!-- NEWS_END -->" in template
    assert "Gemini" in template


def test_generate_news_section(sample_enhanced_news: list[EnhancedNews]) -> None:
    """Test news section generation."""
    date = "2025-12-25"
    section = _generate_news_section(sample_enhanced_news, date)

    assert f"## 📰 Latest AI News - {date}" in section
    assert "GPT-5" in section
    assert "🚀" in section  # MODEL_RELEASE emoji
    assert "**Category**" in section
    assert "**Key Points:**" in section
    assert "**Sources:**" in section


def test_create_daily_news_file(
    sample_enhanced_news: list[EnhancedNews], temp_test_dir: Path, monkeypatch
) -> None:
    """Test daily YAML file creation."""
    # Change to temp directory
    monkeypatch.chdir(temp_test_dir)

    news_file = _create_daily_news_file(sample_enhanced_news)

    assert news_file.exists()
    assert news_file.name.endswith(".yaml")
    assert "news" in str(news_file)

    # Read and verify content
    import yaml

    with open(news_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["total_news"] == 1
    assert len(data["news"]) == 1
    assert data["news"][0]["title"] == "GPT-5 Released by OpenAI"
    assert data["news"][0]["category"] == "model_release"
    link_entry = data["news"][0]["external_links"][0]
    assert link_entry["citations"] == [
        '"Official release details published by OpenAI." - OpenAI Blog'
    ]


def test_create_daily_news_file_uses_fallback_citations(
    sample_enhanced_news: list[EnhancedNews], temp_test_dir: Path, monkeypatch
) -> None:
    """Ensure fallback citations are used when link-level data is missing."""
    monkeypatch.chdir(temp_test_dir)

    news_copy = sample_enhanced_news[0].model_copy(deep=True)
    news_copy.external_links[0].citations = []
    news_copy.citations.append(Citation(text="Fallback quote from OpenAI.", source="OpenAI"))

    news_file = _create_daily_news_file([news_copy])

    import yaml

    with open(news_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    link_entry = data["news"][0]["external_links"][0]
    assert link_entry["citations"] == ['"Fallback quote from OpenAI." - OpenAI']


def test_update_readme_creates_new(
    sample_enhanced_news: list[EnhancedNews], temp_test_dir: Path, monkeypatch
) -> None:
    """Test README creation when file doesn't exist."""
    monkeypatch.chdir(temp_test_dir)
    readme_path = temp_test_dir / "README.md"

    _update_readme(readme_path, sample_enhanced_news)

    assert readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")
    assert "# 🤖 Awesome AI News" in content
    assert "GPT-5" in content
    assert "<!-- NEWS_START -->" in content


def test_update_readme_updates_existing(
    sample_enhanced_news: list[EnhancedNews], temp_test_dir: Path, monkeypatch
) -> None:
    """Test README update when file exists."""
    monkeypatch.chdir(temp_test_dir)
    readme_path = temp_test_dir / "README.md"

    # Create initial README
    initial_content = """# Test README

<!-- NEWS_START -->
Old news here
<!-- NEWS_END -->

Footer content
"""
    readme_path.write_text(initial_content, encoding="utf-8")

    # Update README
    _update_readme(readme_path, sample_enhanced_news)

    content = readme_path.read_text(encoding="utf-8")
    assert "GPT-5" in content
    assert "Old news here" not in content
    assert "Footer content" in content
    assert "<!-- NEWS_START -->" in content
    assert "<!-- NEWS_END -->" in content


def test_update_archive(
    sample_enhanced_news: list[EnhancedNews], temp_test_dir: Path, monkeypatch
) -> None:
    """Test archive update."""
    monkeypatch.chdir(temp_test_dir)
    archive_dir = temp_test_dir / "archive"

    _update_archive(archive_dir, sample_enhanced_news)

    # Check monthly directory was created
    now = datetime.now()
    month_dir = archive_dir / f"{now.year}-{now.month:02d}"
    assert month_dir.exists()

    # Check README was created
    archive_readme = month_dir / "README.md"
    assert archive_readme.exists()

    content = archive_readme.read_text(encoding="utf-8")
    assert "GPT-5" in content
    assert f"# AI News Archive - {now.strftime('%B %Y')}" in content


@pytest.mark.asyncio
async def test_run_step7_disabled(
    step7_config: Step7Config, sample_enhanced_news: list[EnhancedNews]
) -> None:
    """Test Step 7 when disabled."""
    step7_config.enabled = False

    result = await run_step7(step7_config, sample_enhanced_news, dry_run=True)

    assert result.success is True
    assert result.readme_updated is False
    assert result.news_file_created is None
    assert result.files_changed == 0


@pytest.mark.asyncio
async def test_run_step7_empty_input(step7_config: Step7Config) -> None:
    """Test Step 7 with empty news list."""
    result = await run_step7(step7_config, [], dry_run=True)

    assert result.success is True
    assert result.readme_updated is False
    assert result.news_file_created is None
    assert result.files_changed == 0


@pytest.mark.asyncio
async def test_run_step7_dry_run(
    step7_config: Step7Config,
    sample_enhanced_news: list[EnhancedNews],
    temp_test_dir: Path,
    monkeypatch,
) -> None:
    """Test Step 7 in dry run mode."""
    monkeypatch.chdir(temp_test_dir)

    result = await run_step7(step7_config, sample_enhanced_news, dry_run=True)

    assert result.success is True
    assert result.readme_updated is True
    assert result.news_file_created is not None
    assert result.archive_updated is True
    assert result.commit_created is False  # No commit in dry run
    assert result.pushed_to_remote is False
    assert result.files_changed == 3  # README, news file, archive


@pytest.mark.asyncio
async def test_run_step7_creates_files(
    step7_config: Step7Config,
    sample_enhanced_news: list[EnhancedNews],
    temp_test_dir: Path,
    monkeypatch,
) -> None:
    """Test that Step 7 creates all expected files."""
    monkeypatch.chdir(temp_test_dir)

    result = await run_step7(step7_config, sample_enhanced_news, dry_run=True)

    # Check README
    readme_path = temp_test_dir / "README.md"
    assert readme_path.exists()

    # Check news file
    assert result.news_file_created.exists()

    # Check archive
    now = datetime.now()
    archive_readme = temp_test_dir / "archive" / f"{now.year}-{now.month:02d}" / "README.md"
    assert archive_readme.exists()


@pytest.mark.asyncio
async def test_run_step7_with_git_operations(
    step7_config: Step7Config,
    sample_enhanced_news: list[EnhancedNews],
    temp_test_dir: Path,
    monkeypatch,
) -> None:
    """Test Step 7 with Git operations (mocked)."""
    monkeypatch.chdir(temp_test_dir)

    # Mock subprocess calls
    with patch("subprocess.run") as mock_run:
        # Mock git add
        mock_run.return_value = MagicMock(stdout="", stderr="")

        # When we get commit hash
        def side_effect(*args, **kwargs):
            if args[0][1] == "rev-parse":
                return MagicMock(stdout="abc123def456\n")
            return MagicMock(stdout="", stderr="")

        mock_run.side_effect = side_effect

        result = await run_step7(step7_config, sample_enhanced_news, dry_run=False)

        assert result.commit_created is True
        assert result.commit_info is not None
        assert result.commit_info.commit_hash == "abc123de"  # Short hash


@pytest.mark.asyncio
async def test_run_step7_archive_disabled(
    step7_config: Step7Config,
    sample_enhanced_news: list[EnhancedNews],
    temp_test_dir: Path,
    monkeypatch,
) -> None:
    """Test Step 7 with archive disabled."""
    monkeypatch.chdir(temp_test_dir)
    step7_config.archive_enabled = False

    result = await run_step7(step7_config, sample_enhanced_news, dry_run=True)

    assert result.success is True
    assert result.archive_updated is False
    assert result.files_changed == 2  # Only README and news file


@pytest.mark.asyncio
async def test_run_step7_handles_file_errors(
    step7_config: Step7Config,
    sample_enhanced_news: list[EnhancedNews],
    temp_test_dir: Path,
    monkeypatch,
) -> None:
    """Test Step 7 error handling for file operations."""
    monkeypatch.chdir(temp_test_dir)

    # Make news directory read-only to cause write error
    news_dir = temp_test_dir / "news"
    news_dir.mkdir()
    news_dir.chmod(0o444)

    try:
        result = await run_step7(step7_config, sample_enhanced_news, dry_run=True)

        # Should still succeed partially
        assert result.readme_updated is True  # README should work
        assert len(result.errors) > 0  # Should have errors
    finally:
        # Cleanup
        news_dir.chmod(0o755)


@pytest.mark.asyncio
async def test_run_step7_multiple_news(
    step7_config: Step7Config, temp_test_dir: Path, monkeypatch
) -> None:
    """Test Step 7 with multiple news items."""
    monkeypatch.chdir(temp_test_dir)

    # Create multiple news items
    news_items = []
    for i in range(5):
        news_items.append(
            EnhancedNews(
                news=CategorizedNews(
                    news_cluster=NewsCluster(
                        news_id=f"test-{i:03d}",
                        title=f"AI News Article {i}",
                        summary=f"Summary for article {i} with enough content to pass validation requirements.",
                        article_slugs=[f"article-{i}"],
                        article_count=1,
                        main_topic="test",
                        keywords=["AI", "test"],
                        created_at=datetime.utcnow(),
                    ),
                    category=NewsCategory.RESEARCH,
                    importance_score=9.0 - i,
                    reasoning="Test",
                ),
                abstract=f"Brief summary for article {i} with key information about AI research developments and implications.",
                extended_summary=f"Extended summary for article {i} with comprehensive details about the topic and its implications for the field and industry at large. This summary provides in-depth analysis of the developments, their significance, and potential impact on various sectors.",
                external_links=[],
                citations=[],
                key_points=[f"Point {j}" for j in range(3)],
                enhanced_at=datetime.utcnow(),
                grounded=False,
            )
        )

    result = await run_step7(step7_config, news_items, dry_run=True)

    assert result.success is True
    assert result.readme_updated is True

    # Verify README contains all news
    readme_path = temp_test_dir / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    for i in range(5):
        assert f"AI News Article {i}" in content
