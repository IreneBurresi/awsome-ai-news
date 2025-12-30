"""Unit tests for slug utilities."""

from src.utils.slug import generate_slug, generate_unique_slug


class TestGenerateSlug:
    """Test generate_slug function."""

    def test_basic_slug_generation(self) -> None:
        """Test basic slug generation."""
        assert generate_slug("Hello World") == "hello-world"
        assert generate_slug("OpenAI Releases GPT-5") == "openai-releases-gpt-5"

    def test_slug_with_special_characters(self) -> None:
        """Test slug generation with special characters."""
        assert generate_slug("AI & ML: The Future!") == "ai-ml-the-future"
        assert generate_slug("Test (2024) - Part 1") == "test-2024-part-1"

    def test_slug_max_length(self) -> None:
        """Test slug generation with max length."""
        long_title = "This is a very long title that should be truncated"
        slug = generate_slug(long_title, max_length=20)
        assert len(slug) <= 20
        assert slug == "this-is-a-very-long"

    def test_slug_with_unicode(self) -> None:
        """Test slug generation with unicode characters."""
        assert generate_slug("Café résumé") == "cafe-resume"
        assert generate_slug("日本語タイトル") == "ri-ben-yu-taitoru"

    def test_slug_empty_string(self) -> None:
        """Test slug generation with empty string."""
        assert generate_slug("") == "untitled"
        assert generate_slug("   ") == "untitled"

    def test_slug_only_special_chars(self) -> None:
        """Test slug generation with only special characters."""
        assert generate_slug("!!!???") == "untitled"
        assert generate_slug("@#$%") == "untitled"


class TestGenerateUniqueSlug:
    """Test generate_unique_slug function."""

    def test_unique_slug_no_collision(self) -> None:
        """Test unique slug when no collision."""
        existing = set()
        slug = generate_unique_slug("Test Article", existing)
        assert slug == "test-article"

    def test_unique_slug_with_collision(self) -> None:
        """Test unique slug when collision exists."""
        existing = {"test-article"}
        slug = generate_unique_slug("Test Article", existing)
        assert slug == "test-article-2"

    def test_unique_slug_multiple_collisions(self) -> None:
        """Test unique slug with multiple collisions."""
        existing = {"test-article", "test-article-2", "test-article-3"}
        slug = generate_unique_slug("Test Article", existing)
        assert slug == "test-article-4"

    def test_unique_slug_empty_set(self) -> None:
        """Test unique slug with empty set."""
        slug = generate_unique_slug("New Article", set())
        assert slug == "new-article"

    def test_unique_slug_preserves_max_length(self) -> None:
        """Test unique slug respects max length."""
        long_title = "This is a very long title"
        existing = set()
        slug = generate_unique_slug(long_title, existing, max_length=15)
        assert len(slug) <= 15
