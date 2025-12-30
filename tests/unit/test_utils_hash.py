"""Unit tests for hash utilities."""

from src.utils.hash import calculate_similarity, generate_content_hash, normalize_url


class TestGenerateContentHash:
    """Test generate_content_hash function."""

    def test_basic_hash_generation(self) -> None:
        """Test basic hash generation."""
        hash1 = generate_content_hash("Title", "URL")
        hash2 = generate_content_hash("Title", "URL")
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_hash_different_inputs(self) -> None:
        """Test different inputs produce different hashes."""
        hash1 = generate_content_hash("Title 1", "URL 1")
        hash2 = generate_content_hash("Title 2", "URL 2")
        assert hash1 != hash2

    def test_hash_case_insensitive(self) -> None:
        """Test hash is case insensitive."""
        hash1 = generate_content_hash("Title", "URL")
        hash2 = generate_content_hash("TITLE", "url")
        assert hash1 == hash2

    def test_hash_whitespace_normalized(self) -> None:
        """Test hash normalizes whitespace."""
        hash1 = generate_content_hash("  Title  ", "  URL  ")
        hash2 = generate_content_hash("Title", "URL")
        assert hash1 == hash2

    def test_hash_multiple_fields(self) -> None:
        """Test hash with multiple fields."""
        hash1 = generate_content_hash("Title", "URL", "Content", "Author")
        hash2 = generate_content_hash("Title", "URL", "Content", "Author")
        assert hash1 == hash2

    def test_hash_none_values(self) -> None:
        """Test hash handles None values."""
        hash1 = generate_content_hash("Title", None, "Content")
        hash2 = generate_content_hash("Title", "Content")
        assert hash1 == hash2  # None is filtered out

    def test_hash_empty_string(self) -> None:
        """Test hash with empty strings."""
        hash1 = generate_content_hash("", "URL")
        hash2 = generate_content_hash("URL")
        assert hash1 == hash2  # Empty strings filtered out


class TestNormalizeUrl:
    """Test normalize_url function."""

    def test_normalize_protocol(self) -> None:
        """Test URL protocol normalization."""
        assert normalize_url("https://example.com") == "example.com"
        assert normalize_url("http://example.com") == "example.com"

    def test_normalize_www(self) -> None:
        """Test www prefix removal."""
        assert normalize_url("https://www.example.com") == "example.com"
        assert normalize_url("www.example.com") == "example.com"

    def test_normalize_trailing_slash(self) -> None:
        """Test trailing slash removal."""
        assert normalize_url("https://example.com/") == "example.com"
        assert normalize_url("https://example.com/page/") == "example.com/page"

    def test_normalize_query_params(self) -> None:
        """Test query parameter removal."""
        assert normalize_url("https://example.com/article?utm_source=feed") == "example.com/article"
        assert normalize_url("https://example.com?a=1&b=2") == "example.com"

    def test_normalize_fragment(self) -> None:
        """Test fragment removal."""
        assert normalize_url("https://example.com/page#section") == "example.com/page"
        assert normalize_url("https://example.com#top") == "example.com"

    def test_normalize_case(self) -> None:
        """Test case normalization."""
        assert normalize_url("HTTPS://EXAMPLE.COM/PAGE") == "example.com/page"

    def test_normalize_complex_url(self) -> None:
        """Test complex URL normalization."""
        url = "HTTPS://WWW.Example.COM/Article/123/?utm_source=feed&utm_medium=rss#section"
        assert normalize_url(url) == "example.com/article/123"

    def test_normalize_preserves_path(self) -> None:
        """Test path preservation."""
        assert (
            normalize_url("https://example.com/blog/2024/article")
            == "example.com/blog/2024/article"
        )


class TestCalculateSimilarity:
    """Test calculate_similarity function."""

    def test_similarity_identical(self) -> None:
        """Test similarity for identical hashes."""
        hash1 = "abc123"
        hash2 = "abc123"
        assert calculate_similarity(hash1, hash2) == 1.0

    def test_similarity_different(self) -> None:
        """Test similarity for different hashes."""
        hash1 = "abc123"
        hash2 = "def456"
        assert calculate_similarity(hash1, hash2) == 0.0

    def test_similarity_case_sensitive(self) -> None:
        """Test similarity is case sensitive."""
        hash1 = "abc123"
        hash2 = "ABC123"
        assert calculate_similarity(hash1, hash2) == 0.0

    def test_similarity_empty_strings(self) -> None:
        """Test similarity with empty strings."""
        assert calculate_similarity("", "") == 1.0
        assert calculate_similarity("abc", "") == 0.0
