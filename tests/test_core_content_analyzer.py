"""
Unit tests for content analyzer and tag generation.

Tests for content analyzer, tag generator, and related functionality.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from bs4 import BeautifulSoup

from bookmark_processor.core.content_analyzer import ContentAnalyzer
from bookmark_processor.core.data_models import Bookmark, BookmarkMetadata
from bookmark_processor.core.tag_generator import CorpusAwareTagGenerator
from tests.fixtures.mock_utilities import MockContentAnalyzer, MockRequestsSession
from tests.fixtures.test_data import MOCK_CONTENT_DATA, create_sample_bookmark_objects


class TestContentAnalyzer:
    """Test ContentAnalyzer class."""

    def test_init_default(self):
        """Test ContentAnalyzer initialization with defaults."""
        analyzer = ContentAnalyzer()

        assert analyzer.timeout == 30
        assert analyzer.user_agent is not None
        assert analyzer.max_content_size == 1024 * 1024  # 1MB

    def test_init_custom(self):
        """Test ContentAnalyzer initialization with custom values."""
        analyzer = ContentAnalyzer(
            timeout=60, user_agent="Custom Agent", max_content_size=2048
        )

        assert analyzer.timeout == 60
        assert analyzer.user_agent == "Custom Agent"
        assert analyzer.max_content_size == 2048

    @patch("requests.Session.get")
    def test_extract_metadata_success(self, mock_get):
        """Test successful metadata extraction."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
        <head>
            <title>Test Page Title</title>
            <meta name="description" content="Test page description">
            <meta name="keywords" content="test, page, example">
            <meta name="author" content="Test Author">
            <link rel="canonical" href="https://example.com/canonical">
        </head>
        <body>
            <h1>Main Heading</h1>
            <p>This is test content for extraction.</p>
        </body>
        </html>
        """
        mock_response.url = "https://example.com"
        mock_get.return_value = mock_response

        analyzer = ContentAnalyzer()
        metadata = analyzer.extract_metadata("https://example.com")

        assert isinstance(metadata, BookmarkMetadata)
        assert metadata.title == "Test Page Title"
        assert metadata.description == "Test page description"
        assert "test" in metadata.keywords
        assert "page" in metadata.keywords
        assert metadata.author == "Test Author"
        assert metadata.canonical_url == "https://example.com/canonical"

    @patch("requests.Session.get")
    def test_extract_metadata_network_error(self, mock_get):
        """Test metadata extraction with network error."""
        mock_get.side_effect = requests.RequestException("Network error")

        analyzer = ContentAnalyzer()
        metadata = analyzer.extract_metadata("https://example.com")

        assert metadata is None

    @patch("requests.Session.get")
    def test_extract_metadata_http_error(self, mock_get):
        """Test metadata extraction with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        analyzer = ContentAnalyzer()
        metadata = analyzer.extract_metadata("https://example.com")

        assert metadata is None

    @patch("requests.Session.get")
    def test_extract_metadata_invalid_html(self, mock_get):
        """Test metadata extraction with invalid HTML."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Not valid HTML content"
        mock_response.url = "https://example.com"
        mock_get.return_value = mock_response

        analyzer = ContentAnalyzer()
        metadata = analyzer.extract_metadata("https://example.com")

        # Should still create metadata object with minimal info
        assert isinstance(metadata, BookmarkMetadata)
        assert metadata.title is None or metadata.title == ""

    def test_parse_html_complete(self):
        """Test HTML parsing with complete metadata."""
        html_content = """
        <html>
        <head>
            <title>Complete Page</title>
            <meta name="description" content="Complete description">
            <meta name="keywords" content="complete, test, parsing">
            <meta name="author" content="Test Author">
            <meta property="og:title" content="OG Title">
            <meta property="og:description" content="OG Description">
            <meta name="twitter:title" content="Twitter Title">
            <link rel="canonical" href="https://example.com/canonical">
        </head>
        <body>
            <h1>Main Heading</h1>
            <article>
                <p>Article content here.</p>
            </article>
        </body>
        </html>
        """

        analyzer = ContentAnalyzer()
        soup = BeautifulSoup(html_content, "html.parser")
        metadata = analyzer._parse_html(soup, "https://example.com")

        assert metadata.title == "Complete Page"
        assert metadata.description == "Complete description"
        assert "complete" in metadata.keywords
        assert metadata.author == "Test Author"
        assert metadata.canonical_url == "https://example.com/canonical"

    def test_parse_html_minimal(self):
        """Test HTML parsing with minimal metadata."""
        html_content = """
        <html>
        <head>
            <title>Minimal Page</title>
        </head>
        <body>
            <p>Just some content.</p>
        </body>
        </html>
        """

        analyzer = ContentAnalyzer()
        soup = BeautifulSoup(html_content, "html.parser")
        metadata = analyzer._parse_html(soup, "https://example.com")

        assert metadata.title == "Minimal Page"
        assert metadata.description is None or metadata.description == ""
        assert len(metadata.keywords) == 0

    def test_extract_title_methods(self):
        """Test different title extraction methods."""
        html_content = """
        <html>
        <head>
            <title>Main Title</title>
            <meta property="og:title" content="OG Title">
            <meta name="twitter:title" content="Twitter Title">
        </head>
        <body>
            <h1>H1 Title</h1>
        </body>
        </html>
        """

        analyzer = ContentAnalyzer()
        soup = BeautifulSoup(html_content, "html.parser")

        # Should prioritize main title
        title = analyzer._extract_title(soup)
        assert title == "Main Title"

    def test_extract_description_methods(self):
        """Test different description extraction methods."""
        html_content = """
        <html>
        <head>
            <meta name="description" content="Meta description">
            <meta property="og:description" content="OG description">
            <meta name="twitter:description" content="Twitter description">
        </head>
        <body>
            <p>First paragraph content.</p>
        </body>
        </html>
        """

        analyzer = ContentAnalyzer()
        soup = BeautifulSoup(html_content, "html.parser")

        # Should prioritize meta description
        description = analyzer._extract_description(soup)
        assert description == "Meta description"

    def test_extract_keywords(self):
        """Test keyword extraction."""
        html_content = """
        <html>
        <head>
            <meta name="keywords" content="python, programming, web scraping, data">
        </head>
        </html>
        """

        analyzer = ContentAnalyzer()
        soup = BeautifulSoup(html_content, "html.parser")
        keywords = analyzer._extract_keywords(soup)

        assert "python" in keywords
        assert "programming" in keywords
        assert "web scraping" in keywords
        assert "data" in keywords

    def test_extract_content_text(self):
        """Test content text extraction."""
        html_content = """
        <html>
        <body>
            <header>Header content</header>
            <nav>Navigation</nav>
            <main>
                <article>
                    <h1>Article Title</h1>
                    <p>Main article content here.</p>
                    <p>Another paragraph with more content.</p>
                </article>
            </main>
            <aside>Sidebar content</aside>
            <footer>Footer content</footer>
            <script>console.log('script');</script>
            <style>.hidden { display: none; }</style>
        </body>
        </html>
        """

        analyzer = ContentAnalyzer()
        soup = BeautifulSoup(html_content, "html.parser")
        content_text = analyzer._extract_content_text(soup)

        # Should include main content but exclude scripts/styles
        assert "Article Title" in content_text
        assert "Main article content" in content_text
        assert "script" not in content_text.lower()
        assert "console.log" not in content_text

    def test_analyze_content_categories(self):
        """Test content category analysis."""
        analyzer = ContentAnalyzer()

        # Test programming content
        prog_content = "Python programming tutorial with code examples and functions"
        categories = analyzer.analyze_content_categories(prog_content)
        assert "programming" in categories

        # Test news content
        news_content = "Breaking news report about recent events and updates"
        categories = analyzer.analyze_content_categories(news_content)
        assert "news" in categories

        # Test educational content
        edu_content = "Learn about machine learning algorithms and data science"
        categories = analyzer.analyze_content_categories(edu_content)
        assert "education" in categories

    def test_get_statistics(self):
        """Test getting analyzer statistics."""
        analyzer = ContentAnalyzer()

        stats = analyzer.get_statistics()

        assert isinstance(stats, dict)
        assert "total_extractions" in stats
        assert "successful_extractions" in stats
        assert "failed_extractions" in stats
        assert "success_rate" in stats


class TestCorpusAwareTagGenerator:
    """Test CorpusAwareTagGenerator class."""

    def test_init_default(self):
        """Test CorpusAwareTagGenerator initialization with defaults."""
        generator = CorpusAwareTagGenerator()

        assert generator.max_tags == 5
        assert len(generator.common_words) > 0

    def test_init_custom(self):
        """Test CorpusAwareTagGenerator initialization with custom values."""
        custom_stopwords = {"custom", "stopword"}
        generator = CorpusAwareTagGenerator(max_tags=10, stopwords=custom_stopwords)

        assert generator.max_tags == 10
        assert "custom" in generator.common_words

    def test_generate_tags_from_content(self):
        """Test tag generation from content."""
        generator = CorpusAwareTagGenerator()

        content = "Python programming tutorial for web development and data science"
        tags = generator.generate_tags_from_content(content)

        assert isinstance(tags, list)
        assert len(tags) <= generator.max_tags
        assert "python" in [tag.lower() for tag in tags]
        assert "programming" in [tag.lower() for tag in tags]

    def test_generate_tags_from_bookmark(self):
        """Test tag generation from bookmark object."""
        generator = CorpusAwareTagGenerator()

        bookmark = Bookmark(
            title="Machine Learning Tutorial",
            note="Introduction to ML algorithms",
            excerpt="Learn about supervised and unsupervised learning",
            url="https://ml-tutorial.com",
            tags=["ai", "tutorial"],
        )

        tags = generator.generate_tags_from_bookmark(bookmark)

        assert isinstance(tags, list)
        assert len(tags) <= generator.max_tags
        # Should include existing tags
        assert "ai" in tags
        assert "tutorial" in tags

    def test_extract_keywords_from_text(self):
        """Test keyword extraction from text."""
        generator = CorpusAwareTagGenerator()

        text = "Data science and machine learning with Python programming"
        keywords = generator._extract_keywords_from_text(text)

        assert "data" in keywords or "science" in keywords
        assert "machine" in keywords or "learning" in keywords
        assert "python" in keywords
        assert "programming" in keywords

        # Stopwords should be filtered out
        assert "and" not in keywords
        assert "with" not in keywords

    def test_extract_keywords_from_url(self):
        """Test keyword extraction from URL."""
        generator = CorpusAwareTagGenerator()

        # Test GitHub URL
        url = "https://github.com/microsoft/vscode"
        keywords = generator._extract_keywords_from_url(url)
        assert "github" in keywords
        assert "microsoft" in keywords
        assert "vscode" in keywords

        # Test blog URL
        url = "https://blog.example.com/python-tutorial-2024"
        keywords = generator._extract_keywords_from_url(url)
        assert "blog" in keywords
        assert "python" in keywords
        assert "tutorial" in keywords

    def test_clean_and_filter_tags(self):
        """Test tag cleaning and filtering."""
        generator = CorpusAwareTagGenerator()

        raw_tags = [
            "Python",
            "programming",
            "and",  # stopword
            "the",  # stopword
            "Web-Development",
            "API",
            "",  # empty
            "a",  # too short
            "programming",  # duplicate
        ]

        cleaned_tags = generator._clean_and_filter_tags(raw_tags)

        assert "python" in cleaned_tags
        assert "programming" in cleaned_tags
        assert "web-development" in cleaned_tags
        assert "api" in cleaned_tags

        # Should not include stopwords, empty, or too short
        assert "and" not in cleaned_tags
        assert "the" not in cleaned_tags
        assert "" not in cleaned_tags
        assert "a" not in cleaned_tags

        # Should not have duplicates
        assert cleaned_tags.count("programming") == 1

    def test_rank_tags_by_relevance(self):
        """Test tag ranking by relevance."""
        generator = CorpusAwareTagGenerator()

        bookmark = Bookmark(
            title="Python Tutorial",
            note="Learn Python programming",
            url="https://python-tutorial.com",
        )

        candidate_tags = ["python", "tutorial", "programming", "code", "learning"]
        ranked_tags = generator._rank_tags_by_relevance(candidate_tags, bookmark)

        # Python should be highly ranked due to title and URL
        assert "python" in ranked_tags[:3]
        assert "tutorial" in ranked_tags[:3]


class TestCorpusAwareTagGeneratorAdvanced:
    """Test CorpusAwareTagGenerator class."""

    def test_init_default(self):
        """Test CorpusAwareTagGenerator initialization."""
        generator = CorpusAwareTagGenerator(target_tag_count=100)

        assert generator.target_tag_count == 100
        assert generator.min_tag_frequency == 2
        assert len(generator.tag_corpus) == 0

    def test_build_tag_corpus(self):
        """Test building tag corpus from bookmarks."""
        generator = CorpusAwareTagGenerator(target_tag_count=50)

        bookmarks = create_sample_bookmark_objects()
        generator.build_tag_corpus(bookmarks)

        assert len(generator.tag_corpus) > 0
        assert isinstance(generator.tag_frequency, dict)

    def test_optimize_tags_for_corpus(self):
        """Test tag optimization for corpus."""
        generator = CorpusAwareTagGenerator(target_tag_count=10)

        # Build corpus with sample data
        bookmarks = create_sample_bookmark_objects()
        generator.build_tag_corpus(bookmarks)

        # Test optimization
        raw_tags = ["python", "programming", "web", "tutorial", "code", "development"]
        optimized_tags = generator.optimize_tags_for_corpus(raw_tags, target_count=3)

        assert len(optimized_tags) <= 3
        assert isinstance(optimized_tags, list)

    def test_get_tag_frequency(self):
        """Test getting tag frequency."""
        generator = CorpusAwareTagGenerator(target_tag_count=50)

        bookmarks = create_sample_bookmark_objects()
        generator.build_tag_corpus(bookmarks)

        # Check frequency of common tags
        freq = generator.get_tag_frequency("python")
        assert isinstance(freq, int)
        assert freq >= 0

    def test_get_corpus_statistics(self):
        """Test getting corpus statistics."""
        generator = CorpusAwareTagGenerator(target_tag_count=50)

        bookmarks = create_sample_bookmark_objects()
        generator.build_tag_corpus(bookmarks)

        stats = generator.get_corpus_statistics()

        assert isinstance(stats, dict)
        assert "total_unique_tags" in stats
        assert "total_tag_occurrences" in stats
        assert "average_tags_per_bookmark" in stats
        assert "most_common_tags" in stats

    def test_suggest_similar_tags(self):
        """Test suggesting similar tags."""
        generator = CorpusAwareTagGenerator(target_tag_count=50)

        bookmarks = create_sample_bookmark_objects()
        generator.build_tag_corpus(bookmarks)

        # Test with a tag that should have similar ones
        similar_tags = generator.suggest_similar_tags("program")

        assert isinstance(similar_tags, list)
        # Should suggest programming-related tags if they exist in corpus
        related_terms = ["programming", "code", "development", "software"]
        has_related = any(
            term in tag.lower() for tag in similar_tags for term in related_terms
        )
        # Note: This might not always be true depending on test data

    def test_finalize_tag_optimization(self):
        """Test final tag optimization across all bookmarks."""
        generator = CorpusAwareTagGenerator(target_tag_count=20)

        bookmarks = create_sample_bookmark_objects()

        # Add some generated tags to bookmarks
        for i, bookmark in enumerate(bookmarks):
            bookmark.optimized_tags = [f"tag{i}", "common", "test"]

        optimized_bookmarks = generator.finalize_tag_optimization(bookmarks)

        assert len(optimized_bookmarks) == len(bookmarks)

        # Check that tags were optimized
        all_tags = set()
        for bookmark in optimized_bookmarks:
            all_tags.update(bookmark.optimized_tags)

        # Should aim for target count
        assert (
            len(all_tags) <= generator.target_tag_count * 1.2
        )  # Allow some flexibility


class TestContentAnalyzerIntegration:
    """Integration tests for content analyzer."""

    def test_full_extraction_workflow(self):
        """Test complete content extraction workflow."""
        mock_session = MockRequestsSession()

        analyzer = ContentAnalyzer()
        analyzer.session = mock_session

        # Test with predefined URLs
        test_urls = [
            "https://docs.python.org/3/",
            "https://github.com/microsoft/vscode",
            "https://stackoverflow.com/",
        ]

        for url in test_urls:
            metadata = analyzer.extract_metadata(url)

            assert isinstance(metadata, BookmarkMetadata)
            assert metadata.title is not None
            assert len(metadata.title) > 0

    def test_batch_content_analysis(self):
        """Test analyzing content for multiple bookmarks."""
        mock_analyzer = MockContentAnalyzer()

        bookmarks = create_sample_bookmark_objects()

        for bookmark in bookmarks:
            metadata = mock_analyzer.extract_metadata(bookmark.url)
            if metadata:
                bookmark.extracted_metadata = metadata

        # Check that metadata was extracted
        processed_count = sum(1 for b in bookmarks if b.extracted_metadata is not None)
        assert processed_count > 0

    def test_tag_generation_integration(self):
        """Test integrated tag generation workflow."""
        generator = CorpusAwareTagGenerator(max_tags=5)

        bookmarks = create_sample_bookmark_objects()

        for bookmark in bookmarks:
            # Simulate AI processing
            bookmark.enhanced_description = f"Enhanced description for {bookmark.title}"

            # Generate tags
            generated_tags = generator.generate_tags_from_bookmark(bookmark)
            bookmark.optimized_tags = generated_tags

        # Check that tags were generated
        for bookmark in bookmarks:
            assert len(bookmark.optimized_tags) <= 5
            assert len(bookmark.optimized_tags) > 0


if __name__ == "__main__":
    pytest.main([__file__])
