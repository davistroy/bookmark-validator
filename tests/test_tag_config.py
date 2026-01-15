"""
Tests for Tag Configuration and Enhanced Tag Generator.

Phase 3.2: Tests for TagConfig, TagNormalizer, and EnhancedTagGenerator.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from bookmark_processor.core.tag_config import (
    TagConfig,
    TagNormalizer,
    TagWithConfidence,
)
from bookmark_processor.core.tag_generator import EnhancedTagGenerator
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.content_analyzer import ContentData


@pytest.fixture
def sample_tag_config():
    """Create a sample tag configuration."""
    return TagConfig(
        protected_tags={"important", "to-read", "favorite"},
        synonyms={
            "artificial-intelligence": "ai",
            "machine-learning": "ml",
            "js": "javascript",
        },
        hierarchy={
            "ai": "technology/ai",
            "python": "technology/programming/python",
        },
        target_unique_tags=100,
        max_tags_per_bookmark=5,
    )


@pytest.fixture
def sample_bookmark():
    """Create a sample bookmark for testing."""
    return Bookmark(
        url="https://example.com/python-tutorial",
        title="Python Tutorial for Beginners",
        created=datetime.now(),
        tags=["python", "programming", "tutorial"],
    )


@pytest.fixture
def sample_content():
    """Create sample content data."""
    return ContentData(
        url="https://example.com/python-tutorial",
        title="Python Tutorial for Beginners",
        meta_description="Learn Python programming step by step",
        word_count=500,
        content_categories=["tutorial", "programming"],
        headings=["Introduction", "Getting Started", "Python Basics"],
    )


class TestTagConfig:
    """Test TagConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TagConfig()

        assert "important" in config.protected_tags
        assert "to-read" in config.protected_tags
        assert config.target_unique_tags == 150
        assert config.max_tags_per_bookmark == 5
        assert config.min_tag_frequency == 2
        assert "artificial-intelligence" in config.synonyms

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TagConfig(
            protected_tags={"custom-tag"},
            synonyms={"test": "testing"},
            hierarchy={"web": "technology/web"},
            target_unique_tags=200,
        )

        assert "custom-tag" in config.protected_tags
        assert config.synonyms["test"] == "testing"
        assert config.hierarchy["web"] == "technology/web"
        assert config.target_unique_tags == 200

    def test_to_dict(self):
        """Test config serialization."""
        config = TagConfig(
            protected_tags={"test"},
            target_unique_tags=100,
        )
        data = config.to_dict()

        assert "test" in data["protected_tags"]
        assert data["target_unique_tags"] == 100
        assert "synonyms" in data
        assert "hierarchy" in data

    def test_from_dict(self):
        """Test config deserialization."""
        data = {
            "protected_tags": ["important", "custom"],
            "synonyms": {"ai": "artificial-intelligence"},
            "hierarchy": {"web": "tech/web"},
            "target_unique_tags": 200,
        }
        config = TagConfig.from_dict(data)

        assert "important" in config.protected_tags
        assert "custom" in config.protected_tags
        assert config.synonyms["ai"] == "artificial-intelligence"
        assert config.target_unique_tags == 200

    def test_from_toml_file(self, tmp_path):
        """Test loading config from TOML file."""
        toml_content = '''
[tags]
protected_tags = ["important", "reference", "custom-tag"]
target_unique_tags = 180
max_tags_per_bookmark = 6

[tags.synonyms]
"machine-learning" = "ml"
"artificial-intelligence" = "ai"

[tags.hierarchy]
"python" = "programming/python"
'''
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        try:
            config = TagConfig.from_toml_file(str(toml_file))

            assert "important" in config.protected_tags
            assert "custom-tag" in config.protected_tags
            assert config.synonyms["machine-learning"] == "ml"
            assert config.hierarchy["python"] == "programming/python"
            assert config.target_unique_tags == 180
        except ValueError:
            # TOML parsing not available, skip test
            pytest.skip("TOML parsing not available")

    def test_save_to_toml(self, tmp_path):
        """Test saving config to TOML file."""
        config = TagConfig(
            protected_tags={"important", "test"},
            synonyms={"ai": "artificial-intelligence"},
            hierarchy={"web": "tech/web"},
        )

        toml_file = tmp_path / "output.toml"
        config.save_to_toml(str(toml_file))

        assert toml_file.exists()
        content = toml_file.read_text()
        assert "[tags]" in content
        assert "protected_tags" in content


class TestTagNormalizer:
    """Test TagNormalizer class."""

    def test_normalizer_initialization(self, sample_tag_config):
        """Test normalizer initialization."""
        normalizer = TagNormalizer(sample_tag_config)

        assert normalizer.config == sample_tag_config

    def test_normalize_tag_basic(self):
        """Test basic tag normalization."""
        # Use a config without synonyms to test basic cleaning
        config = TagConfig(synonyms={})
        normalizer = TagNormalizer(config)

        # Basic cleaning
        assert normalizer.normalize_tag("  Python  ") == "python"
        assert normalizer.normalize_tag("Machine_Learning") == "machine-learning"

    def test_normalize_tag_synonyms(self, sample_tag_config):
        """Test synonym resolution."""
        normalizer = TagNormalizer(sample_tag_config)

        assert normalizer.normalize_tag("artificial-intelligence") == "ai"
        assert normalizer.normalize_tag("machine-learning") == "ml"
        assert normalizer.normalize_tag("js") == "javascript"

    def test_normalize_tag_protected(self, sample_tag_config):
        """Test protected tags are preserved."""
        normalizer = TagNormalizer(sample_tag_config)

        # Protected tags should be preserved as-is (lowercased)
        assert normalizer.normalize_tag("important") == "important"
        assert normalizer.normalize_tag("to-read") == "to-read"

    def test_apply_hierarchy(self, sample_tag_config):
        """Test hierarchy application."""
        normalizer = TagNormalizer(sample_tag_config)

        assert normalizer.apply_hierarchy("ai") == "technology/ai"
        assert normalizer.apply_hierarchy("python") == "technology/programming/python"

    def test_apply_hierarchy_no_match(self, sample_tag_config):
        """Test hierarchy with no matching rule."""
        normalizer = TagNormalizer(sample_tag_config)

        # Tags without hierarchy rules should be normalized only
        result = normalizer.apply_hierarchy("random-tag")
        assert result == "random-tag"

    def test_is_protected(self, sample_tag_config):
        """Test protected tag detection."""
        normalizer = TagNormalizer(sample_tag_config)

        assert normalizer.is_protected("important") is True
        assert normalizer.is_protected("to-read") is True
        assert normalizer.is_protected("favorite") is True
        assert normalizer.is_protected("random") is False

    def test_normalize_tags_list(self, sample_tag_config):
        """Test normalizing list of tags."""
        normalizer = TagNormalizer(sample_tag_config)

        tags = ["Python", "artificial-intelligence", "js", "python"]
        result = normalizer.normalize_tags(tags)

        # Should deduplicate and normalize
        assert "python" in result
        assert "ai" in result
        assert "javascript" in result
        assert len([t for t in result if t == "python"]) == 1  # No duplicates

    def test_normalize_tags_with_confidence(self, sample_tag_config):
        """Test normalizing tags with confidence scores."""
        normalizer = TagNormalizer(sample_tag_config)

        tags = [("Python", 0.9), ("artificial-intelligence", 0.8), ("Python", 0.7)]
        result = normalizer.normalize_tags_with_confidence(tags)

        # Should keep highest confidence for duplicates
        python_tag = next((t for t in result if t.tag == "python"), None)
        assert python_tag is not None
        assert python_tag.confidence == 0.9

        ai_tag = next((t for t in result if t.tag == "ai"), None)
        assert ai_tag is not None
        assert ai_tag.confidence == 0.8

    def test_get_category_for_tag(self, sample_tag_config):
        """Test category detection for tags."""
        normalizer = TagNormalizer(sample_tag_config)

        # Add some category mappings
        sample_tag_config.category_mappings = {
            "technology": ["programming", "python", "javascript"],
            "design": ["ui", "ux", "graphic"],
        }
        normalizer.config = sample_tag_config

        assert normalizer.get_category_for_tag("python") == "technology"
        assert normalizer.get_category_for_tag("ui") == "design"
        assert normalizer.get_category_for_tag("random") is None


class TestTagWithConfidence:
    """Test TagWithConfidence dataclass."""

    def test_creation(self):
        """Test creating TagWithConfidence."""
        tag = TagWithConfidence(
            tag="python",
            confidence=0.9,
            source="extracted",
        )

        assert tag.tag == "python"
        assert tag.confidence == 0.9
        assert tag.source == "extracted"

    def test_to_tuple(self):
        """Test conversion to tuple."""
        tag = TagWithConfidence(tag="python", confidence=0.9)
        assert tag.to_tuple() == ("python", 0.9)

    def test_to_dict(self):
        """Test conversion to dict."""
        tag = TagWithConfidence(tag="python", confidence=0.9, source="ai_generated")
        data = tag.to_dict()

        assert data["tag"] == "python"
        assert data["confidence"] == 0.9
        assert data["source"] == "ai_generated"


class TestEnhancedTagGenerator:
    """Test EnhancedTagGenerator class."""

    def test_initialization_default(self):
        """Test default initialization."""
        generator = EnhancedTagGenerator()

        assert generator.target_tag_count == 150
        assert generator.max_tags_per_bookmark == 5
        assert generator.tag_config is not None

    def test_initialization_with_config(self, sample_tag_config):
        """Test initialization with custom config."""
        generator = EnhancedTagGenerator(config=sample_tag_config)

        assert generator.target_tag_count == 100
        assert generator.max_tags_per_bookmark == 5
        assert generator.tag_config == sample_tag_config

    def test_initialization_with_config_file(self, tmp_path):
        """Test initialization with config file."""
        toml_content = '''
[tags]
protected_tags = ["custom"]
target_unique_tags = 120
'''
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        try:
            generator = EnhancedTagGenerator(config_file=str(toml_file))
            assert generator.target_tag_count == 120
        except ValueError:
            pytest.skip("TOML parsing not available")

    def test_normalize_tag(self, sample_tag_config):
        """Test tag normalization."""
        generator = EnhancedTagGenerator(config=sample_tag_config)

        assert generator.normalize_tag("artificial-intelligence") == "ai"
        assert generator.normalize_tag("  Python  ") == "python"

    def test_apply_hierarchy(self, sample_tag_config):
        """Test hierarchy application."""
        generator = EnhancedTagGenerator(config=sample_tag_config)

        assert generator.apply_hierarchy("ai") == "technology/ai"

    def test_is_protected(self, sample_tag_config):
        """Test protected tag detection."""
        generator = EnhancedTagGenerator(config=sample_tag_config)

        assert generator.is_protected("important") is True
        assert generator.is_protected("random") is False

    def test_get_protected_tags(self, sample_tag_config):
        """Test getting protected tags set."""
        generator = EnhancedTagGenerator(config=sample_tag_config)

        protected = generator.get_protected_tags()
        assert "important" in protected
        assert "to-read" in protected
        assert "favorite" in protected

    def test_get_synonyms(self, sample_tag_config):
        """Test getting synonyms."""
        generator = EnhancedTagGenerator(config=sample_tag_config)

        synonyms = generator.get_synonyms()
        assert synonyms["artificial-intelligence"] == "ai"
        assert synonyms["machine-learning"] == "ml"

    def test_get_hierarchy(self, sample_tag_config):
        """Test getting hierarchy."""
        generator = EnhancedTagGenerator(config=sample_tag_config)

        hierarchy = generator.get_hierarchy()
        assert hierarchy["ai"] == "technology/ai"

    def test_generate_with_confidence(self, sample_bookmark, sample_content, sample_tag_config):
        """Test generating tags with confidence scores."""
        generator = EnhancedTagGenerator(config=sample_tag_config)

        # Build corpus first with the bookmark
        bookmarks = [sample_bookmark]
        content_map = {sample_bookmark.url: sample_content}

        result = generator.generate_with_confidence(
            bookmarks, content_data_map=content_map
        )

        assert sample_bookmark.url in result
        tags_with_conf = result[sample_bookmark.url]
        assert len(tags_with_conf) <= generator.max_tags_per_bookmark

        # Check that results are (tag, confidence) tuples
        for tag, conf in tags_with_conf:
            assert isinstance(tag, str)
            assert 0.0 <= conf <= 1.0

    def test_calculate_tag_confidence_protected(self, sample_bookmark, sample_tag_config):
        """Test confidence calculation for protected tags."""
        generator = EnhancedTagGenerator(config=sample_tag_config)

        confidence = generator._calculate_tag_confidence(
            sample_bookmark, "important", {}
        )

        # Protected tags should have high confidence
        assert confidence >= 0.95

    def test_calculate_tag_confidence_in_title(self, sample_tag_config):
        """Test confidence boost for tags in title."""
        generator = EnhancedTagGenerator(config=sample_tag_config)

        bookmark = Bookmark(
            url="https://example.com",
            title="Python Programming Tutorial",
            created=datetime.now(),
        )

        confidence = generator._calculate_tag_confidence(
            bookmark, "python", {}
        )

        # Should be boosted for being in title
        assert confidence >= 0.7

    def test_calculate_tag_confidence_in_existing_tags(self, sample_bookmark, sample_tag_config):
        """Test confidence boost for existing tags."""
        generator = EnhancedTagGenerator(config=sample_tag_config)

        # "python" is in bookmark's existing tags
        confidence = generator._calculate_tag_confidence(
            sample_bookmark, "python", {}
        )

        # Should be boosted for being in existing tags
        assert confidence >= 0.75

    def test_generate_corpus_tags_with_hierarchy(self, sample_bookmark, sample_content, sample_tag_config):
        """Test generating hierarchical tags."""
        generator = EnhancedTagGenerator(config=sample_tag_config)

        bookmarks = [sample_bookmark]
        content_map = {sample_bookmark.url: sample_content}

        result = generator.generate_corpus_tags_with_hierarchy(
            bookmarks, content_data_map=content_map, apply_hierarchy=True
        )

        # Should have assignments
        assert sample_bookmark.url in result.tag_assignments

    def test_clean_tags_with_normalization(self, sample_tag_config):
        """Test tag cleaning with normalization."""
        generator = EnhancedTagGenerator(config=sample_tag_config)

        tags = {"artificial-intelligence", "Python", "machine-learning"}
        cleaned = generator._clean_tags(tags)

        # Should be normalized
        assert "ai" in cleaned or "python" in cleaned or "ml" in cleaned


class TestEnhancedTagGeneratorIntegration:
    """Integration tests for EnhancedTagGenerator."""

    def test_full_workflow(self):
        """Test complete tag generation workflow."""
        # Create bookmarks
        bookmarks = [
            Bookmark(
                url=f"https://example.com/{i}",
                title=f"Python Tutorial Part {i}",
                created=datetime.now(),
                tags=["python", "tutorial"],
            )
            for i in range(5)
        ]

        # Create content data
        content_map = {
            b.url: ContentData(
                url=b.url,
                title=b.title,
                meta_description="Python programming",
                word_count=500,
                content_categories=["programming", "tutorial"],
            )
            for b in bookmarks
        }

        # Create generator with custom config
        config = TagConfig(
            protected_tags={"important"},
            synonyms={"artificial-intelligence": "ai"},
            hierarchy={"python": "programming/python"},
            target_unique_tags=50,
        )
        generator = EnhancedTagGenerator(config=config)

        # Generate tags
        result = generator.generate_corpus_tags(bookmarks, content_map)

        assert result.total_unique_tags > 0
        assert all(url in result.tag_assignments for url in [b.url for b in bookmarks])

    def test_with_multiple_synonyms(self):
        """Test handling multiple synonyms."""
        config = TagConfig(
            synonyms={
                "artificial-intelligence": "ai",
                "machine-learning": "ml",
                "deep-learning": "dl",
                "javascript": "js",
            },
        )
        generator = EnhancedTagGenerator(config=config)

        # Test each synonym
        assert generator.normalize_tag("artificial-intelligence") == "ai"
        assert generator.normalize_tag("machine-learning") == "ml"
        assert generator.normalize_tag("deep-learning") == "dl"
        assert generator.normalize_tag("javascript") == "js"
