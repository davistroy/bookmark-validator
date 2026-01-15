"""
Tests for Enhanced Folder Generator functionality.

Phase 3.3: Tests for EnhancedFolderGenerator, FolderSuggestion, and folder modes.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from bookmark_processor.core.folder_generator import (
    EnhancedFolderGenerator,
    FolderGenerationResult,
    FolderNode,
    FolderSuggestion,
    FolderSuggestionResult,
)
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.content_analyzer import ContentData


@pytest.fixture
def sample_bookmarks():
    """Create sample bookmarks for testing."""
    return [
        Bookmark(
            url="https://github.com/user/repo1",
            title="Python Project 1",
            folder="Development/Python",
            created=datetime.now(),
            tags=["python", "github"],
        ),
        Bookmark(
            url="https://github.com/user/repo2",
            title="Python Project 2",
            folder="Development/Python",
            created=datetime.now(),
            tags=["python", "github"],
        ),
        Bookmark(
            url="https://docs.python.org/tutorial",
            title="Python Tutorial",
            folder="Education/Tutorials",
            created=datetime.now(),
            tags=["python", "tutorial"],
        ),
        Bookmark(
            url="https://arxiv.org/paper/ml",
            title="Machine Learning Research Paper",
            folder="AI & Machine Learning/Research",
            created=datetime.now(),
            tags=["ai", "research"],
        ),
        Bookmark(
            url="https://medium.com/article",
            title="Tech Blog Post",
            folder="News & Media/Blogs",
            created=datetime.now(),
            tags=["blog", "tech"],
        ),
    ]


@pytest.fixture
def content_data_map(sample_bookmarks):
    """Create content data map for bookmarks."""
    return {
        b.url: ContentData(
            url=b.url,
            title=b.title,
            meta_description=f"Description for {b.title}",
            word_count=500,
            content_categories=b.tags[:2] if b.tags else [],
        )
        for b in sample_bookmarks
    }


class TestFolderSuggestion:
    """Test FolderSuggestion dataclass."""

    def test_creation(self):
        """Test creating folder suggestion."""
        suggestion = FolderSuggestion(
            path="Development/Python",
            confidence=0.85,
            reasoning="Domain github.com typically mapped to Development",
            bookmark_count=5,
        )

        assert suggestion.path == "Development/Python"
        assert suggestion.confidence == 0.85
        assert "github.com" in suggestion.reasoning
        assert suggestion.bookmark_count == 5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        suggestion = FolderSuggestion(
            path="AI/Research",
            confidence=0.9,
            reasoning="Content categorized as research",
        )
        data = suggestion.to_dict()

        assert data["path"] == "AI/Research"
        assert data["confidence"] == 0.9
        assert "research" in data["reasoning"]


class TestFolderSuggestionResult:
    """Test FolderSuggestionResult dataclass."""

    def test_creation(self):
        """Test creating folder suggestion result."""
        suggestions = {
            "https://example.com/1": FolderSuggestion(
                path="Dev", confidence=0.8, reasoning="Test"
            ),
            "https://example.com/2": FolderSuggestion(
                path="AI", confidence=0.9, reasoning="Test 2"
            ),
        }

        result = FolderSuggestionResult(
            suggestions=suggestions,
            learned_patterns={"Dev": ["python", "code"]},
            total_bookmarks=2,
            confidence_avg=0.85,
        )

        assert len(result.suggestions) == 2
        assert result.total_bookmarks == 2
        assert result.confidence_avg == 0.85

    def test_to_dict(self):
        """Test conversion to dictionary."""
        suggestions = {
            "https://example.com": FolderSuggestion(
                path="Dev", confidence=0.8, reasoning="Test"
            ),
        }

        result = FolderSuggestionResult(
            suggestions=suggestions,
            learned_patterns={"Dev": ["python"]},
            total_bookmarks=1,
            confidence_avg=0.8,
        )
        data = result.to_dict()

        assert "suggestions" in data
        assert "learned_patterns" in data
        assert data["total_bookmarks"] == 1
        assert data["confidence_avg"] == 0.8

    def test_to_json(self, tmp_path):
        """Test saving to JSON file."""
        suggestions = {
            "https://example.com": FolderSuggestion(
                path="Dev", confidence=0.8, reasoning="Test"
            ),
        }

        result = FolderSuggestionResult(
            suggestions=suggestions,
            learned_patterns={"Dev": ["python"]},
            total_bookmarks=1,
            confidence_avg=0.8,
        )

        json_file = tmp_path / "suggestions.json"
        result.to_json(str(json_file))

        assert json_file.exists()
        with open(json_file) as f:
            data = json.load(f)
        assert "suggestions" in data


class TestEnhancedFolderGenerator:
    """Test EnhancedFolderGenerator class."""

    def test_initialization_default(self):
        """Test default initialization."""
        generator = EnhancedFolderGenerator()

        assert generator.preserve_existing is False
        assert generator.suggest_only is False
        assert generator.learn_from_existing is False
        assert generator.max_depth == 3

    def test_initialization_custom(self):
        """Test custom initialization."""
        generator = EnhancedFolderGenerator(
            max_bookmarks_per_folder=30,
            preserve_existing=True,
            suggest_only=False,
            learn_from_existing=True,
            max_depth=2,
        )

        assert generator.max_bookmarks_per_folder == 30
        assert generator.preserve_existing is True
        assert generator.learn_from_existing is True
        assert generator.max_depth == 2


class TestPreserveFolders:
    """Test folder preservation mode."""

    def test_preserve_existing_folders(self, sample_bookmarks):
        """Test preserving original folder assignments."""
        generator = EnhancedFolderGenerator(
            preserve_existing=True,
            max_depth=3,
        )

        result = generator.generate_folder_structure(sample_bookmarks)

        # All original folders should be preserved
        for bookmark in sample_bookmarks:
            url = bookmark.url
            expected_folder = bookmark.folder if bookmark.folder else "Uncategorized"
            # Folder depth may be limited
            expected_parts = expected_folder.split("/")[:generator.max_depth]
            expected_limited = "/".join(expected_parts)

            assert result.folder_assignments[url] == expected_limited

    def test_preserve_with_depth_limit(self, sample_bookmarks):
        """Test preserving folders with depth limit."""
        generator = EnhancedFolderGenerator(
            preserve_existing=True,
            max_depth=1,  # Only top level
        )

        result = generator.generate_folder_structure(sample_bookmarks)

        # All folders should be limited to 1 level
        for url, path in result.folder_assignments.items():
            assert "/" not in path or path.count("/") == 0

    def test_preserve_uncategorized_fallback(self):
        """Test bookmarks without folder get Uncategorized."""
        bookmarks = [
            Bookmark(
                url="https://example.com",
                title="No Folder",
                created=datetime.now(),
                folder=None,
            )
        ]

        generator = EnhancedFolderGenerator(preserve_existing=True)
        result = generator.generate_folder_structure(bookmarks)

        assert result.folder_assignments["https://example.com"] == "Uncategorized"


class TestSuggestFolders:
    """Test folder suggestion mode."""

    def test_suggest_folders_basic(self, sample_bookmarks, content_data_map):
        """Test generating folder suggestions."""
        generator = EnhancedFolderGenerator(
            suggest_only=True,
            learn_from_existing=True,
        )

        result = generator.suggest_folders(
            sample_bookmarks,
            content_data_map=content_data_map,
        )

        # Should have suggestions for all bookmarks
        assert result.total_bookmarks == len(sample_bookmarks)
        assert len(result.suggestions) == len(sample_bookmarks)

        # Should have learned patterns
        assert len(result.learned_patterns) > 0

        # Check suggestion structure
        for url, suggestion in result.suggestions.items():
            assert suggestion.path is not None
            assert 0.0 <= suggestion.confidence <= 1.0
            assert suggestion.reasoning is not None

    def test_suggest_folders_confidence(self, sample_bookmarks, content_data_map):
        """Test confidence calculation in suggestions."""
        generator = EnhancedFolderGenerator(
            suggest_only=True,
            learn_from_existing=True,
        )

        result = generator.suggest_folders(
            sample_bookmarks,
            content_data_map=content_data_map,
        )

        # Average confidence should be reasonable
        assert 0.0 <= result.confidence_avg <= 1.0

    def test_suggest_folders_with_depth_limit(self, sample_bookmarks, content_data_map):
        """Test suggestions respect depth limit."""
        generator = EnhancedFolderGenerator(
            suggest_only=True,
            max_depth=1,
        )

        result = generator.suggest_folders(
            sample_bookmarks,
            content_data_map=content_data_map,
        )

        # All suggestions should be limited to 1 level
        for url, suggestion in result.suggestions.items():
            assert "/" not in suggestion.path or suggestion.path.count("/") == 0


class TestLearnFolders:
    """Test learning from existing folder structure."""

    def test_learn_from_existing(self, sample_bookmarks):
        """Test learning patterns from existing folders."""
        generator = EnhancedFolderGenerator(
            learn_from_existing=True,
        )

        # Build original folders map
        original_folders = {b.url: b.folder for b in sample_bookmarks if b.folder}

        generator._learn_from_existing_structure(sample_bookmarks, original_folders)

        # Should have learned domain mappings
        assert len(generator.folder_domain_mapping) > 0 or len(generator.learned_patterns) > 0

    def test_learn_domain_mapping(self):
        """Test learning domain to folder mapping."""
        bookmarks = [
            Bookmark(url="https://github.com/a", title="A", folder="Development", created=datetime.now()),
            Bookmark(url="https://github.com/b", title="B", folder="Development", created=datetime.now()),
            Bookmark(url="https://github.com/c", title="C", folder="Development", created=datetime.now()),
        ]

        generator = EnhancedFolderGenerator(learn_from_existing=True)
        original_folders = {b.url: b.folder for b in bookmarks}

        generator._learn_from_existing_structure(bookmarks, original_folders)

        # Should have learned github.com -> Development
        assert "github.com" in generator.folder_domain_mapping
        assert generator.folder_domain_mapping["github.com"] == "Development"

    def test_learn_keyword_patterns(self):
        """Test learning keyword to folder patterns."""
        bookmarks = [
            Bookmark(url="https://example.com/1", title="Python Tutorial One", folder="Tutorials", created=datetime.now()),
            Bookmark(url="https://example.com/2", title="Python Tutorial Two", folder="Tutorials", created=datetime.now()),
            Bookmark(url="https://example.com/3", title="Python Guide Three", folder="Tutorials", created=datetime.now()),
        ]

        generator = EnhancedFolderGenerator(learn_from_existing=True)
        original_folders = {b.url: b.folder for b in bookmarks}

        generator._learn_from_existing_structure(bookmarks, original_folders)

        # Should have learned patterns
        assert "Tutorials" in generator.learned_patterns
        assert "python" in generator.learned_patterns["Tutorials"]


class TestMaxDepth:
    """Test max folder depth functionality."""

    def test_apply_max_depth(self, sample_bookmarks):
        """Test applying max depth to result."""
        generator = EnhancedFolderGenerator(max_depth=2)

        # Create a result with deep paths
        result = FolderGenerationResult(
            root_folder=FolderNode(name="root", path=""),
            folder_assignments={
                "https://example.com": "Level1/Level2/Level3/Level4",
            },
            total_folders=1,
            max_depth=4,
            folder_stats={"Level1/Level2/Level3/Level4": 1},
            processing_time=0.1,
        )

        limited = generator._apply_max_depth(result)

        # Should be limited to 2 levels
        assert limited.folder_assignments["https://example.com"] == "Level1/Level2"
        assert limited.max_depth == 2

    def test_max_depth_in_generation(self, sample_bookmarks, content_data_map):
        """Test max depth during generation."""
        generator = EnhancedFolderGenerator(
            max_depth=1,
            preserve_existing=False,
        )

        result = generator.generate_folder_structure(
            sample_bookmarks,
            content_data_map=content_data_map,
        )

        # All folders should be limited to 1 level
        for url, path in result.folder_assignments.items():
            depth = path.count("/") + 1 if path else 0
            assert depth <= 1


class TestFolderConfidence:
    """Test folder confidence calculation."""

    def test_confidence_original_match(self):
        """Test confidence boost for matching original folder."""
        generator = EnhancedFolderGenerator()

        bookmark = Bookmark(
            url="https://example.com",
            title="Test",
            folder="Development",
            created=datetime.now(),
        )

        confidence = generator._calculate_folder_confidence(
            bookmark, None, "Development", "Development"
        )

        # Should have high confidence for exact match
        assert confidence >= 0.75

    def test_confidence_domain_learned(self):
        """Test confidence boost for learned domain."""
        generator = EnhancedFolderGenerator()
        generator.folder_domain_mapping["github.com"] = "Development"

        bookmark = Bookmark(
            url="https://github.com/test",
            title="Test Repo",
            created=datetime.now(),
        )

        confidence = generator._calculate_folder_confidence(
            bookmark, None, "Development", ""
        )

        # Should be boosted for domain match
        assert confidence >= 0.7


class TestGenerateReasoning:
    """Test reasoning generation for suggestions."""

    def test_reasoning_domain_based(self):
        """Test domain-based reasoning."""
        generator = EnhancedFolderGenerator()
        generator.folder_domain_mapping["github.com"] = "Development"

        bookmark = Bookmark(
            url="https://github.com/test",
            title="Test Repo",
            created=datetime.now(),
        )

        reasoning = generator._generate_reasoning(
            bookmark, None, "Development", "General", ""
        )

        assert "github.com" in reasoning

    def test_reasoning_original_folder(self):
        """Test reasoning mentioning original folder."""
        generator = EnhancedFolderGenerator()

        bookmark = Bookmark(
            url="https://example.com",
            title="Test",
            created=datetime.now(),
        )

        reasoning = generator._generate_reasoning(
            bookmark, None, "Development", "General", "Development/Old"
        )

        assert "Development" in reasoning

    def test_reasoning_fallback(self):
        """Test fallback reasoning when no specific match."""
        generator = EnhancedFolderGenerator()

        bookmark = Bookmark(
            url="https://random-site.com",
            title="Random",
            created=datetime.now(),
        )

        reasoning = generator._generate_reasoning(
            bookmark, None, "Uncategorized", "General", ""
        )

        assert "content analysis" in reasoning.lower()


class TestEnhancedFolderGeneratorIntegration:
    """Integration tests for EnhancedFolderGenerator."""

    def test_full_workflow_preserve(self, sample_bookmarks, content_data_map):
        """Test complete workflow with preserve mode."""
        generator = EnhancedFolderGenerator(
            preserve_existing=True,
            max_depth=2,
        )

        result = generator.generate_folder_structure(
            sample_bookmarks,
            content_data_map=content_data_map,
        )

        assert result.total_folders > 0
        assert all(b.url in result.folder_assignments for b in sample_bookmarks)

    def test_full_workflow_learn_and_suggest(self, sample_bookmarks, content_data_map):
        """Test learning and suggesting workflow."""
        generator = EnhancedFolderGenerator(
            learn_from_existing=True,
            suggest_only=True,
        )

        result = generator.suggest_folders(
            sample_bookmarks,
            content_data_map=content_data_map,
        )

        assert result.total_bookmarks == len(sample_bookmarks)
        assert result.confidence_avg > 0.0

    def test_output_to_json(self, sample_bookmarks, content_data_map, tmp_path):
        """Test outputting suggestions to JSON."""
        generator = EnhancedFolderGenerator(
            learn_from_existing=True,
            suggest_only=True,
        )

        result = generator.suggest_folders(
            sample_bookmarks,
            content_data_map=content_data_map,
        )

        json_file = tmp_path / "suggestions.json"
        result.to_json(str(json_file))

        assert json_file.exists()
        with open(json_file) as f:
            data = json.load(f)

        assert "suggestions" in data
        assert len(data["suggestions"]) == len(sample_bookmarks)
