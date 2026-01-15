"""
Tests for Interactive Bookmark Processor

Tests the interactive processing mode including:
- Proposed changes generation
- User action handling
- Session statistics
- Undo functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.interactive_processor import (
    InteractiveProcessor,
    InteractiveAction,
    ProposedChanges,
    ProcessedBookmark,
    InteractiveSessionStats,
)
from bookmark_processor.core.ai_processor import AIProcessingResult


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_bookmark():
    """Create a sample bookmark for testing."""
    return Bookmark(
        id="test1",
        title="Test Bookmark",
        url="https://example.com/article",
        folder="Technology",
        tags=["tech", "python"],
        note="A test article about technology",
    )


@pytest.fixture
def sample_bookmarks():
    """Create multiple sample bookmarks."""
    return [
        Bookmark(
            id="1",
            title="Python Tutorial",
            url="https://python.org/tutorial",
            folder="Programming",
            tags=["python", "tutorial"],
            note="Official Python tutorial",
        ),
        Bookmark(
            id="2",
            title="AI Research Paper",
            url="https://arxiv.org/ai-paper",
            folder="Research",
            tags=["ai", "research"],
            note="Recent AI research",
        ),
        Bookmark(
            id="3",
            title="Web Development Guide",
            url="https://webdev.com/guide",
            folder="Development",
            tags=["web", "frontend"],
            note="Comprehensive web dev guide",
        ),
    ]


@pytest.fixture
def sample_ai_result():
    """Create a sample AI processing result."""
    return AIProcessingResult(
        original_url="https://example.com/article",
        enhanced_description="An in-depth article covering modern technology trends",
        processing_method="ai_with_context",
        processing_time=1.5,
        model_used="test-model",
        confidence_score=0.85,
    )


@pytest.fixture
def sample_proposed_changes():
    """Create sample proposed changes."""
    return ProposedChanges(
        url="https://example.com/article",
        original_description="A test article",
        proposed_description="Enhanced: A comprehensive technology article",
        description_confidence=0.85,
        description_method="ai_with_context",
        original_tags=["tech"],
        proposed_tags=["tech", "python", "programming"],
        tags_confidence=0.8,
        original_folder="Unsorted",
        proposed_folder="Technology",
        folder_confidence=0.75,
    )


@pytest.fixture
def interactive_processor():
    """Create an interactive processor instance."""
    return InteractiveProcessor(
        pipeline=None,
        confirm_threshold=0.0,
        show_diff=True,
        compact_mode=False,
    )


# ============================================================================
# ProposedChanges Tests
# ============================================================================


class TestProposedChanges:
    """Tests for ProposedChanges dataclass."""

    def test_proposed_changes_creation(self, sample_proposed_changes):
        """Test creating proposed changes."""
        assert sample_proposed_changes.url == "https://example.com/article"
        assert sample_proposed_changes.description_confidence == 0.85
        assert len(sample_proposed_changes.proposed_tags) == 3

    def test_overall_confidence_calculation(self):
        """Test overall confidence is calculated correctly."""
        changes = ProposedChanges(
            url="https://test.com",
            original_description="Original",
            proposed_description="Proposed",
            description_confidence=0.8,
            description_method="ai",
            original_tags=["tag1"],
            proposed_tags=["tag1", "tag2"],
            tags_confidence=0.7,
            original_folder="Old",
            proposed_folder="New",
            folder_confidence=0.6,
        )
        # Weighted: 0.8*0.4 + 0.7*0.3 + 0.6*0.3 = 0.32 + 0.21 + 0.18 = 0.71
        assert 0.70 <= changes.overall_confidence <= 0.72

    def test_has_description_change(self, sample_proposed_changes):
        """Test description change detection."""
        assert sample_proposed_changes.has_description_change() is True

        no_change = ProposedChanges(
            url="https://test.com",
            original_description="Same",
            proposed_description="Same",
            description_confidence=1.0,
            description_method="unchanged",
            original_tags=[],
            proposed_tags=[],
            tags_confidence=1.0,
            original_folder="",
            proposed_folder="",
            folder_confidence=1.0,
        )
        assert no_change.has_description_change() is False

    def test_has_tags_change(self, sample_proposed_changes):
        """Test tags change detection."""
        assert sample_proposed_changes.has_tags_change() is True

    def test_has_folder_change(self, sample_proposed_changes):
        """Test folder change detection."""
        assert sample_proposed_changes.has_folder_change() is True

    def test_has_any_change(self, sample_proposed_changes):
        """Test any change detection."""
        assert sample_proposed_changes.has_any_change() is True

        no_changes = ProposedChanges(
            url="https://test.com",
            original_description="Same",
            proposed_description="Same",
            description_confidence=1.0,
            description_method="unchanged",
            original_tags=["tag1"],
            proposed_tags=["tag1"],
            tags_confidence=1.0,
            original_folder="Folder",
            proposed_folder="Folder",
            folder_confidence=1.0,
        )
        assert no_changes.has_any_change() is False

    def test_to_dict(self, sample_proposed_changes):
        """Test serialization to dictionary."""
        result = sample_proposed_changes.to_dict()
        assert isinstance(result, dict)
        assert result["url"] == sample_proposed_changes.url
        assert result["description_confidence"] == sample_proposed_changes.description_confidence
        assert "overall_confidence" in result


# ============================================================================
# InteractiveSessionStats Tests
# ============================================================================


class TestInteractiveSessionStats:
    """Tests for InteractiveSessionStats."""

    def test_initial_state(self):
        """Test initial statistics state."""
        stats = InteractiveSessionStats()
        assert stats.total_bookmarks == 0
        assert stats.processed_count == 0
        assert stats.accepted_all == 0
        assert stats.skipped == 0

    def test_progress_percentage(self):
        """Test progress percentage calculation."""
        stats = InteractiveSessionStats(total_bookmarks=100, processed_count=50)
        assert stats.get_progress_percentage() == 50.0

    def test_progress_percentage_zero_total(self):
        """Test progress percentage with zero total."""
        stats = InteractiveSessionStats(total_bookmarks=0, processed_count=0)
        assert stats.get_progress_percentage() == 0.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        stats = InteractiveSessionStats(
            total_bookmarks=100,
            processed_count=50,
            accepted_all=30,
            skipped=10,
        )
        result = stats.to_dict()
        assert result["total_bookmarks"] == 100
        assert result["processed_count"] == 50
        assert result["accepted_all"] == 30
        assert result["skipped"] == 10


# ============================================================================
# InteractiveProcessor Tests
# ============================================================================


class TestInteractiveProcessor:
    """Tests for InteractiveProcessor."""

    def test_processor_initialization(self, interactive_processor):
        """Test processor initialization."""
        assert interactive_processor.confirm_threshold == 0.0
        assert interactive_processor.show_diff is True
        assert interactive_processor.compact_mode is False

    def test_processor_with_threshold(self):
        """Test processor with custom threshold."""
        processor = InteractiveProcessor(confirm_threshold=0.7)
        assert processor.confirm_threshold == 0.7

    def test_propose_changes(self, interactive_processor, sample_bookmark, sample_ai_result):
        """Test proposing changes for a bookmark."""
        changes = interactive_processor.propose_changes(
            bookmark=sample_bookmark,
            ai_result=sample_ai_result,
            proposed_tags=["tech", "ai", "programming"],
            proposed_folder="Technology",
        )

        assert isinstance(changes, ProposedChanges)
        assert changes.url == sample_bookmark.url
        assert changes.proposed_description == sample_ai_result.enhanced_description
        assert changes.description_confidence == sample_ai_result.confidence_score
        assert "ai" in changes.proposed_tags

    def test_propose_changes_without_ai(self, interactive_processor, sample_bookmark):
        """Test proposing changes without AI result."""
        changes = interactive_processor.propose_changes(
            bookmark=sample_bookmark,
            ai_result=None,
        )

        assert changes.proposed_description == sample_bookmark.get_effective_description()
        assert changes.description_confidence == 1.0
        assert changes.description_method == "unchanged"

    def test_set_callbacks(self, interactive_processor):
        """Test setting callbacks."""
        progress_callback = Mock()
        save_callback = Mock()

        interactive_processor.set_on_progress(progress_callback)
        interactive_processor.set_on_save(save_callback)

        assert interactive_processor._on_progress is progress_callback
        assert interactive_processor._on_save is save_callback


# ============================================================================
# Action Handling Tests
# ============================================================================


class TestActionHandling:
    """Tests for action handling."""

    def test_apply_accept_all(self, interactive_processor, sample_bookmark, sample_proposed_changes):
        """Test accepting all changes."""
        # Use internal method to apply action
        result = interactive_processor._apply_action(
            bookmark=sample_bookmark,
            changes=sample_proposed_changes,
            action=InteractiveAction.ACCEPT_ALL,
        )

        assert isinstance(result, ProcessedBookmark)
        assert result.action_taken == InteractiveAction.ACCEPT_ALL
        assert "description" in result.changes_applied
        assert "tags" in result.changes_applied
        assert "folder" in result.changes_applied
        assert result.was_modified is True

    def test_apply_description_only(self, interactive_processor, sample_bookmark, sample_proposed_changes):
        """Test accepting description only."""
        result = interactive_processor._apply_action(
            bookmark=sample_bookmark,
            changes=sample_proposed_changes,
            action=InteractiveAction.DESCRIPTION_ONLY,
        )

        assert "description" in result.changes_applied
        assert "tags" not in result.changes_applied
        assert "folder" not in result.changes_applied

    def test_apply_tags_only(self, interactive_processor, sample_bookmark, sample_proposed_changes):
        """Test accepting tags only."""
        result = interactive_processor._apply_action(
            bookmark=sample_bookmark,
            changes=sample_proposed_changes,
            action=InteractiveAction.TAGS_ONLY,
        )

        assert "tags" in result.changes_applied
        assert "description" not in result.changes_applied

    def test_apply_folder_only(self, interactive_processor, sample_bookmark, sample_proposed_changes):
        """Test accepting folder only."""
        result = interactive_processor._apply_action(
            bookmark=sample_bookmark,
            changes=sample_proposed_changes,
            action=InteractiveAction.FOLDER_ONLY,
        )

        assert "folder" in result.changes_applied
        assert "description" not in result.changes_applied
        assert "tags" not in result.changes_applied

    def test_apply_skip(self, interactive_processor, sample_bookmark, sample_proposed_changes):
        """Test skipping a bookmark."""
        result = interactive_processor._apply_action(
            bookmark=sample_bookmark,
            changes=sample_proposed_changes,
            action=InteractiveAction.SKIP,
        )

        assert result.action_taken == InteractiveAction.SKIP
        assert len(result.changes_applied) == 0
        assert result.was_modified is False


# ============================================================================
# State Management Tests
# ============================================================================


class TestStateManagement:
    """Tests for state capture and restore."""

    def test_capture_state(self, interactive_processor, sample_bookmark):
        """Test capturing bookmark state."""
        state = interactive_processor._capture_state(sample_bookmark)

        assert isinstance(state, dict)
        assert "note" in state
        assert "tags" in state
        assert "folder" in state

    def test_restore_state(self, interactive_processor, sample_bookmark):
        """Test restoring bookmark state."""
        # Capture original state
        original_state = interactive_processor._capture_state(sample_bookmark)

        # Modify bookmark
        sample_bookmark.folder = "Modified"
        sample_bookmark.tags = ["modified"]

        # Restore
        interactive_processor._restore_state(sample_bookmark, original_state)

        assert sample_bookmark.folder == original_state["folder"]
        assert sample_bookmark.tags == original_state["tags"]


# ============================================================================
# Statistics Update Tests
# ============================================================================


class TestStatisticsUpdate:
    """Tests for statistics updates."""

    def test_update_stats_accept_all(self, interactive_processor):
        """Test stats update for accept all."""
        interactive_processor.stats = InteractiveSessionStats()
        interactive_processor._update_stats(InteractiveAction.ACCEPT_ALL)
        assert interactive_processor.stats.accepted_all == 1

    def test_update_stats_skip(self, interactive_processor):
        """Test stats update for skip."""
        interactive_processor.stats = InteractiveSessionStats()
        interactive_processor._update_stats(InteractiveAction.SKIP)
        assert interactive_processor.stats.skipped == 1

    def test_update_stats_description_only(self, interactive_processor):
        """Test stats update for description only."""
        interactive_processor.stats = InteractiveSessionStats()
        interactive_processor._update_stats(InteractiveAction.DESCRIPTION_ONLY)
        assert interactive_processor.stats.description_only == 1


# ============================================================================
# Auto-Accept Tests
# ============================================================================


class TestAutoAccept:
    """Tests for auto-accept functionality."""

    def test_auto_accept_above_threshold(self, sample_bookmark, sample_proposed_changes):
        """Test auto-accept when confidence is above threshold."""
        processor = InteractiveProcessor(confirm_threshold=0.5)

        # Sample changes have confidence > 0.5
        result = processor._auto_accept(sample_bookmark, sample_proposed_changes)

        assert isinstance(result, ProcessedBookmark)
        assert result.action_taken == InteractiveAction.ACCEPT_ALL
        assert result.was_modified is True

    def test_auto_accept_preserves_original_state(self, sample_bookmark, sample_proposed_changes):
        """Test that auto-accept preserves original state for undo."""
        processor = InteractiveProcessor(confirm_threshold=0.5)
        result = processor._auto_accept(sample_bookmark, sample_proposed_changes)

        assert "note" in result.original_state
        assert "tags" in result.original_state
        assert "folder" in result.original_state


# ============================================================================
# Undo Tests
# ============================================================================


class TestUndo:
    """Tests for undo functionality."""

    def test_undo_restores_state(self, interactive_processor, sample_bookmark, sample_proposed_changes):
        """Test undo restores bookmark to previous state."""
        # Apply changes
        result = interactive_processor._apply_action(
            bookmark=sample_bookmark,
            changes=sample_proposed_changes,
            action=InteractiveAction.ACCEPT_ALL,
        )

        # Add to history
        interactive_processor.history.append(result)
        interactive_processor.stats = InteractiveSessionStats(
            total_bookmarks=1,
            processed_count=1,
            accepted_all=1,
        )

        # Verify changes applied
        assert sample_bookmark.enhanced_description == sample_proposed_changes.proposed_description

        # Undo
        interactive_processor._undo_last()

        # Verify stats updated
        assert interactive_processor.stats.accepted_all == 0
        assert interactive_processor.stats.processed_count == 0

    def test_undo_empty_history(self, interactive_processor):
        """Test undo with empty history does nothing."""
        interactive_processor.history = []
        # Should not raise an error
        interactive_processor._undo_last()


# ============================================================================
# ProcessedBookmark Tests
# ============================================================================


class TestProcessedBookmark:
    """Tests for ProcessedBookmark dataclass."""

    def test_processed_bookmark_creation(self, sample_bookmark):
        """Test creating a processed bookmark."""
        result = ProcessedBookmark(
            bookmark=sample_bookmark,
            changes_applied=["description", "tags"],
            action_taken=InteractiveAction.ACCEPT_ALL,
            original_state={"note": "", "tags": []},
            was_modified=True,
        )

        assert result.bookmark == sample_bookmark
        assert "description" in result.changes_applied
        assert result.was_modified is True

    def test_to_dict(self, sample_bookmark):
        """Test serialization to dictionary."""
        result = ProcessedBookmark(
            bookmark=sample_bookmark,
            changes_applied=["description"],
            action_taken=InteractiveAction.ACCEPT_ALL,
            original_state={},
            was_modified=True,
        )

        data = result.to_dict()
        assert data["url"] == sample_bookmark.url
        assert data["action_taken"] == "a"
        assert data["was_modified"] is True


# ============================================================================
# Interactive Action Enum Tests
# ============================================================================


class TestInteractiveAction:
    """Tests for InteractiveAction enum."""

    def test_action_values(self):
        """Test action enum values."""
        assert InteractiveAction.ACCEPT_ALL.value == "a"
        assert InteractiveAction.DESCRIPTION_ONLY.value == "d"
        assert InteractiveAction.TAGS_ONLY.value == "t"
        assert InteractiveAction.FOLDER_ONLY.value == "f"
        assert InteractiveAction.SKIP.value == "s"
        assert InteractiveAction.QUIT.value == "q"

    def test_all_actions_defined(self):
        """Test all expected actions are defined."""
        actions = list(InteractiveAction)
        assert len(actions) >= 10  # At least 10 actions
        assert InteractiveAction.HELP in actions
        assert InteractiveAction.UNDO in actions


# ============================================================================
# Integration Tests
# ============================================================================


class TestInteractiveProcessorIntegration:
    """Integration tests for interactive processor."""

    def test_process_empty_bookmarks(self, interactive_processor):
        """Test processing empty bookmark list."""
        results = interactive_processor.process_interactive([])
        assert results == []

    @patch.object(InteractiveProcessor, '_prompt_action')
    def test_process_with_skip_all(self, mock_prompt, interactive_processor, sample_bookmarks):
        """Test processing with skip action for all."""
        mock_prompt.return_value = InteractiveAction.SKIP

        # Build proposed changes dict to pass to process_interactive
        proposed_changes = {}
        for bookmark in sample_bookmarks:
            changes = ProposedChanges(
                url=bookmark.url,
                original_description="original",
                proposed_description="new description",  # Different from original
                description_confidence=0.8,
                description_method="ai",
                original_tags=bookmark.tags,
                proposed_tags=bookmark.tags + ["new_tag"],  # Force a tag change
                tags_confidence=0.8,
                original_folder=bookmark.folder,
                proposed_folder="NewFolder",  # Force a folder change
                folder_confidence=0.8,
            )
            proposed_changes[bookmark.url] = changes

        results = interactive_processor.process_interactive(
            sample_bookmarks, proposed_changes=proposed_changes
        )

        assert len(results) == len(sample_bookmarks)
        for result in results:
            assert result.action_taken == InteractiveAction.SKIP
            assert result.was_modified is False

    @patch.object(InteractiveProcessor, '_prompt_action')
    def test_process_with_quit(self, mock_prompt, interactive_processor, sample_bookmarks):
        """Test processing with quit action - verifies QUIT breaks the loop."""
        # First call returns QUIT, which should stop processing
        mock_prompt.return_value = InteractiveAction.QUIT

        # Build proposed changes dict to pass to process_interactive
        proposed_changes = {}
        for bookmark in sample_bookmarks:
            changes = ProposedChanges(
                url=bookmark.url,
                original_description="original",
                proposed_description="new description",  # Different from original
                description_confidence=0.8,
                description_method="ai",
                original_tags=bookmark.tags,
                proposed_tags=bookmark.tags + ["new_tag"],  # Force a tag change
                tags_confidence=0.8,
                original_folder=bookmark.folder,
                proposed_folder="NewFolder",  # Force a folder change
                folder_confidence=0.8,
            )
            proposed_changes[bookmark.url] = changes

        # Pass proposed_changes as parameter (not setting on instance)
        results = interactive_processor.process_interactive(
            sample_bookmarks, proposed_changes=proposed_changes
        )

        # Quit should stop processing - no bookmarks should be marked as modified
        # since QUIT breaks out of the loop before applying changes
        assert len(results) == 0  # QUIT should break before any results are added


# ============================================================================
# Display Method Tests (with mocked console)
# ============================================================================


class TestDisplayMethods:
    """Tests for display methods."""

    def test_get_confidence_style_high(self, interactive_processor):
        """Test high confidence style."""
        style = interactive_processor._get_confidence_style(0.85)
        assert style == "green"

    def test_get_confidence_style_medium(self, interactive_processor):
        """Test medium confidence style."""
        style = interactive_processor._get_confidence_style(0.6)
        assert style == "yellow"

    def test_get_confidence_style_low(self, interactive_processor):
        """Test low confidence style."""
        style = interactive_processor._get_confidence_style(0.3)
        assert style == "red"


# Export test markers for pytest
__all__ = [
    "TestProposedChanges",
    "TestInteractiveSessionStats",
    "TestInteractiveProcessor",
    "TestActionHandling",
    "TestStateManagement",
    "TestStatisticsUpdate",
    "TestAutoAccept",
    "TestUndo",
    "TestProcessedBookmark",
    "TestInteractiveAction",
    "TestInteractiveProcessorIntegration",
    "TestDisplayMethods",
]
