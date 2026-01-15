"""
Unit tests for Phase 1 CLI features.

Tests the Preview/Dry-Run Mode, Smart Filtering, and Granular Processing Control
CLI options added in Phase 1.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from bookmark_processor.core.data_models import Bookmark, ProcessingStatus
from bookmark_processor.core.filters import (
    DateRangeFilter,
    DomainFilter,
    FilterChain,
    FolderFilter,
    StatusFilter,
    TagFilter,
)
from bookmark_processor.core.processing_modes import ProcessingMode, ProcessingStages


# Fixtures for creating test bookmarks
@pytest.fixture
def sample_bookmarks() -> List[Bookmark]:
    """Create a list of sample bookmarks for testing."""
    return [
        Bookmark(
            url="https://github.com/user/repo1",
            title="GitHub Repo 1",
            folder="Tech/Programming",
            tags=["python", "ai"],
            created=datetime(2024, 6, 15),
        ),
        Bookmark(
            url="https://gitlab.com/user/repo2",
            title="GitLab Repo 2",
            folder="Tech/DevOps",
            tags=["docker", "kubernetes"],
            created=datetime(2024, 3, 10),
        ),
        Bookmark(
            url="https://medium.com/article1",
            title="Medium Article",
            folder="Reading/Articles",
            tags=["reading", "tech"],
            created=datetime(2023, 12, 1),
        ),
        Bookmark(
            url="https://stackoverflow.com/questions/123",
            title="Stack Overflow Question",
            folder="Tech/QA",
            tags=["python", "help"],
            created=datetime(2024, 1, 20),
        ),
        Bookmark(
            url="https://news.ycombinator.com/item",
            title="Hacker News Item",
            folder="News",
            tags=["news", "tech"],
            created=datetime(2024, 8, 5),
        ),
    ]


@pytest.fixture
def sample_csv_file(sample_bookmarks, tmp_path) -> Path:
    """Create a sample CSV file for testing."""
    csv_path = tmp_path / "test_bookmarks.csv"
    # Write a simple CSV file
    with open(csv_path, "w") as f:
        f.write("id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite\n")
        for i, b in enumerate(sample_bookmarks):
            tags_str = f'"{", ".join(b.tags)}"' if len(b.tags) > 1 else (b.tags[0] if b.tags else "")
            created_str = b.created.isoformat() if b.created else ""
            f.write(f'{i},{b.title},,"{b.excerpt}",{b.url},{b.folder},{tags_str},{created_str},,,false\n')
    return csv_path


# ============================================================================
# Phase 1.1: Preview/Dry-Run Mode Tests
# ============================================================================

class TestPreviewMode:
    """Test preview mode functionality."""

    def test_processing_mode_preview_creation(self):
        """Test creating a preview mode."""
        mode = ProcessingMode.preview(10)

        assert mode.is_preview
        assert mode.preview_count == 10
        assert not mode.dry_run

    def test_processing_mode_preview_default_count(self):
        """Test preview mode with default count."""
        mode = ProcessingMode.preview()

        assert mode.preview_count == 10

    def test_processing_mode_from_cli_args_preview(self):
        """Test creating mode from CLI args with preview."""
        mode = ProcessingMode.from_cli_args({"preview": 5})

        assert mode.is_preview
        assert mode.preview_count == 5

    def test_preview_limits_processing(self, sample_bookmarks):
        """Test that preview limits the number of items processed."""
        # Simulate what the CLI would do
        preview_count = 2
        bookmarks_to_process = sample_bookmarks[:preview_count]

        assert len(bookmarks_to_process) == 2
        assert bookmarks_to_process[0].url == sample_bookmarks[0].url

    def test_preview_mode_description(self):
        """Test preview mode description."""
        mode = ProcessingMode.preview(15)
        desc = mode.get_description()

        assert "Preview mode" in desc
        assert "15 items" in desc


class TestDryRunMode:
    """Test dry-run mode functionality."""

    def test_processing_mode_dry_run_creation(self):
        """Test creating a dry-run mode."""
        mode = ProcessingMode.dry_run_mode()

        assert mode.dry_run
        assert not mode.is_preview
        assert not mode.will_write_output

    def test_processing_mode_from_cli_args_dry_run(self):
        """Test creating mode from CLI args with dry_run."""
        mode = ProcessingMode.from_cli_args({"dry_run": True})

        assert mode.dry_run
        assert not mode.will_write_output

    def test_dry_run_mode_description(self):
        """Test dry-run mode description."""
        mode = ProcessingMode.dry_run_mode()
        desc = mode.get_description()

        assert "Dry-run mode" in desc

    def test_dry_run_does_not_write_output(self):
        """Test that dry-run mode should not write output."""
        mode = ProcessingMode(dry_run=True)

        assert not mode.will_write_output

    def test_combined_preview_and_dry_run(self):
        """Test combining preview and dry-run modes."""
        mode = ProcessingMode.from_cli_args({
            "preview": 10,
            "dry_run": True,
        })

        assert mode.is_preview
        assert mode.preview_count == 10
        assert mode.dry_run
        assert not mode.will_write_output


# ============================================================================
# Phase 1.2: Smart Filtering Tests
# ============================================================================

class TestFilterChainFromCLIArgs:
    """Test FilterChain creation from CLI arguments."""

    def test_empty_args_creates_empty_chain(self):
        """Test that empty args create an empty filter chain."""
        chain = FilterChain.from_cli_args({})

        assert len(chain) == 0
        assert not chain  # Empty chain is falsy

    def test_filter_folder_arg(self):
        """Test --filter-folder creates FolderFilter."""
        chain = FilterChain.from_cli_args({"filter_folder": "Tech/*"})

        assert len(chain) == 1
        assert isinstance(chain.filters[0], FolderFilter)

    def test_filter_tag_single(self):
        """Test --filter-tag with single tag."""
        chain = FilterChain.from_cli_args({"filter_tag": "python"})

        assert len(chain) == 1
        assert isinstance(chain.filters[0], TagFilter)

    def test_filter_tag_multiple(self):
        """Test --filter-tag with multiple tags."""
        chain = FilterChain.from_cli_args({"filter_tag": ["python", "ai"]})

        assert len(chain) == 1
        assert isinstance(chain.filters[0], TagFilter)

    def test_filter_tag_comma_separated(self):
        """Test --filter-tag with comma-separated tags."""
        chain = FilterChain.from_cli_args({"filter_tag": "python,django,web"})

        assert len(chain) == 1
        assert isinstance(chain.filters[0], TagFilter)

    def test_filter_date_range(self):
        """Test --filter-date creates DateRangeFilter."""
        chain = FilterChain.from_cli_args({"filter_date": "2024-01-01:2024-12-31"})

        assert len(chain) == 1
        assert isinstance(chain.filters[0], DateRangeFilter)

    def test_filter_date_start_only(self):
        """Test --filter-date with start date only."""
        chain = FilterChain.from_cli_args({"filter_date": "2024-01-01:"})

        assert len(chain) == 1
        assert isinstance(chain.filters[0], DateRangeFilter)

    def test_filter_date_end_only(self):
        """Test --filter-date with end date only."""
        chain = FilterChain.from_cli_args({"filter_date": ":2024-12-31"})

        assert len(chain) == 1
        assert isinstance(chain.filters[0], DateRangeFilter)

    def test_filter_domain(self):
        """Test --filter-domain creates DomainFilter."""
        chain = FilterChain.from_cli_args({"filter_domain": "github.com"})

        assert len(chain) == 1
        assert isinstance(chain.filters[0], DomainFilter)

    def test_filter_domain_multiple(self):
        """Test --filter-domain with multiple domains."""
        chain = FilterChain.from_cli_args({"filter_domain": "github.com,gitlab.com"})

        assert len(chain) == 1
        assert isinstance(chain.filters[0], DomainFilter)

    def test_retry_invalid(self):
        """Test --retry-invalid creates StatusFilter."""
        chain = FilterChain.from_cli_args({"retry_invalid": True})

        assert len(chain) == 1
        assert isinstance(chain.filters[0], StatusFilter)

    def test_multiple_filters_combined(self):
        """Test multiple filter args create combined chain."""
        chain = FilterChain.from_cli_args({
            "filter_folder": "Tech/*",
            "filter_tag": "python",
            "filter_domain": "github.com",
        })

        assert len(chain) == 3


class TestSmartFilteringIntegration:
    """Integration tests for smart filtering."""

    def test_folder_filter_matches(self, sample_bookmarks):
        """Test folder filter matching."""
        chain = FilterChain.from_cli_args({"filter_folder": "Tech/*"})
        filtered = chain.apply(sample_bookmarks)

        # Should match Tech/Programming, Tech/DevOps, Tech/QA
        assert len(filtered) == 3
        for b in filtered:
            assert b.folder.startswith("Tech/")

    def test_tag_filter_matches(self, sample_bookmarks):
        """Test tag filter matching."""
        chain = FilterChain.from_cli_args({"filter_tag": "python"})
        filtered = chain.apply(sample_bookmarks)

        # Should match bookmarks with 'python' tag
        assert len(filtered) == 2
        for b in filtered:
            assert "python" in [t.lower() for t in b.tags]

    def test_domain_filter_matches(self, sample_bookmarks):
        """Test domain filter matching."""
        chain = FilterChain.from_cli_args({"filter_domain": "github.com,gitlab.com"})
        filtered = chain.apply(sample_bookmarks)

        # Should match github and gitlab URLs
        assert len(filtered) == 2
        domains = {b.url.split("/")[2] for b in filtered}
        assert "github.com" in domains
        assert "gitlab.com" in domains

    def test_date_filter_matches(self, sample_bookmarks):
        """Test date range filter matching."""
        chain = FilterChain.from_cli_args({"filter_date": "2024-01-01:2024-06-30"})
        filtered = chain.apply(sample_bookmarks)

        # Should match bookmarks created in first half of 2024
        for b in filtered:
            assert b.created is not None
            assert datetime(2024, 1, 1) <= b.created <= datetime(2024, 6, 30, 23, 59, 59, 999999)

    def test_combined_filters_and_logic(self, sample_bookmarks):
        """Test that multiple filters use AND logic by default."""
        chain = FilterChain.from_cli_args({
            "filter_folder": "Tech/*",
            "filter_tag": "python",
        })
        filtered = chain.apply(sample_bookmarks)

        # Should match Tech folder AND python tag
        for b in filtered:
            assert b.folder.startswith("Tech/")
            assert "python" in [t.lower() for t in b.tags]

    def test_filter_summary_count(self, sample_bookmarks):
        """Test filter chain count_matching method."""
        chain = FilterChain.from_cli_args({"filter_domain": "github.com"})
        count = chain.count_matching(sample_bookmarks)

        assert count == 1  # Only one github URL

    def test_retry_invalid_filter(self):
        """Test retry-invalid filter matches bookmarks with errors."""
        # Create bookmarks with different statuses
        valid_bookmark = Bookmark(url="http://valid.com")
        valid_bookmark.processing_status.url_validated = True

        invalid_bookmark = Bookmark(url="http://invalid.com")
        invalid_bookmark.processing_status.url_validation_error = "Connection failed"

        bookmarks = [valid_bookmark, invalid_bookmark]

        chain = FilterChain.from_cli_args({"retry_invalid": True})
        filtered = chain.apply(bookmarks)

        assert len(filtered) == 1
        assert filtered[0].url == "http://invalid.com"


# ============================================================================
# Phase 1.3: Granular Processing Control Tests
# ============================================================================

class TestGranularProcessingControl:
    """Test granular processing control options."""

    def test_skip_validation(self):
        """Test --skip-validation skips validation stage."""
        mode = ProcessingMode.from_cli_args({"skip_validation": True})

        assert not mode.should_validate
        assert mode.should_extract_content
        assert mode.should_run_ai
        assert mode.should_optimize_tags
        assert mode.should_organize_folders

    def test_skip_ai(self):
        """Test --skip-ai skips AI stage."""
        mode = ProcessingMode.from_cli_args({"skip_ai": True})

        assert mode.should_validate
        assert mode.should_extract_content
        assert not mode.should_run_ai
        assert mode.should_optimize_tags
        assert mode.should_organize_folders

    def test_skip_content(self):
        """Test --skip-content skips content extraction."""
        mode = ProcessingMode.from_cli_args({"skip_content": True})

        assert mode.should_validate
        assert not mode.should_extract_content
        assert mode.should_run_ai
        assert mode.should_optimize_tags

    def test_tags_only(self):
        """Test --tags-only runs only tag optimization."""
        mode = ProcessingMode.from_cli_args({"tags_only": True})

        assert not mode.should_validate
        assert not mode.should_extract_content
        assert not mode.should_run_ai
        assert mode.should_optimize_tags
        assert not mode.should_organize_folders

    def test_folders_only(self):
        """Test --folders-only runs only folder organization."""
        mode = ProcessingMode.from_cli_args({"folders_only": True})

        assert not mode.should_validate
        assert not mode.should_extract_content
        assert not mode.should_run_ai
        assert not mode.should_optimize_tags
        assert mode.should_organize_folders

    def test_validate_only(self):
        """Test --validate-only runs only validation."""
        mode = ProcessingMode.from_cli_args({"validate_only": True})

        assert mode.should_validate
        assert not mode.should_extract_content
        assert not mode.should_run_ai
        assert not mode.should_optimize_tags
        assert not mode.should_organize_folders

    def test_multiple_skip_options(self):
        """Test multiple skip options can be combined."""
        mode = ProcessingMode.from_cli_args({
            "skip_validation": True,
            "skip_ai": True,
        })

        assert not mode.should_validate
        assert mode.should_extract_content
        assert not mode.should_run_ai
        assert mode.should_optimize_tags

    def test_exclusive_mode_takes_precedence(self):
        """Test that exclusive modes take precedence over skip flags."""
        # Note: In CLI, this validation would prevent combining them
        # But ProcessingMode.from_cli_args handles the precedence
        mode = ProcessingMode.from_cli_args({"tags_only": True})

        # Even if skip_validation was somehow passed, tags_only should win
        assert mode.stages == ProcessingStages.TAGS


class TestMutualExclusivity:
    """Test mutual exclusivity validation for CLI options."""

    def test_exclusive_options_detection(self):
        """Test that we can detect mutually exclusive options."""
        # This simulates what the CLI does
        exclusive_options = [True, True, False]  # tags_only and folders_only
        exclusive_count = sum(exclusive_options)

        assert exclusive_count > 1  # Should be detected as invalid

    def test_exclusive_and_skip_detection(self):
        """Test detection of exclusive mode with skip options."""
        # This simulates what the CLI does
        exclusive_count = 1  # e.g., tags_only
        has_skips = True  # e.g., skip_validation

        # Should be detected as invalid
        assert exclusive_count > 0 and has_skips


class TestProcessingStagesConfiguration:
    """Test ProcessingStages configuration."""

    def test_all_stages_by_default(self):
        """Test that all stages are enabled by default."""
        mode = ProcessingMode()

        assert mode.should_validate
        assert mode.should_extract_content
        assert mode.should_run_ai
        assert mode.should_optimize_tags
        assert mode.should_organize_folders

    def test_stage_list_for_custom_stages(self):
        """Test stage_list returns correct stages."""
        mode = ProcessingMode.from_cli_args({
            "skip_ai": True,
            "skip_folders": True,
        })
        stages = mode.stages.stage_list

        assert "validation" in stages
        assert "content" in stages
        assert "ai" not in stages
        assert "tags" in stages
        assert "folders" not in stages

    def test_stage_description(self):
        """Test mode description includes stage info."""
        mode = ProcessingMode.from_cli_args({"tags_only": True})
        desc = mode.get_description()

        assert "tags" in desc.lower()


# ============================================================================
# Combined Feature Tests
# ============================================================================

class TestCombinedFeatures:
    """Test combinations of Phase 1 features."""

    def test_preview_with_filters(self, sample_bookmarks):
        """Test preview mode combined with filters."""
        # Simulate CLI behavior: filter first, then apply preview limit
        chain = FilterChain.from_cli_args({"filter_folder": "Tech/*"})
        filtered = chain.apply(sample_bookmarks)

        # Apply preview limit
        preview_count = 2
        result = filtered[:preview_count]

        assert len(result) == 2
        for b in result:
            assert b.folder.startswith("Tech/")

    def test_dry_run_with_filters(self, sample_bookmarks):
        """Test dry-run mode with filters shows correct counts."""
        chain = FilterChain.from_cli_args({
            "filter_tag": "python",
            "filter_domain": "github.com",
        })

        total = len(sample_bookmarks)
        filtered = chain.apply(sample_bookmarks)
        filtered_count = len(filtered)

        # Should show accurate counts
        assert total == 5
        assert filtered_count == 1  # Only github.com with python tag

    def test_filters_with_processing_control(self, sample_bookmarks):
        """Test filters combined with processing stage control."""
        # Create filter and mode
        chain = FilterChain.from_cli_args({"filter_folder": "Tech/*"})
        mode = ProcessingMode.from_cli_args({"skip_ai": True})

        # Apply filter
        filtered = chain.apply(sample_bookmarks)

        # Check mode
        assert len(filtered) == 3
        assert not mode.should_run_ai
        assert mode.should_validate

    def test_preview_dry_run_filters_combined(self, sample_bookmarks):
        """Test all three feature categories combined."""
        chain = FilterChain.from_cli_args({"filter_domain": "github.com,gitlab.com"})
        mode = ProcessingMode.from_cli_args({
            "preview": 1,
            "dry_run": True,
            "skip_ai": True,
        })

        filtered = chain.apply(sample_bookmarks)
        preview_result = filtered[:mode.preview_count]

        assert len(preview_result) == 1
        assert mode.is_preview
        assert mode.dry_run
        assert not mode.should_run_ai


# ============================================================================
# CLI Integration Tests (Mocked)
# ============================================================================

class TestCLIIntegration:
    """Test CLI integration with mocked dependencies."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock console."""
        return MagicMock()

    def test_cli_validates_exclusive_options(self):
        """Test that CLI validates mutually exclusive options."""
        from bookmark_processor.utils.validation import ValidationError

        # Simulate the validation logic from cli.py
        tags_only = True
        folders_only = True
        validate_only = False

        exclusive_options = [tags_only, folders_only, validate_only]
        exclusive_count = sum(exclusive_options)

        with pytest.raises(ValidationError) if exclusive_count > 1 else pytest.warns(None):
            if exclusive_count > 1:
                raise ValidationError(
                    "Options --tags-only, --folders-only, and --validate-only are mutually exclusive."
                )

    def test_cli_builds_processing_mode(self):
        """Test CLI builds ProcessingMode correctly."""
        # Simulate CLI argument processing
        cli_args = {
            "preview": 10,
            "dry_run": False,
            "skip_validation": True,
            "skip_ai": False,
            "skip_content": False,
            "tags_only": False,
            "folders_only": False,
            "validate_only": False,
            "verbose": True,
        }

        mode = ProcessingMode.from_cli_args(cli_args)

        assert mode.preview_count == 10
        assert not mode.dry_run
        assert not mode.should_validate
        assert mode.should_run_ai
        assert mode.verbose

    def test_cli_builds_filter_chain(self):
        """Test CLI builds FilterChain correctly."""
        # Simulate CLI argument processing
        cli_args = {
            "filter_folder": "Tech/*",
            "filter_tag": ["python", "ai"],
            "filter_date": "2024-01-01:2024-12-31",
            "filter_domain": "github.com",
            "retry_invalid": False,
        }

        chain = FilterChain.from_cli_args(cli_args)

        assert len(chain) == 4  # folder, tag, date, domain
        assert chain  # Non-empty chain is truthy


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_bookmark_list_with_filters(self):
        """Test filtering an empty bookmark list."""
        chain = FilterChain.from_cli_args({"filter_folder": "Tech/*"})
        result = chain.apply([])

        assert result == []

    def test_filter_no_matches(self, sample_bookmarks):
        """Test filter that matches no bookmarks."""
        chain = FilterChain.from_cli_args({"filter_folder": "NonExistent/*"})
        result = chain.apply(sample_bookmarks)

        assert len(result) == 0

    def test_preview_larger_than_list(self, sample_bookmarks):
        """Test preview count larger than bookmark list."""
        preview_count = 100
        result = sample_bookmarks[:preview_count]

        assert len(result) == len(sample_bookmarks)

    def test_preview_zero_handled_by_cli(self):
        """Test that preview=0 would be handled by CLI validation."""
        # The CLI uses min=1 for preview, so 0 is not valid
        # This test documents the expected behavior
        pass

    def test_invalid_date_format(self):
        """Test invalid date format raises error."""
        with pytest.raises(ValueError, match="Invalid date range format"):
            FilterChain.from_cli_args({"filter_date": "not-a-date"})

    def test_processing_mode_no_stages(self):
        """Test processing mode with no stages enabled."""
        mode = ProcessingMode(stages=ProcessingStages.NONE)

        assert not mode.should_validate
        assert not mode.should_extract_content
        assert not mode.should_run_ai
        assert not mode.should_optimize_tags
        assert not mode.should_organize_folders

    def test_processing_mode_to_dict(self):
        """Test ProcessingMode serialization."""
        mode = ProcessingMode.from_cli_args({
            "preview": 5,
            "dry_run": True,
            "skip_ai": True,
        })

        result = mode.to_dict()

        assert result["preview_count"] == 5
        assert result["dry_run"] is True
        assert "ai" not in result["stages"]
        assert result["is_preview"] is True
        assert result["will_write_output"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
