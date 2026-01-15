"""
Unit tests for processing mode abstraction.

Tests the ProcessingStages flag enum and ProcessingMode configuration
for controlling which processing stages are executed.
"""

import pytest

from bookmark_processor.core.processing_modes import (
    PROCESSING_MODES,
    ProcessingMode,
    ProcessingStages,
    get_predefined_mode,
)


class TestProcessingStages:
    """Test ProcessingStages flag enum."""

    def test_individual_stages(self):
        """Test individual stage values."""
        assert ProcessingStages.VALIDATION.value > 0
        assert ProcessingStages.CONTENT.value > 0
        assert ProcessingStages.AI.value > 0
        assert ProcessingStages.TAGS.value > 0
        assert ProcessingStages.FOLDERS.value > 0
        assert ProcessingStages.NONE.value == 0

    def test_stage_combination(self):
        """Test combining stages with bitwise OR."""
        stages = ProcessingStages.VALIDATION | ProcessingStages.AI

        assert stages.includes(ProcessingStages.VALIDATION)
        assert stages.includes(ProcessingStages.AI)
        assert not stages.includes(ProcessingStages.CONTENT)
        assert not stages.includes(ProcessingStages.TAGS)

    def test_all_stages(self):
        """Test ALL property includes all stages."""
        all_stages = ProcessingStages.get_all()

        assert all_stages.includes(ProcessingStages.VALIDATION)
        assert all_stages.includes(ProcessingStages.CONTENT)
        assert all_stages.includes(ProcessingStages.AI)
        assert all_stages.includes(ProcessingStages.TAGS)
        assert all_stages.includes(ProcessingStages.FOLDERS)

    def test_validate_only(self):
        """Test VALIDATE_ONLY includes only validation."""
        stages = ProcessingStages.get_validate_only()

        assert stages.includes(ProcessingStages.VALIDATION)
        assert not stages.includes(ProcessingStages.CONTENT)
        assert not stages.includes(ProcessingStages.AI)

    def test_tags_only(self):
        """Test TAGS_ONLY includes only tags."""
        stages = ProcessingStages.get_tags_only()

        assert stages.includes(ProcessingStages.TAGS)
        assert not stages.includes(ProcessingStages.VALIDATION)
        assert not stages.includes(ProcessingStages.AI)

    def test_folders_only(self):
        """Test FOLDERS_ONLY includes only folders."""
        stages = ProcessingStages.get_folders_only()

        assert stages.includes(ProcessingStages.FOLDERS)
        assert not stages.includes(ProcessingStages.VALIDATION)
        assert not stages.includes(ProcessingStages.AI)

    def test_no_ai(self):
        """Test NO_AI includes all except AI."""
        stages = ProcessingStages.get_no_ai()

        assert stages.includes(ProcessingStages.VALIDATION)
        assert stages.includes(ProcessingStages.CONTENT)
        assert not stages.includes(ProcessingStages.AI)
        assert stages.includes(ProcessingStages.TAGS)
        assert stages.includes(ProcessingStages.FOLDERS)

    def test_no_validation(self):
        """Test NO_VALIDATION includes all except validation."""
        stages = ProcessingStages.get_no_validation()

        assert not stages.includes(ProcessingStages.VALIDATION)
        assert stages.includes(ProcessingStages.CONTENT)
        assert stages.includes(ProcessingStages.AI)
        assert stages.includes(ProcessingStages.TAGS)
        assert stages.includes(ProcessingStages.FOLDERS)

    def test_includes_method(self):
        """Test includes method."""
        stages = ProcessingStages.VALIDATION | ProcessingStages.CONTENT

        assert stages.includes(ProcessingStages.VALIDATION) is True
        assert stages.includes(ProcessingStages.CONTENT) is True
        assert stages.includes(ProcessingStages.AI) is False

    def test_without_method(self):
        """Test without method removes a stage."""
        stages = ProcessingStages.get_all()
        stages = stages.without(ProcessingStages.AI)

        assert stages.includes(ProcessingStages.VALIDATION)
        assert stages.includes(ProcessingStages.CONTENT)
        assert not stages.includes(ProcessingStages.AI)
        assert stages.includes(ProcessingStages.TAGS)

    def test_with_stage_method(self):
        """Test with_stage method adds a stage."""
        stages = ProcessingStages.VALIDATION
        stages = stages.with_stage(ProcessingStages.AI)

        assert stages.includes(ProcessingStages.VALIDATION)
        assert stages.includes(ProcessingStages.AI)
        assert not stages.includes(ProcessingStages.CONTENT)

    def test_stage_list(self):
        """Test stage_list property."""
        stages = ProcessingStages.VALIDATION | ProcessingStages.AI | ProcessingStages.TAGS
        stage_list = stages.stage_list

        assert "validation" in stage_list
        assert "ai" in stage_list
        assert "tags" in stage_list
        assert "content" not in stage_list
        assert "folders" not in stage_list

    def test_stage_list_all(self):
        """Test stage_list for all stages."""
        stage_list = ProcessingStages.get_all().stage_list

        assert len(stage_list) == 5
        assert set(stage_list) == {"validation", "content", "ai", "tags", "folders"}

    def test_stage_list_none(self):
        """Test stage_list for no stages."""
        stage_list = ProcessingStages.NONE.stage_list

        assert len(stage_list) == 0

    def test_from_list(self):
        """Test creating stages from list."""
        stages = ProcessingStages.from_list(["validation", "ai", "tags"])

        assert stages.includes(ProcessingStages.VALIDATION)
        assert stages.includes(ProcessingStages.AI)
        assert stages.includes(ProcessingStages.TAGS)
        assert not stages.includes(ProcessingStages.CONTENT)

    def test_from_list_all(self):
        """Test creating all stages from list."""
        stages = ProcessingStages.from_list(["all"])

        assert stages == ProcessingStages.get_all()

    def test_from_list_case_insensitive(self):
        """Test from_list is case-insensitive."""
        stages = ProcessingStages.from_list(["VALIDATION", "Ai", "tags"])

        assert stages.includes(ProcessingStages.VALIDATION)
        assert stages.includes(ProcessingStages.AI)
        assert stages.includes(ProcessingStages.TAGS)

    def test_from_list_invalid_stage(self):
        """Test from_list with invalid stage raises error."""
        with pytest.raises(ValueError, match="Unknown stage"):
            ProcessingStages.from_list(["validation", "invalid_stage"])


class TestProcessingMode:
    """Test ProcessingMode class."""

    def test_default_creation(self):
        """Test default ProcessingMode creation."""
        mode = ProcessingMode()

        assert mode.stages == ProcessingStages.get_all()
        assert mode.preview_count is None
        assert mode.dry_run is False
        assert mode.verbose is False
        assert mode.continue_on_error is True

    def test_is_preview(self):
        """Test is_preview property."""
        full_mode = ProcessingMode()
        preview_mode = ProcessingMode(preview_count=10)

        assert not full_mode.is_preview
        assert preview_mode.is_preview

    def test_is_full_run(self):
        """Test is_full_run property."""
        full_mode = ProcessingMode()
        preview_mode = ProcessingMode(preview_count=10)
        dry_run_mode = ProcessingMode(dry_run=True)

        assert full_mode.is_full_run
        assert not preview_mode.is_full_run
        assert not dry_run_mode.is_full_run

    def test_will_write_output(self):
        """Test will_write_output property."""
        normal_mode = ProcessingMode()
        dry_run_mode = ProcessingMode(dry_run=True)

        assert normal_mode.will_write_output
        assert not dry_run_mode.will_write_output

    def test_should_run_stage(self):
        """Test should_run_stage method."""
        mode = ProcessingMode(stages=ProcessingStages.VALIDATION | ProcessingStages.AI)

        assert mode.should_run_stage(ProcessingStages.VALIDATION)
        assert mode.should_run_stage(ProcessingStages.AI)
        assert not mode.should_run_stage(ProcessingStages.CONTENT)
        assert not mode.should_run_stage(ProcessingStages.TAGS)

    def test_stage_convenience_properties(self):
        """Test convenience properties for stage checks."""
        mode = ProcessingMode(stages=ProcessingStages.VALIDATION | ProcessingStages.AI)

        assert mode.should_validate
        assert not mode.should_extract_content
        assert mode.should_run_ai
        assert not mode.should_optimize_tags
        assert not mode.should_organize_folders

    def test_get_description_full(self):
        """Test description for full processing."""
        mode = ProcessingMode()
        desc = mode.get_description()

        assert "Full processing" in desc
        assert "all stages enabled" in desc

    def test_get_description_preview(self):
        """Test description for preview mode."""
        mode = ProcessingMode(preview_count=10)
        desc = mode.get_description()

        assert "Preview mode" in desc
        assert "10 items" in desc

    def test_get_description_dry_run(self):
        """Test description for dry-run mode."""
        mode = ProcessingMode(dry_run=True)
        desc = mode.get_description()

        assert "Dry-run mode" in desc

    def test_get_description_limited_stages(self):
        """Test description with limited stages."""
        mode = ProcessingMode(stages=ProcessingStages.VALIDATION | ProcessingStages.TAGS)
        desc = mode.get_description()

        assert "stages:" in desc
        assert "validation" in desc
        assert "tags" in desc


class TestProcessingModeFromCLI:
    """Test ProcessingMode.from_cli_args method."""

    def test_from_cli_args_defaults(self):
        """Test creating mode from empty args."""
        mode = ProcessingMode.from_cli_args({})

        assert mode.stages == ProcessingStages.get_all()
        assert mode.preview_count is None
        assert mode.dry_run is False

    def test_from_cli_args_preview(self):
        """Test preview argument."""
        mode = ProcessingMode.from_cli_args({"preview": 10})

        assert mode.preview_count == 10
        assert mode.is_preview

    def test_from_cli_args_dry_run(self):
        """Test dry_run argument."""
        mode = ProcessingMode.from_cli_args({"dry_run": True})

        assert mode.dry_run is True

    def test_from_cli_args_skip_validation(self):
        """Test skip_validation argument."""
        mode = ProcessingMode.from_cli_args({"skip_validation": True})

        assert not mode.should_validate
        assert mode.should_extract_content
        assert mode.should_run_ai
        assert mode.should_optimize_tags
        assert mode.should_organize_folders

    def test_from_cli_args_skip_ai(self):
        """Test skip_ai argument."""
        mode = ProcessingMode.from_cli_args({"skip_ai": True})

        assert mode.should_validate
        assert mode.should_extract_content
        assert not mode.should_run_ai
        assert mode.should_optimize_tags

    def test_from_cli_args_tags_only(self):
        """Test tags_only argument."""
        mode = ProcessingMode.from_cli_args({"tags_only": True})

        assert not mode.should_validate
        assert not mode.should_extract_content
        assert not mode.should_run_ai
        assert mode.should_optimize_tags
        assert not mode.should_organize_folders

    def test_from_cli_args_folders_only(self):
        """Test folders_only argument."""
        mode = ProcessingMode.from_cli_args({"folders_only": True})

        assert not mode.should_validate
        assert not mode.should_run_ai
        assert not mode.should_optimize_tags
        assert mode.should_organize_folders

    def test_from_cli_args_validate_only(self):
        """Test validate_only argument."""
        mode = ProcessingMode.from_cli_args({"validate_only": True})

        assert mode.should_validate
        assert not mode.should_extract_content
        assert not mode.should_run_ai
        assert not mode.should_optimize_tags

    def test_from_cli_args_explicit_stages(self):
        """Test explicit stages argument."""
        mode = ProcessingMode.from_cli_args({
            "stages": ["validation", "ai", "tags"]
        })

        assert mode.should_validate
        assert not mode.should_extract_content
        assert mode.should_run_ai
        assert mode.should_optimize_tags
        assert not mode.should_organize_folders

    def test_from_cli_args_multiple_skips(self):
        """Test multiple skip arguments."""
        mode = ProcessingMode.from_cli_args({
            "skip_validation": True,
            "skip_ai": True,
            "skip_folders": True,
        })

        assert not mode.should_validate
        assert mode.should_extract_content
        assert not mode.should_run_ai
        assert mode.should_optimize_tags
        assert not mode.should_organize_folders

    def test_from_cli_args_verbose(self):
        """Test verbose argument."""
        mode = ProcessingMode.from_cli_args({"verbose": True})

        assert mode.verbose is True

    def test_from_cli_args_continue_on_error(self):
        """Test continue_on_error argument."""
        mode = ProcessingMode.from_cli_args({"continue_on_error": False})

        assert mode.continue_on_error is False


class TestProcessingModeFactoryMethods:
    """Test ProcessingMode factory methods."""

    def test_preview_factory(self):
        """Test preview factory method."""
        mode = ProcessingMode.preview(20)

        assert mode.preview_count == 20
        assert mode.is_preview

    def test_preview_factory_default_count(self):
        """Test preview factory with default count."""
        mode = ProcessingMode.preview()

        assert mode.preview_count == 10

    def test_dry_run_mode_factory(self):
        """Test dry_run_mode factory method."""
        mode = ProcessingMode.dry_run_mode()

        assert mode.dry_run is True

    def test_tags_only_mode_factory(self):
        """Test tags_only_mode factory method."""
        mode = ProcessingMode.tags_only_mode()

        assert mode.should_optimize_tags
        assert not mode.should_validate
        assert not mode.should_run_ai

    def test_validation_only_mode_factory(self):
        """Test validation_only_mode factory method."""
        mode = ProcessingMode.validation_only_mode()

        assert mode.should_validate
        assert not mode.should_extract_content
        assert not mode.should_run_ai

    def test_no_ai_mode_factory(self):
        """Test no_ai_mode factory method."""
        mode = ProcessingMode.no_ai_mode()

        assert mode.should_validate
        assert mode.should_extract_content
        assert not mode.should_run_ai
        assert mode.should_optimize_tags
        assert mode.should_organize_folders


class TestProcessingModeCopy:
    """Test ProcessingMode.copy method."""

    def test_copy_no_overrides(self):
        """Test copy without overrides."""
        original = ProcessingMode(preview_count=10, dry_run=True, verbose=True)
        copy = original.copy()

        assert copy.preview_count == 10
        assert copy.dry_run is True
        assert copy.verbose is True
        assert copy is not original

    def test_copy_with_overrides(self):
        """Test copy with overrides."""
        original = ProcessingMode(preview_count=10, dry_run=True)
        copy = original.copy(preview_count=20, dry_run=False)

        assert copy.preview_count == 20
        assert copy.dry_run is False
        # Original unchanged
        assert original.preview_count == 10
        assert original.dry_run is True

    def test_copy_stages_override(self):
        """Test copying with stage override."""
        original = ProcessingMode()
        copy = original.copy(stages=ProcessingStages.get_tags_only())

        assert copy.should_optimize_tags
        assert not copy.should_validate
        # Original unchanged
        assert original.should_validate


class TestProcessingModeToDict:
    """Test ProcessingMode.to_dict method."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        mode = ProcessingMode(preview_count=10, dry_run=True)
        result = mode.to_dict()

        assert result["preview_count"] == 10
        assert result["dry_run"] is True
        assert result["is_preview"] is True
        assert result["will_write_output"] is False

    def test_to_dict_stages(self):
        """Test to_dict includes stages as list."""
        mode = ProcessingMode(stages=ProcessingStages.VALIDATION | ProcessingStages.AI)
        result = mode.to_dict()

        assert "validation" in result["stages"]
        assert "ai" in result["stages"]
        assert "content" not in result["stages"]


class TestPredefinedModes:
    """Test predefined processing modes."""

    def test_predefined_modes_exist(self):
        """Test that predefined modes exist."""
        assert "full" in PROCESSING_MODES
        assert "preview" in PROCESSING_MODES
        assert "dry_run" in PROCESSING_MODES
        assert "tags_only" in PROCESSING_MODES
        assert "validation_only" in PROCESSING_MODES
        assert "no_ai" in PROCESSING_MODES

    def test_get_predefined_mode_full(self):
        """Test getting full mode."""
        mode = get_predefined_mode("full")

        assert mode.is_full_run

    def test_get_predefined_mode_preview(self):
        """Test getting preview mode."""
        mode = get_predefined_mode("preview")

        assert mode.is_preview

    def test_get_predefined_mode_dry_run(self):
        """Test getting dry_run mode."""
        mode = get_predefined_mode("dry_run")

        assert mode.dry_run

    def test_get_predefined_mode_case_insensitive(self):
        """Test get_predefined_mode is case-insensitive."""
        mode = get_predefined_mode("FULL")

        assert mode.is_full_run

    def test_get_predefined_mode_invalid(self):
        """Test get_predefined_mode with invalid name."""
        with pytest.raises(ValueError, match="Unknown mode"):
            get_predefined_mode("invalid_mode")

    def test_predefined_modes_return_copies(self):
        """Test that get_predefined_mode returns copies."""
        mode1 = get_predefined_mode("full")
        mode2 = get_predefined_mode("full")

        assert mode1 is not mode2

        # Modifying one should not affect the other
        mode1.preview_count = 10
        assert mode2.preview_count is None


class TestProcessingModeIntegration:
    """Integration tests for processing modes."""

    def test_mode_combinations(self):
        """Test various mode combinations work correctly."""
        # Preview with limited stages
        mode = ProcessingMode(
            stages=ProcessingStages.VALIDATION | ProcessingStages.AI,
            preview_count=5
        )

        assert mode.is_preview
        assert mode.should_validate
        assert mode.should_run_ai
        assert not mode.should_extract_content

    def test_cli_args_to_mode_roundtrip(self):
        """Test that CLI args create expected mode."""
        args = {
            "preview": 10,
            "dry_run": False,
            "skip_ai": True,
            "verbose": True,
        }

        mode = ProcessingMode.from_cli_args(args)
        result = mode.to_dict()

        assert result["preview_count"] == 10
        assert result["dry_run"] is False
        assert "ai" not in result["stages"]
        assert result["verbose"] is True

    def test_stage_exclusivity_exclusive_modes(self):
        """Test that exclusive modes (tags_only, etc.) take precedence."""
        # tags_only should override skip flags
        args = {
            "tags_only": True,
            "skip_validation": False,
            "skip_ai": False,
        }

        mode = ProcessingMode.from_cli_args(args)

        # Only tags should be enabled regardless of skip flags
        assert mode.should_optimize_tags
        assert not mode.should_validate
        assert not mode.should_run_ai


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
