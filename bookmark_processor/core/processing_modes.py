"""
Processing Mode Abstraction for Bookmark Processing.

This module provides configuration for controlling which processing
stages are executed, supporting preview mode, dry-run mode, and
granular stage control.
"""

from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any, Dict, List, Optional, Set


class ProcessingStages(Flag):
    """
    Flag enum for specifying which processing stages to execute.

    Stages can be combined using bitwise operators:
        stages = ProcessingStages.VALIDATION | ProcessingStages.CONTENT
    """

    NONE = 0
    VALIDATION = auto()      # URL validation
    CONTENT = auto()         # Content extraction
    AI = auto()              # AI description generation
    TAGS = auto()            # Tag optimization
    FOLDERS = auto()         # Folder organization

    @classmethod
    def get_all(cls) -> "ProcessingStages":
        """All processing stages."""
        return (
            cls.VALIDATION | cls.CONTENT | cls.AI | cls.TAGS | cls.FOLDERS
        )

    @classmethod
    def get_validate_only(cls) -> "ProcessingStages":
        """Only URL validation."""
        return cls.VALIDATION

    @classmethod
    def get_tags_only(cls) -> "ProcessingStages":
        """Only tag optimization."""
        return cls.TAGS

    @classmethod
    def get_folders_only(cls) -> "ProcessingStages":
        """Only folder organization."""
        return cls.FOLDERS

    @classmethod
    def get_no_ai(cls) -> "ProcessingStages":
        """All stages except AI."""
        return cls.VALIDATION | cls.CONTENT | cls.TAGS | cls.FOLDERS

    @classmethod
    def get_no_validation(cls) -> "ProcessingStages":
        """All stages except validation."""
        return cls.CONTENT | cls.AI | cls.TAGS | cls.FOLDERS

    def includes(self, stage: "ProcessingStages") -> bool:
        """
        Check if this stage configuration includes a specific stage.

        Args:
            stage: The stage to check for

        Returns:
            True if the stage is included
        """
        return bool(self & stage)

    def without(self, stage: "ProcessingStages") -> "ProcessingStages":
        """
        Return a new ProcessingStages without the specified stage.

        Args:
            stage: The stage to remove

        Returns:
            New ProcessingStages with the stage removed
        """
        return self & ~stage

    def with_stage(self, stage: "ProcessingStages") -> "ProcessingStages":
        """
        Return a new ProcessingStages with the specified stage added.

        Args:
            stage: The stage to add

        Returns:
            New ProcessingStages with the stage added
        """
        return self | stage

    @property
    def stage_list(self) -> List[str]:
        """
        Get a list of stage names that are enabled.

        Returns:
            List of stage names
        """
        stages = []
        if self.includes(ProcessingStages.VALIDATION):
            stages.append("validation")
        if self.includes(ProcessingStages.CONTENT):
            stages.append("content")
        if self.includes(ProcessingStages.AI):
            stages.append("ai")
        if self.includes(ProcessingStages.TAGS):
            stages.append("tags")
        if self.includes(ProcessingStages.FOLDERS):
            stages.append("folders")
        return stages

    @classmethod
    def from_list(cls, stage_names: List[str]) -> "ProcessingStages":
        """
        Create ProcessingStages from a list of stage names.

        Args:
            stage_names: List of stage names (validation, content, ai, tags, folders)

        Returns:
            ProcessingStages with specified stages enabled
        """
        result = cls.NONE

        name_map = {
            "validation": cls.VALIDATION,
            "content": cls.CONTENT,
            "ai": cls.AI,
            "tags": cls.TAGS,
            "folders": cls.FOLDERS,
            "all": cls.get_all(),
        }

        for name in stage_names:
            name_lower = name.lower().strip()
            if name_lower in name_map:
                result = result | name_map[name_lower]
            else:
                raise ValueError(
                    f"Unknown stage: {name}. Valid stages: {list(name_map.keys())}"
                )

        return result


@dataclass
class ProcessingMode:
    """
    Configuration for processing behavior.

    Controls which stages are executed, preview limits, and dry-run mode.
    """

    stages: ProcessingStages = field(default_factory=lambda: _get_all_stages())
    preview_count: Optional[int] = None  # None = process all
    dry_run: bool = False  # If True, don't write output
    verbose: bool = False  # Enable verbose output
    continue_on_error: bool = True  # Continue processing if errors occur

    @property
    def is_preview(self) -> bool:
        """Check if this is a preview run (limited item count)."""
        return self.preview_count is not None

    @property
    def is_full_run(self) -> bool:
        """Check if this is a full processing run."""
        return not self.is_preview and not self.dry_run

    @property
    def will_write_output(self) -> bool:
        """Check if this mode will write output files."""
        return not self.dry_run

    def should_run_stage(self, stage: ProcessingStages) -> bool:
        """
        Check if a specific stage should be executed.

        Args:
            stage: The processing stage to check

        Returns:
            True if the stage should be executed
        """
        return self.stages.includes(stage)

    @property
    def should_validate(self) -> bool:
        """Check if URL validation should run."""
        return self.should_run_stage(ProcessingStages.VALIDATION)

    @property
    def should_extract_content(self) -> bool:
        """Check if content extraction should run."""
        return self.should_run_stage(ProcessingStages.CONTENT)

    @property
    def should_run_ai(self) -> bool:
        """Check if AI processing should run."""
        return self.should_run_stage(ProcessingStages.AI)

    @property
    def should_optimize_tags(self) -> bool:
        """Check if tag optimization should run."""
        return self.should_run_stage(ProcessingStages.TAGS)

    @property
    def should_organize_folders(self) -> bool:
        """Check if folder organization should run."""
        return self.should_run_stage(ProcessingStages.FOLDERS)

    def get_description(self) -> str:
        """
        Get a human-readable description of this processing mode.

        Returns:
            Description string
        """
        parts = []

        # Mode type
        if self.dry_run:
            parts.append("Dry-run mode")
        elif self.is_preview:
            parts.append(f"Preview mode ({self.preview_count} items)")
        else:
            parts.append("Full processing")

        # Stages
        stage_list = self.stages.stage_list
        if len(stage_list) == 5:  # All stages
            parts.append("all stages enabled")
        elif stage_list:
            parts.append(f"stages: {', '.join(stage_list)}")
        else:
            parts.append("no stages enabled")

        return " - ".join(parts)

    @classmethod
    def from_cli_args(cls, args: Dict[str, Any]) -> "ProcessingMode":
        """
        Create a ProcessingMode from CLI arguments.

        Args:
            args: Dictionary of CLI arguments with keys like:
                 - preview: int or None
                 - dry_run: bool
                 - skip_validation: bool
                 - skip_ai: bool
                 - skip_content: bool
                 - tags_only: bool
                 - folders_only: bool
                 - validate_only: bool
                 - stages: List[str] (explicit stage list)
                 - verbose: bool
                 - continue_on_error: bool

        Returns:
            ProcessingMode configured from the arguments
        """
        # Start with all stages
        stages = ProcessingStages.VALIDATION | ProcessingStages.CONTENT | ProcessingStages.AI | ProcessingStages.TAGS | ProcessingStages.FOLDERS

        # Handle exclusive modes first
        if args.get("tags_only"):
            stages = ProcessingStages.TAGS
        elif args.get("folders_only"):
            stages = ProcessingStages.FOLDERS
        elif args.get("validate_only"):
            stages = ProcessingStages.VALIDATION
        elif args.get("stages"):
            # Explicit stage list
            stages = ProcessingStages.from_list(args["stages"])
        else:
            # Handle skip flags
            if args.get("skip_validation"):
                stages = stages.without(ProcessingStages.VALIDATION)
            if args.get("skip_ai"):
                stages = stages.without(ProcessingStages.AI)
            if args.get("skip_content"):
                stages = stages.without(ProcessingStages.CONTENT)
            if args.get("skip_tags"):
                stages = stages.without(ProcessingStages.TAGS)
            if args.get("skip_folders"):
                stages = stages.without(ProcessingStages.FOLDERS)

        return cls(
            stages=stages,
            preview_count=args.get("preview"),
            dry_run=args.get("dry_run", False),
            verbose=args.get("verbose", False),
            continue_on_error=args.get("continue_on_error", True),
        )

    @classmethod
    def preview(cls, count: int = 10) -> "ProcessingMode":
        """
        Create a preview mode configuration.

        Args:
            count: Number of items to preview

        Returns:
            ProcessingMode configured for preview
        """
        return cls(preview_count=count)

    @classmethod
    def dry_run_mode(cls) -> "ProcessingMode":
        """
        Create a dry-run mode configuration.

        Returns:
            ProcessingMode configured for dry-run
        """
        return cls(dry_run=True)

    @classmethod
    def tags_only_mode(cls) -> "ProcessingMode":
        """
        Create a tags-only mode configuration.

        Returns:
            ProcessingMode configured for tags only
        """
        return cls(stages=ProcessingStages.TAGS)

    @classmethod
    def validation_only_mode(cls) -> "ProcessingMode":
        """
        Create a validation-only mode configuration.

        Returns:
            ProcessingMode configured for validation only
        """
        return cls(stages=ProcessingStages.VALIDATION)

    @classmethod
    def no_ai_mode(cls) -> "ProcessingMode":
        """
        Create a mode with all stages except AI.

        Returns:
            ProcessingMode with AI disabled
        """
        stages = ProcessingStages.VALIDATION | ProcessingStages.CONTENT | ProcessingStages.TAGS | ProcessingStages.FOLDERS
        return cls(stages=stages)

    def copy(self, **overrides) -> "ProcessingMode":
        """
        Create a copy of this mode with optional overrides.

        Args:
            **overrides: Fields to override

        Returns:
            New ProcessingMode instance
        """
        return ProcessingMode(
            stages=overrides.get("stages", self.stages),
            preview_count=overrides.get("preview_count", self.preview_count),
            dry_run=overrides.get("dry_run", self.dry_run),
            verbose=overrides.get("verbose", self.verbose),
            continue_on_error=overrides.get("continue_on_error", self.continue_on_error),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary with mode configuration
        """
        return {
            "stages": self.stages.stage_list,
            "preview_count": self.preview_count,
            "dry_run": self.dry_run,
            "verbose": self.verbose,
            "continue_on_error": self.continue_on_error,
            "is_preview": self.is_preview,
            "is_full_run": self.is_full_run,
            "will_write_output": self.will_write_output,
        }


# Helper function to work around dataclass default_factory limitations
def _get_all_stages() -> ProcessingStages:
    """Get all processing stages."""
    return (
        ProcessingStages.VALIDATION
        | ProcessingStages.CONTENT
        | ProcessingStages.AI
        | ProcessingStages.TAGS
        | ProcessingStages.FOLDERS
    )


# Predefined mode configurations
PROCESSING_MODES: Dict[str, ProcessingMode] = {
    "full": ProcessingMode(),
    "preview": ProcessingMode.preview(10),
    "dry_run": ProcessingMode.dry_run_mode(),
    "tags_only": ProcessingMode.tags_only_mode(),
    "validation_only": ProcessingMode.validation_only_mode(),
    "no_ai": ProcessingMode.no_ai_mode(),
}


def get_predefined_mode(name: str) -> ProcessingMode:
    """
    Get a predefined processing mode by name.

    Args:
        name: Mode name (full, preview, dry_run, tags_only, validation_only, no_ai)

    Returns:
        ProcessingMode for the specified name

    Raises:
        ValueError: If the mode name is not recognized
    """
    if name.lower() not in PROCESSING_MODES:
        raise ValueError(
            f"Unknown mode: {name}. Valid modes: {list(PROCESSING_MODES.keys())}"
        )
    return PROCESSING_MODES[name.lower()].copy()
