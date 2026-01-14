"""
Pipeline Configuration and Results

Defines configuration dataclasses and results structures for the
bookmark processing pipeline.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...utils.progress_tracker import ProgressLevel


@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline"""

    # Input/Output
    input_file: str
    output_file: str

    # Output format options
    generate_chrome_html: bool = False
    chrome_html_output: Optional[str] = None
    output_title: str = "Bookmarks"

    # Processing options
    batch_size: int = 100
    max_retries: int = 3
    resume_enabled: bool = True
    clear_checkpoints: bool = False

    # Validation settings
    url_timeout: float = 30.0
    max_concurrent_requests: int = 10
    verify_ssl: bool = True

    # AI processing
    ai_enabled: bool = True
    max_description_length: int = 150

    # Tag generation
    target_tag_count: int = 150
    max_tags_per_bookmark: int = 5

    # Folder generation
    generate_folders: bool = True
    max_bookmarks_per_folder: int = 20
    ai_engine: str = "local"  # For folder generation AI

    # Memory management
    memory_batch_size: int = 100
    memory_warning_threshold: float = 3000  # MB
    memory_critical_threshold: float = 3500  # MB

    # Duplicate detection
    detect_duplicates: bool = True
    duplicate_strategy: str = (
        "highest_quality"  # newest, oldest, most_complete, highest_quality
    )

    # Progress reporting
    verbose: bool = False
    progress_level: ProgressLevel = ProgressLevel.STANDARD

    # Checkpoint settings
    checkpoint_dir: str = ".bookmark_checkpoints"
    save_interval: int = 50


@dataclass
class PipelineResults:
    """Results of pipeline execution"""

    total_bookmarks: int
    valid_bookmarks: int
    invalid_bookmarks: int
    ai_processed: int
    tagged_bookmarks: int
    unique_tags: int
    processing_time: float
    stages_completed: List[str]
    error_summary: Dict[str, int]
    statistics: Dict[str, Any]


__all__ = ["PipelineConfig", "PipelineResults"]
