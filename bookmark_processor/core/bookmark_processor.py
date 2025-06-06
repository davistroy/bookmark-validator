"""
Main application controller for the Bookmark Processor.

This module provides the main BookmarkProcessor class that orchestrates
all the components of the bookmark processing pipeline using the new
comprehensive pipeline architecture.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from bookmark_processor.config.configuration import Configuration

from ..utils.progress_tracker import ProgressLevel
from .data_models import Bookmark
from .multi_file_processor import MultiFileProcessor
from .pipeline import BookmarkProcessingPipeline, PipelineConfig, PipelineResults


class ProcessingResults:
    """Container for processing results and statistics."""

    def __init__(self, pipeline_results: PipelineResults = None):
        if pipeline_results:
            self.total_bookmarks = pipeline_results.total_bookmarks
            self.processed_bookmarks = pipeline_results.valid_bookmarks
            self.valid_bookmarks = pipeline_results.valid_bookmarks
            self.invalid_bookmarks = pipeline_results.invalid_bookmarks
            self.ai_processed = pipeline_results.ai_processed
            self.tagged_bookmarks = pipeline_results.tagged_bookmarks
            self.unique_tags = pipeline_results.unique_tags
            self.processing_time = pipeline_results.processing_time
            self.stages_completed = pipeline_results.stages_completed
            self.error_summary = pipeline_results.error_summary
            self.statistics = pipeline_results.statistics
            self.errors = []
        else:
            self.total_bookmarks = 0
            self.processed_bookmarks = 0
            self.valid_bookmarks = 0
            self.invalid_bookmarks = 0
            self.ai_processed = 0
            self.tagged_bookmarks = 0
            self.unique_tags = 0
            self.errors = []
            self.processing_time = 0.0
            self.stages_completed = []
            self.error_summary = {}
            self.statistics = {}

    def __str__(self) -> str:
        return (
            f"ProcessingResults("
            f"total={self.total_bookmarks}, "
            f"valid={self.valid_bookmarks}, "
            f"invalid={self.invalid_bookmarks}, "
            f"ai_processed={self.ai_processed}, "
            f"tagged={self.tagged_bookmarks}, "
            f"unique_tags={self.unique_tags}, "
            f"errors={len(self.errors)}, "
            f"time={self.processing_time:.2f}s)"
        )


class BookmarkProcessor:
    """Main application controller with checkpoint/resume capability."""

    def __init__(self, config: Configuration):
        """
        Initialize the bookmark processor.

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Components will be initialized when needed
        self.checkpoint_manager = None
        self.progress_tracker = None

    def process_bookmarks(
        self, input_file: Path, output_file: Path, resume: bool = True, **kwargs
    ) -> ProcessingResults:
        """
        Main processing pipeline with checkpoint support.

        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            resume: Whether to resume from existing checkpoint
            **kwargs: Additional pipeline configuration options

        Returns:
            ProcessingResults object with statistics
        """
        self.logger.info("Starting comprehensive bookmark processing pipeline")
        self.logger.info(f"Input: {input_file}")
        self.logger.info(f"Output: {output_file}")
        self.logger.info(f"Resume: {resume}")

        try:
            # Create pipeline configuration
            pipeline_config = PipelineConfig(
                input_file=str(input_file),
                output_file=str(output_file),
                resume_enabled=resume,
                verbose=kwargs.get("verbose", False),
                batch_size=kwargs.get("batch_size", 100),
                max_retries=kwargs.get("max_retries", 3),
                clear_checkpoints=kwargs.get("clear_checkpoints", False),
                detect_duplicates=kwargs.get("detect_duplicates", True),
                duplicate_strategy=kwargs.get("duplicate_strategy", "highest_quality"),
                progress_level=(
                    ProgressLevel.DETAILED
                    if kwargs.get("verbose", False)
                    else ProgressLevel.STANDARD
                ),
            )

            # Create and execute pipeline
            pipeline = BookmarkProcessingPipeline(pipeline_config)

            def progress_callback(message: str):
                if pipeline_config.verbose:
                    self.logger.info(message)

            pipeline_results = pipeline.execute(progress_callback=progress_callback)

            # Convert to legacy format
            results = ProcessingResults(pipeline_results)

            self.logger.info("Processing completed successfully")
            self.logger.info(f"Results: {results}")

            return results

        except Exception as e:
            self.logger.exception("Error during bookmark processing")
            results = ProcessingResults()
            results.errors.append(str(e))
            raise

    def resume_processing(self) -> ProcessingResults:
        """
        Resume from existing checkpoint.

        Returns:
            ProcessingResults object with statistics
        """
        self.logger.info("Resuming processing from checkpoint")
        # Will be implemented when checkpoint manager is ready
        return ProcessingResults()

    def _run_auto_detection_mode(self, validated_args: Dict[str, Any]) -> int:
        """
        Handle auto-detection mode for multiple files.

        Args:
            validated_args: Dictionary of validated CLI arguments

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            multi_processor = MultiFileProcessor()

            # Auto-detect files in current directory
            detected_files = multi_processor.auto_detect_files()

            if not detected_files:
                print("\n✗ No bookmark files found in current directory")
                print("  Supported formats: CSV (.csv), HTML (.html, .htm)")
                return 1

            print(f"\n✓ Auto-detected {len(detected_files)} bookmark file(s):")
            for file_path in detected_files:
                print(f"  - {file_path}")

            # Process multiple files
            all_bookmarks, processing_stats = multi_processor.process_multiple_files(
                detected_files, merge_strategy="combine"
            )

            if not all_bookmarks:
                print("\n✗ No valid bookmarks found in detected files")
                return 1

            print(
                f"\n✓ Combined {len(all_bookmarks)} bookmarks from {len(detected_files)} files"
            )

            # Generate output filename with timestamp for combined processing
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = validated_args["output_path"] or Path(
                f"combined_bookmarks_{timestamp}.csv"
            )

            # Create a temporary combined file and process it
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)

                # Write combined bookmarks to temporary file in export format
                self._save_bookmarks_as_export_csv(all_bookmarks, temp_path)

            try:
                # Process the combined file
                results = self.process_bookmarks(
                    input_file=temp_path,
                    output_file=output_file,
                    resume=validated_args["resume"]
                    and not validated_args["clear_checkpoints"],
                    verbose=validated_args.get("verbose", False),
                    batch_size=validated_args.get("batch_size", 100),
                    max_retries=validated_args.get("max_retries", 3),
                    clear_checkpoints=validated_args.get("clear_checkpoints", False),
                    detect_duplicates=validated_args.get("detect_duplicates", True),
                    duplicate_strategy=validated_args.get(
                        "duplicate_strategy", "highest_quality"
                    ),
                )

                # Print comprehensive results summary for multi-file processing
                print("\n✓ Multi-file processing completed successfully!")
                print(f"  Source files: {len(detected_files)}")
                print(f"  Total bookmarks: {results.total_bookmarks}")
                print(f"  Valid bookmarks: {results.valid_bookmarks}")
                print(f"  Invalid bookmarks: {results.invalid_bookmarks}")
                print(f"  AI processed: {results.ai_processed}")
                print(f"  Tagged bookmarks: {results.tagged_bookmarks}")
                print(f"  Unique tags: {results.unique_tags}")
                print(f"  Processing time: {results.processing_time:.2f}s")
                print(f"  Output file: {output_file}")

                if processing_stats.get("file_errors"):
                    print(
                        f"  File processing errors: {len(processing_stats['file_errors'])}"
                    )
                    for file_path, error in processing_stats["file_errors"].items():
                        print(f"    {file_path}: {error}")

                return 0

            finally:
                # Clean up temporary file
                temp_path.unlink(missing_ok=True)

        except Exception as e:
            print(f"\n✗ Auto-detection processing failed: {e}")
            self.logger.exception("Auto-detection processing failed")
            return 1

    def _save_bookmarks_as_export_csv(
        self, bookmarks: List[Bookmark], file_path: Path
    ) -> None:
        """
        Save bookmarks as a raindrop.io export CSV file (11 columns).

        Args:
            bookmarks: List of bookmarks to save
            file_path: Path to save the CSV file
        """
        import pandas as pd

        # Convert bookmarks to export format (11 columns)
        export_data = []
        for i, bookmark in enumerate(bookmarks, 1):
            export_data.append(
                {
                    "id": i,  # Generate sequential IDs
                    "title": bookmark.title or "",
                    "note": bookmark.note or "",
                    "excerpt": bookmark.excerpt or "",
                    "url": bookmark.url,
                    "folder": bookmark.folder or "",
                    "tags": ", ".join(bookmark.tags) if bookmark.tags else "",
                    "created": bookmark.created.isoformat() if bookmark.created else "",
                    "cover": "",  # Not used
                    "highlights": "",  # Not used
                    "favorite": (
                        str(bookmark.favorite).lower()
                        if hasattr(bookmark, "favorite")
                        else "false"
                    ),
                }
            )

        # Create DataFrame and save
        df = pd.DataFrame(export_data)
        df.to_csv(file_path, index=False, encoding="utf-8")

    def run_cli(self, validated_args: Dict[str, Any]) -> int:
        """
        CLI entry point for bookmark processing.

        Args:
            validated_args: Dictionary of validated CLI arguments

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Handle auto-detection mode when no input file specified
            if validated_args["input_path"] is None:
                return self._run_auto_detection_mode(validated_args)

            # Single file processing mode
            results = self.process_bookmarks(
                input_file=validated_args["input_path"],
                output_file=validated_args["output_path"],
                resume=validated_args["resume"]
                and not validated_args["clear_checkpoints"],
                verbose=validated_args.get("verbose", False),
                batch_size=validated_args.get("batch_size", 100),
                max_retries=validated_args.get("max_retries", 3),
                clear_checkpoints=validated_args.get("clear_checkpoints", False),
                detect_duplicates=validated_args.get("detect_duplicates", True),
                duplicate_strategy=validated_args.get(
                    "duplicate_strategy", "highest_quality"
                ),
            )

            # Print comprehensive results summary
            print("\n✓ Processing completed successfully!")
            print(f"  Total bookmarks: {results.total_bookmarks}")
            print(f"  Valid bookmarks: {results.valid_bookmarks}")
            print(f"  Invalid bookmarks: {results.invalid_bookmarks}")
            print(f"  AI processed: {results.ai_processed}")
            print(f"  Tagged bookmarks: {results.tagged_bookmarks}")
            print(f"  Unique tags: {results.unique_tags}")
            print(f"  Processing time: {results.processing_time:.2f}s")
            print(f"  Stages completed: {', '.join(results.stages_completed)}")

            if results.error_summary:
                print("  Error summary:")
                for error_type, count in results.error_summary.items():
                    print(f"    {error_type}: {count}")

            if results.errors:
                print(f"  Critical errors: {len(results.errors)}")
                self.logger.warning(
                    f"Processing completed with {len(results.errors)} critical errors"
                )

            return 0

        except Exception as e:
            print(f"\n✗ Processing failed: {e}")
            self.logger.exception("Processing failed")
            return 1
