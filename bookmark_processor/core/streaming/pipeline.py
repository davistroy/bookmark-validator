"""
Streaming Pipeline.

Provides streaming pipeline execution for processing large bookmark
collections with minimal memory footprint.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..data_models import Bookmark, ProcessingStatus
from ..data_sources.state_tracker import ProcessingStateTracker
from .reader import StreamingBookmarkReader
from .writer import StreamingBookmarkWriter


@dataclass
class StreamingPipelineConfig:
    """Configuration for streaming pipeline execution."""

    # I/O
    input_file: Union[str, Path]
    output_file: Union[str, Path]

    # Batch processing
    batch_size: int = 100
    flush_interval: int = 10

    # Validation
    url_timeout: float = 30.0
    max_concurrent_requests: int = 10
    verify_ssl: bool = True

    # AI processing
    ai_enabled: bool = True
    max_description_length: int = 150

    # Tag generation
    target_tag_count: int = 150
    max_tags_per_bookmark: int = 5

    # State tracking
    use_state_tracker: bool = True
    state_db_path: Optional[Union[str, Path]] = None

    # Checkpointing
    checkpoint_interval: int = 50
    checkpoint_dir: str = ".bookmark_checkpoints"

    # Progress
    progress_callback: Optional[Callable[[str, int, int], None]] = None
    verbose: bool = False


@dataclass
class ProcessingStats:
    """Statistics for streaming pipeline execution."""

    total_read: int = 0
    total_processed: int = 0
    total_written: int = 0
    total_skipped: int = 0
    total_errors: int = 0

    # Stage-specific counts
    validation_success: int = 0
    validation_failed: int = 0
    content_extracted: int = 0
    ai_processed: int = 0
    tags_generated: int = 0

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def processing_time(self) -> timedelta:
        """Get total processing time."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return datetime.now() - self.start_time
        return timedelta(0)

    @property
    def success_rate(self) -> float:
        """Get processing success rate."""
        if self.total_read == 0:
            return 0.0
        return (self.total_processed / self.total_read) * 100

    @property
    def throughput(self) -> float:
        """Get bookmarks per second."""
        seconds = self.processing_time.total_seconds()
        if seconds == 0:
            return 0.0
        return self.total_processed / seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_read": self.total_read,
            "total_processed": self.total_processed,
            "total_written": self.total_written,
            "total_skipped": self.total_skipped,
            "total_errors": self.total_errors,
            "validation_success": self.validation_success,
            "validation_failed": self.validation_failed,
            "content_extracted": self.content_extracted,
            "ai_processed": self.ai_processed,
            "tags_generated": self.tags_generated,
            "processing_time_seconds": self.processing_time.total_seconds(),
            "success_rate": self.success_rate,
            "throughput": self.throughput,
            "error_count": len(self.errors),
        }


@dataclass
class StreamingPipelineResults:
    """Results of streaming pipeline execution."""

    stats: ProcessingStats
    config: StreamingPipelineConfig
    completed: bool = False
    error_message: Optional[str] = None

    @property
    def total_bookmarks(self) -> int:
        return self.stats.total_read

    @property
    def valid_bookmarks(self) -> int:
        return self.stats.validation_success

    @property
    def invalid_bookmarks(self) -> int:
        return self.stats.validation_failed

    @property
    def processed_bookmarks(self) -> int:
        return self.stats.total_processed

    @property
    def processing_time(self) -> float:
        return self.stats.processing_time.total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "completed": self.completed,
            "error_message": self.error_message,
            "input_file": str(self.config.input_file),
            "output_file": str(self.config.output_file),
            "stats": self.stats.to_dict(),
        }


class StreamingPipeline:
    """
    Process bookmarks in a streaming fashion.

    This pipeline processes bookmarks without loading all data into memory,
    enabling processing of very large datasets (100k+ bookmarks).

    The pipeline:
    1. Reads bookmarks in batches using StreamingBookmarkReader
    2. Processes each batch through validation, content analysis, AI, and tagging
    3. Writes results incrementally using StreamingBookmarkWriter
    4. Optionally tracks state for incremental processing

    Example:
        >>> config = StreamingPipelineConfig(
        ...     input_file="bookmarks.csv",
        ...     output_file="enhanced.csv"
        ... )
        >>> pipeline = StreamingPipeline(config)
        >>> results = pipeline.execute()
        >>> print(f"Processed {results.processed_bookmarks} bookmarks")

        >>> # Or use reader/writer directly
        >>> reader = StreamingBookmarkReader(Path("input.csv"))
        >>> with StreamingBookmarkWriter(Path("output.csv")) as writer:
        ...     results = pipeline.execute_streaming(reader, writer)
    """

    def __init__(
        self,
        config: StreamingPipelineConfig,
        url_validator: Optional[Any] = None,
        content_analyzer: Optional[Any] = None,
        ai_processor: Optional[Any] = None,
        tag_generator: Optional[Any] = None,
        state_tracker: Optional[ProcessingStateTracker] = None,
    ):
        """
        Initialize the streaming pipeline.

        Args:
            config: Pipeline configuration
            url_validator: URL validator component (optional)
            content_analyzer: Content analyzer component (optional)
            ai_processor: AI processor component (optional)
            tag_generator: Tag generator component (optional)
            state_tracker: State tracker for incremental processing (optional)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Components (lazy initialization)
        self._url_validator = url_validator
        self._content_analyzer = content_analyzer
        self._ai_processor = ai_processor
        self._tag_generator = tag_generator
        self._state_tracker = state_tracker

        # Statistics
        self.stats = ProcessingStats()

    def _get_url_validator(self):
        """Lazy initialization of URL validator."""
        if self._url_validator is None:
            from ..url_validator import URLValidator
            self._url_validator = URLValidator(
                timeout=self.config.url_timeout,
                max_concurrent=self.config.max_concurrent_requests,
                verify_ssl=self.config.verify_ssl,
            )
        return self._url_validator

    def _get_content_analyzer(self):
        """Lazy initialization of content analyzer."""
        if self._content_analyzer is None:
            from ..content_analyzer import ContentAnalyzer
            self._content_analyzer = ContentAnalyzer(
                timeout=self.config.url_timeout
            )
        return self._content_analyzer

    def _get_ai_processor(self):
        """Lazy initialization of AI processor."""
        if self._ai_processor is None and self.config.ai_enabled:
            from ..ai_processor import EnhancedAIProcessor
            self._ai_processor = EnhancedAIProcessor(
                max_description_length=self.config.max_description_length
            )
        return self._ai_processor

    def _get_tag_generator(self):
        """Lazy initialization of tag generator."""
        if self._tag_generator is None:
            from ..tag_generator import CorpusAwareTagGenerator
            self._tag_generator = CorpusAwareTagGenerator(
                target_tag_count=self.config.target_tag_count,
                max_tags_per_bookmark=self.config.max_tags_per_bookmark,
            )
        return self._tag_generator

    def _get_state_tracker(self) -> Optional[ProcessingStateTracker]:
        """Lazy initialization of state tracker."""
        if self._state_tracker is None and self.config.use_state_tracker:
            db_path = self.config.state_db_path or ".bookmark_processor_state.db"
            self._state_tracker = ProcessingStateTracker(db_path=db_path)
        return self._state_tracker

    def execute(self) -> StreamingPipelineResults:
        """
        Execute the streaming pipeline.

        Returns:
            StreamingPipelineResults with processing statistics
        """
        reader = StreamingBookmarkReader(self.config.input_file)
        writer = StreamingBookmarkWriter(
            self.config.output_file,
            flush_interval=self.config.flush_interval
        )

        with writer:
            return self.execute_streaming(reader, writer)

    def execute_streaming(
        self,
        reader: StreamingBookmarkReader,
        writer: StreamingBookmarkWriter
    ) -> StreamingPipelineResults:
        """
        Execute pipeline with provided reader and writer.

        This method allows custom reader/writer configurations and
        is useful for testing or advanced use cases.

        Args:
            reader: StreamingBookmarkReader for input
            writer: StreamingBookmarkWriter for output

        Returns:
            StreamingPipelineResults with processing statistics
        """
        self.stats = ProcessingStats()
        self.stats.start_time = datetime.now()

        self.logger.info(f"Starting streaming pipeline: {reader.input_path}")

        try:
            # Get state tracker for incremental processing
            state_tracker = self._get_state_tracker()
            if state_tracker:
                run_id = state_tracker.start_processing_run(
                    source=str(reader.input_path)
                )
                self.logger.info(f"Started processing run {run_id}")

            # Process in batches
            batch_num = 0
            for batch in reader.stream_batches(self.config.batch_size):
                batch_num += 1
                self.logger.debug(f"Processing batch {batch_num}")

                # Filter for unprocessed bookmarks if using state tracker
                if state_tracker:
                    bookmarks_to_process = state_tracker.get_unprocessed(batch)
                    skipped = len(batch) - len(bookmarks_to_process)
                    self.stats.total_skipped += skipped
                else:
                    bookmarks_to_process = batch

                self.stats.total_read += len(batch)

                # Process the batch
                processed_batch = self._process_batch(bookmarks_to_process)

                # Write processed bookmarks
                written = writer.write_batch(processed_batch)
                self.stats.total_written += written

                # Mark as processed in state tracker
                if state_tracker:
                    for bookmark in processed_batch:
                        state_tracker.mark_processed(
                            bookmark,
                            ai_engine="local" if self.config.ai_enabled else "none"
                        )

                # Progress callback
                if self.config.progress_callback:
                    total_estimate = reader.total_count or self.stats.total_read
                    self.config.progress_callback(
                        f"Batch {batch_num}",
                        self.stats.total_processed,
                        total_estimate
                    )

                # Checkpoint periodically
                if batch_num % (self.config.checkpoint_interval // self.config.batch_size + 1) == 0:
                    self._save_checkpoint()

            # Complete processing run
            if state_tracker:
                state_tracker.complete_processing_run(
                    total_processed=self.stats.total_processed,
                    total_succeeded=self.stats.validation_success,
                    total_failed=self.stats.validation_failed
                )

            self.stats.end_time = datetime.now()

            self.logger.info(
                f"Streaming pipeline complete: {self.stats.total_processed} processed, "
                f"{self.stats.total_written} written in "
                f"{self.stats.processing_time.total_seconds():.2f}s"
            )

            return StreamingPipelineResults(
                stats=self.stats,
                config=self.config,
                completed=True
            )

        except Exception as e:
            self.stats.end_time = datetime.now()
            self.logger.error(f"Pipeline execution failed: {e}")

            return StreamingPipelineResults(
                stats=self.stats,
                config=self.config,
                completed=False,
                error_message=str(e)
            )

    def _process_batch(self, bookmarks: List[Bookmark]) -> List[Bookmark]:
        """
        Process a batch of bookmarks through all pipeline stages.

        Args:
            bookmarks: List of bookmarks to process

        Returns:
            List of processed bookmarks
        """
        if not bookmarks:
            return []

        processed = []

        for bookmark in bookmarks:
            try:
                # Stage 1: URL Validation
                validated = self._validate_url(bookmark)
                if not validated:
                    self.stats.validation_failed += 1
                    continue

                self.stats.validation_success += 1

                # Stage 2: Content Analysis (optional based on validation)
                content_data = self._analyze_content(bookmark)
                if content_data:
                    self.stats.content_extracted += 1

                # Stage 3: AI Processing (if enabled)
                if self.config.ai_enabled:
                    ai_result = self._process_ai(bookmark, content_data)
                    if ai_result:
                        self.stats.ai_processed += 1

                # Stage 4: Tag Generation
                tags = self._generate_tags(bookmark, content_data)
                if tags:
                    bookmark.optimized_tags = tags
                    self.stats.tags_generated += 1

                processed.append(bookmark)
                self.stats.total_processed += 1

            except Exception as e:
                self.stats.total_errors += 1
                self.stats.errors.append({
                    "url": bookmark.url,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                self.logger.debug(f"Error processing {bookmark.url}: {e}")

        return processed

    def _validate_url(self, bookmark: Bookmark) -> bool:
        """
        Validate a bookmark's URL.

        Args:
            bookmark: Bookmark to validate

        Returns:
            True if URL is valid
        """
        if not bookmark.url:
            return False

        try:
            validator = self._get_url_validator()
            if validator:
                result = validator.validate_url(bookmark.url)
                bookmark.processing_status.url_validated = True
                bookmark.processing_status.url_validation_error = (
                    result.error_message if not result.is_valid else None
                )
                return result.is_valid
            return True  # No validator, assume valid

        except Exception as e:
            bookmark.processing_status.url_validation_error = str(e)
            return False

    def _analyze_content(self, bookmark: Bookmark) -> Optional[Dict[str, Any]]:
        """
        Analyze content for a bookmark.

        Args:
            bookmark: Bookmark to analyze

        Returns:
            Content data dictionary or None
        """
        try:
            analyzer = self._get_content_analyzer()
            if analyzer:
                content_data = analyzer.analyze_content(
                    bookmark.url,
                    existing_title=bookmark.title or "",
                    existing_note=bookmark.note or "",
                    existing_excerpt=bookmark.excerpt or "",
                )
                bookmark.processing_status.content_extracted = True
                return content_data
            return None

        except Exception as e:
            bookmark.processing_status.content_extraction_error = str(e)
            return None

    def _process_ai(
        self,
        bookmark: Bookmark,
        content_data: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Process AI description generation.

        Args:
            bookmark: Bookmark to process
            content_data: Optional content data from analysis

        Returns:
            Enhanced description or None
        """
        try:
            processor = self._get_ai_processor()
            if processor:
                # Prepare content for AI
                content = ""
                if content_data:
                    content = content_data.get("content", "") or ""

                result = processor.process_single(
                    bookmark,
                    content=content
                )

                if result and result.enhanced_description:
                    bookmark.enhanced_description = result.enhanced_description
                    bookmark.processing_status.ai_processed = True
                    return result.enhanced_description

            return None

        except Exception as e:
            bookmark.processing_status.ai_processing_error = str(e)
            return None

    def _generate_tags(
        self,
        bookmark: Bookmark,
        content_data: Optional[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate tags for a bookmark.

        Args:
            bookmark: Bookmark to generate tags for
            content_data: Optional content data

        Returns:
            List of generated tags
        """
        try:
            generator = self._get_tag_generator()
            if generator:
                # Use single bookmark tag generation
                tags = generator.generate_for_single_bookmark(
                    bookmark,
                    content_data=content_data
                )
                bookmark.processing_status.tags_optimized = True
                return tags

            return bookmark.tags  # Return original tags

        except Exception as e:
            self.logger.debug(f"Tag generation error: {e}")
            return bookmark.tags

    def _save_checkpoint(self) -> None:
        """Save processing checkpoint."""
        # Checkpoint is handled by state tracker in this implementation
        self.logger.debug(f"Checkpoint: {self.stats.total_processed} processed")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current processing statistics.

        Returns:
            Dictionary with statistics
        """
        return self.stats.to_dict()

    def close(self) -> None:
        """Clean up resources."""
        if self._url_validator and hasattr(self._url_validator, "close"):
            self._url_validator.close()
        if self._content_analyzer and hasattr(self._content_analyzer, "close"):
            self._content_analyzer.close()
        if self._ai_processor and hasattr(self._ai_processor, "close"):
            self._ai_processor.close()

        self.logger.debug("Pipeline resources cleaned up")

    def __repr__(self) -> str:
        return (
            f"StreamingPipeline(input={self.config.input_file}, "
            f"output={self.config.output_file})"
        )
