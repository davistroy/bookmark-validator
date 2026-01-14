"""
Pipeline Factory

Factory methods for creating pipeline instances with dependency injection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PipelineConfig
    from .pipeline import BookmarkProcessingPipeline


class PipelineFactory:
    """Factory for creating processing pipelines with dependency injection."""

    @staticmethod
    def create(config: "PipelineConfig") -> "BookmarkProcessingPipeline":
        """
        Create a processing pipeline with all default components.

        This factory method instantiates all components with their default
        configurations based on the provided PipelineConfig. Use this when
        you want standard pipeline behavior.

        Args:
            config: Pipeline configuration

        Returns:
            Configured BookmarkProcessingPipeline with all default components
        """
        from ...utils.intelligent_rate_limiter import IntelligentRateLimiter
        from ...utils.memory_optimizer import BatchProcessor, MemoryMonitor
        from ..ai_processor import EnhancedAIProcessor
        from ..checkpoint_manager import CheckpointManager
        from ..chrome_html_generator import ChromeHTMLGenerator
        from ..content_analyzer import ContentAnalyzer
        from ..csv_handler import RaindropCSVHandler
        from ..duplicate_detector import DuplicateDetector
        from ..folder_generator import AIFolderGenerator
        from ..import_module import MultiFormatImporter
        from ..tag_generator import CorpusAwareTagGenerator
        from ..url_validator import URLValidator
        from .pipeline import BookmarkProcessingPipeline

        # Create core components
        csv_handler = RaindropCSVHandler()
        multi_importer = MultiFormatImporter()

        # Create rate limiter
        rate_limiter = IntelligentRateLimiter(
            max_concurrent=config.max_concurrent_requests
        )

        # Create URL validator with rate limiter
        url_validator = URLValidator(
            timeout=config.url_timeout,
            max_concurrent=config.max_concurrent_requests,
            verify_ssl=config.verify_ssl,
            rate_limiter=rate_limiter,
        )

        # Create content analyzer
        content_analyzer = ContentAnalyzer(timeout=config.url_timeout)

        # Create AI processor if enabled
        ai_processor = (
            EnhancedAIProcessor(max_description_length=config.max_description_length)
            if config.ai_enabled
            else None
        )

        # Create tag generator
        tag_generator = CorpusAwareTagGenerator(
            target_tag_count=config.target_tag_count,
            max_tags_per_bookmark=config.max_tags_per_bookmark,
        )

        # Create duplicate detector if enabled
        duplicate_detector = (
            DuplicateDetector() if config.detect_duplicates else None
        )

        # Create folder generator if enabled
        folder_generator = (
            AIFolderGenerator(
                max_bookmarks_per_folder=config.max_bookmarks_per_folder,
                ai_engine=config.ai_engine,
            )
            if config.generate_folders
            else None
        )

        # Create Chrome HTML generator if enabled
        chrome_html_generator = (
            ChromeHTMLGenerator() if config.generate_chrome_html else None
        )

        # Create memory management components
        memory_monitor = MemoryMonitor(
            warning_threshold_mb=config.memory_warning_threshold,
            critical_threshold_mb=config.memory_critical_threshold,
        )

        batch_processor = BatchProcessor(
            batch_size=config.memory_batch_size, memory_monitor=memory_monitor
        )

        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir, save_interval=config.save_interval
        )

        # Create and return pipeline with all components
        return BookmarkProcessingPipeline(
            config=config,
            csv_handler=csv_handler,
            multi_importer=multi_importer,
            rate_limiter=rate_limiter,
            url_validator=url_validator,
            content_analyzer=content_analyzer,
            ai_processor=ai_processor,
            tag_generator=tag_generator,
            duplicate_detector=duplicate_detector,
            folder_generator=folder_generator,
            chrome_html_generator=chrome_html_generator,
            memory_monitor=memory_monitor,
            batch_processor=batch_processor,
            checkpoint_manager=checkpoint_manager,
        )

    @staticmethod
    def create_with_custom_components(
        config: "PipelineConfig", **components
    ) -> "BookmarkProcessingPipeline":
        """
        Create a pipeline with custom components.

        This factory method allows you to override specific components while
        using defaults for others. Useful for testing or custom configurations.

        Args:
            config: Pipeline configuration
            **components: Custom component instances (e.g., url_validator=mock_validator)

        Returns:
            Configured BookmarkProcessingPipeline with mixed default and custom components

        Example:
            >>> mock_validator = Mock(spec=URLValidator)
            >>> pipeline = PipelineFactory.create_with_custom_components(
            ...     config, url_validator=mock_validator
            ... )
        """
        from .pipeline import BookmarkProcessingPipeline

        # Create pipeline with defaults, then override with custom components
        default_pipeline = PipelineFactory.create(config)

        # Extract components from default pipeline or use custom ones
        return BookmarkProcessingPipeline(
            config=config,
            csv_handler=components.get("csv_handler", default_pipeline.csv_handler),
            multi_importer=components.get(
                "multi_importer", default_pipeline.multi_importer
            ),
            rate_limiter=components.get("rate_limiter", default_pipeline.rate_limiter),
            url_validator=components.get(
                "url_validator", default_pipeline.url_validator
            ),
            content_analyzer=components.get(
                "content_analyzer", default_pipeline.content_analyzer
            ),
            ai_processor=components.get("ai_processor", default_pipeline.ai_processor),
            tag_generator=components.get(
                "tag_generator", default_pipeline.tag_generator
            ),
            duplicate_detector=components.get(
                "duplicate_detector", default_pipeline.duplicate_detector
            ),
            folder_generator=components.get(
                "folder_generator", default_pipeline.folder_generator
            ),
            chrome_html_generator=components.get(
                "chrome_html_generator", default_pipeline.chrome_html_generator
            ),
            memory_monitor=components.get(
                "memory_monitor", default_pipeline.memory_monitor
            ),
            batch_processor=components.get(
                "batch_processor", default_pipeline.batch_processor
            ),
            checkpoint_manager=components.get(
                "checkpoint_manager", default_pipeline.checkpoint_manager
            ),
        )


def create_pipeline(
    input_file: str, output_file: str, **kwargs
) -> "BookmarkProcessingPipeline":
    """
    Create a processing pipeline with standard configuration.

    This is a convenience function that creates a PipelineConfig and uses
    the PipelineFactory to instantiate a pipeline with all default components.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        **kwargs: Additional configuration options

    Returns:
        Configured BookmarkProcessingPipeline

    Example:
        >>> pipeline = create_pipeline("input.csv", "output.csv", batch_size=50)
    """
    from .config import PipelineConfig

    config = PipelineConfig(input_file=input_file, output_file=output_file, **kwargs)
    return PipelineFactory.create(config)


__all__ = ["PipelineFactory", "create_pipeline"]
