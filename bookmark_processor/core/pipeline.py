"""
Main Processing Pipeline

Orchestrates all components to process bookmark collections from raindrop.io export
to enhanced import format with validation, AI descriptions, and optimized tagging.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .data_models import Bookmark
from .csv_handler import RaindropCSVHandler
from .url_validator import URLValidator, ValidationResult
from .content_analyzer import ContentAnalyzer, ContentData
from .ai_processor import EnhancedAIProcessor, AIProcessingResult
from .tag_generator import CorpusAwareTagGenerator, TagOptimizationResult
from .checkpoint_manager import CheckpointManager, ProcessingStage, ProcessingState
from ..utils.progress_tracker import AdvancedProgressTracker, ProgressLevel, ProcessingStage as PTStage
from ..utils.intelligent_rate_limiter import IntelligentRateLimiter


@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline"""
    # Input/Output
    input_file: str
    output_file: str
    
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


class BookmarkProcessingPipeline:
    """Main processing pipeline orchestrating all components"""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize processing pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Initialize components
        self.csv_handler = RaindropCSVHandler()
        self.rate_limiter = IntelligentRateLimiter(
            max_concurrent=config.max_concurrent_requests
        )
        self.url_validator = URLValidator(
            timeout=config.url_timeout,
            max_concurrent=config.max_concurrent_requests,
            verify_ssl=config.verify_ssl,
            rate_limiter=self.rate_limiter
        )
        self.content_analyzer = ContentAnalyzer(timeout=config.url_timeout)
        self.ai_processor = EnhancedAIProcessor(
            max_description_length=config.max_description_length
        ) if config.ai_enabled else None
        self.tag_generator = CorpusAwareTagGenerator(
            target_tag_count=config.target_tag_count,
            max_tags_per_bookmark=config.max_tags_per_bookmark
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            save_interval=config.save_interval
        )
        
        # Progress tracking
        self.progress_tracker: Optional[AdvancedProgressTracker] = None
        
        # Data storage
        self.bookmarks: List[Bookmark] = []
        self.validation_results: Dict[str, ValidationResult] = {}
        self.content_data: Dict[str, ContentData] = {}
        self.ai_results: Dict[str, AIProcessingResult] = {}
        self.tag_assignments: Dict[str, List[str]] = {}
        
        # Statistics
        self.error_summary: Dict[str, int] = {}
        self.processing_stats: Dict[str, Any] = {}
        
        logging.info(f"Processing pipeline initialized for {config.input_file}")
    
    def execute(self, progress_callback: Optional[Callable[[str], None]] = None) -> PipelineResults:
        """
        Execute the complete processing pipeline.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            PipelineResults with processing statistics
        """
        start_time = time.time()
        
        try:
            # Handle checkpoint resume
            if self.config.resume_enabled and not self.config.clear_checkpoints:
                if self.checkpoint_manager.has_checkpoint(self.config.input_file):
                    return self._resume_processing(progress_callback)
            
            # Clear checkpoints if requested
            if self.config.clear_checkpoints:
                self.checkpoint_manager.clear_checkpoint()
            
            # Start new processing
            return self._start_new_processing(progress_callback)
            
        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}")
            raise
        
        finally:
            processing_time = time.time() - start_time
            logging.info(f"Pipeline execution completed in {processing_time:.2f}s")
            
            # Cleanup
            self._cleanup_resources()
    
    def _start_new_processing(self, progress_callback: Optional[Callable[[str], None]]) -> PipelineResults:
        """Start new processing from beginning"""
        logging.info("Starting new bookmark processing pipeline")
        
        # Stage 1: Load and initialize
        self._stage_load_bookmarks()
        
        # Initialize progress tracking
        self.progress_tracker = AdvancedProgressTracker(
            total_items=len(self.bookmarks),
            description="Processing Bookmarks",
            level=self.config.progress_level
        )
        
        if progress_callback:
            # Store progress callback for manual calls since add_progress_callback doesn't exist
            self._progress_callback = progress_callback
        else:
            self._progress_callback = None
        
        # Initialize checkpoint
        config_dict = self.config.__dict__.copy()
        self.checkpoint_manager.initialize_processing(
            self.config.input_file,
            self.config.output_file,
            len(self.bookmarks),
            config_dict
        )
        
        # Execute pipeline stages
        try:
            self._stage_validate_urls()
            self._stage_analyze_content()
            if self.config.ai_enabled:
                self._stage_ai_processing()
            self._stage_generate_tags()
            self._stage_generate_output()
            
            return self._create_results()
            
        except Exception as e:
            self.checkpoint_manager.update_stage(ProcessingStage.ERROR)
            raise
        
        finally:
            if self.progress_tracker:
                self.progress_tracker.complete()
    
    def _resume_processing(self, progress_callback: Optional[Callable[[str], None]]) -> PipelineResults:
        """Resume processing from checkpoint"""
        logging.info("Resuming from checkpoint")
        
        # Load checkpoint
        state = self.checkpoint_manager.load_checkpoint(self.config.input_file)
        if not state:
            logging.warning("Could not load checkpoint, starting new processing")
            return self._start_new_processing(progress_callback)
        
        # Restore data from checkpoint
        self._restore_from_checkpoint(state)
        
        # Initialize progress tracking
        self.progress_tracker = AdvancedProgressTracker(
            total_items=len(self.bookmarks),
            description="Resuming Processing",
            level=self.config.progress_level
        )
        
        self.progress_tracker.update_progress(items_delta=len(state.processed_urls))
        
        if progress_callback:
            # Store progress callback for manual calls since add_progress_callback doesn't exist
            self._progress_callback = progress_callback
        else:
            self._progress_callback = None
        
        # Continue from current stage
        try:
            if state.current_stage == ProcessingStage.URL_VALIDATION:
                self._stage_validate_urls(resume=True)
            
            if state.current_stage in [ProcessingStage.URL_VALIDATION, ProcessingStage.CONTENT_ANALYSIS]:
                self._stage_analyze_content(resume=True)
            
            if (self.config.ai_enabled and 
                state.current_stage in [ProcessingStage.URL_VALIDATION, 
                                       ProcessingStage.CONTENT_ANALYSIS, 
                                       ProcessingStage.AI_PROCESSING]):
                self._stage_ai_processing(resume=True)
            
            if state.current_stage != ProcessingStage.COMPLETED:
                self._stage_generate_tags()
                self._stage_generate_output()
            
            return self._create_results()
            
        except Exception as e:
            self.checkpoint_manager.update_stage(ProcessingStage.ERROR)
            raise
        
        finally:
            if self.progress_tracker:
                self.progress_tracker.complete()
    
    def _stage_load_bookmarks(self) -> None:
        """Stage 1: Load bookmarks from input file"""
        logging.info("Stage 1: Loading bookmarks")
        
        self.checkpoint_manager.update_stage(ProcessingStage.LOADING)
        
        try:
            # Load CSV data
            df = self.csv_handler.load_export_csv(self.config.input_file)
            
            # Convert to bookmark objects
            self.bookmarks = []
            for _, row in df.iterrows():
                try:
                    bookmark = Bookmark.from_raindrop_export(row.to_dict())
                    self.bookmarks.append(bookmark)
                except Exception as e:
                    logging.warning(f"Failed to create bookmark from row: {e}")
            
            # Remove duplicates
            unique_bookmarks = {}
            for bookmark in self.bookmarks:
                if bookmark.url not in unique_bookmarks:
                    unique_bookmarks[bookmark.url] = bookmark
            
            self.bookmarks = list(unique_bookmarks.values())
            
            logging.info(f"Loaded {len(self.bookmarks)} unique bookmarks")
            
        except Exception as e:
            logging.error(f"Failed to load bookmarks: {e}")
            raise
    
    def _stage_validate_urls(self, resume: bool = False) -> None:
        """Stage 2: Validate bookmark URLs"""
        logging.info("Stage 2: Validating URLs")
        
        self.checkpoint_manager.update_stage(ProcessingStage.URL_VALIDATION)
        
        if self.progress_tracker:
            self.progress_tracker.start_stage(PTStage.VALIDATING_URLS, len(self.bookmarks))
        
        # Get URLs to validate (skip if resuming and already validated)
        if resume:
            state = self.checkpoint_manager.current_state
            processed_urls = state.processed_urls if state else set()
            urls_to_validate = [b.url for b in self.bookmarks if b.url not in processed_urls]
        else:
            urls_to_validate = [b.url for b in self.bookmarks]
        
        if urls_to_validate:
            logging.info(f"Validating {len(urls_to_validate)} URLs")
            
            # Batch validate URLs
            validation_results = self.url_validator.batch_validate(
                urls_to_validate,
                progress_callback=self._validation_progress_callback,
                enable_retries=True
            )
            
            # Store results and update checkpoint
            for result in validation_results:
                self.validation_results[result.url] = result
                
                # Find corresponding bookmark
                bookmark = next((b for b in self.bookmarks if b.url == result.url), None)
                if bookmark:
                    self.checkpoint_manager.add_validated_bookmark(result, bookmark)
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(force=True)
        
        valid_count = sum(1 for r in self.validation_results.values() if r.is_valid)
        logging.info(f"URL validation complete: {valid_count}/{len(self.validation_results)} valid")
    
    def _stage_analyze_content(self, resume: bool = False) -> None:
        """Stage 3: Analyze content for valid URLs"""
        logging.info("Stage 3: Analyzing content")
        
        self.checkpoint_manager.update_stage(ProcessingStage.CONTENT_ANALYSIS)
        
        # Get valid URLs for content analysis
        valid_urls = [url for url, result in self.validation_results.items() if result.is_valid]
        
        if resume:
            state = self.checkpoint_manager.current_state
            analyzed_urls = set(state.content_data.keys()) if state else set()
            urls_to_analyze = [url for url in valid_urls if url not in analyzed_urls]
        else:
            urls_to_analyze = valid_urls
        
        if self.progress_tracker:
            self.progress_tracker.start_stage(PTStage.EXTRACTING_CONTENT, len(urls_to_analyze))
        
        if urls_to_analyze:
            logging.info(f"Analyzing content for {len(urls_to_analyze)} URLs")
            
            for i, url in enumerate(urls_to_analyze):
                try:
                    # Find bookmark
                    bookmark = next((b for b in self.bookmarks if b.url == url), None)
                    if not bookmark:
                        continue
                    
                    # Analyze content
                    content_data = self.content_analyzer.analyze_content(
                        url,
                        existing_title=bookmark.title or "",
                        existing_note=bookmark.note or "",
                        existing_excerpt=bookmark.excerpt or ""
                    )
                    
                    self.content_data[url] = content_data
                    self.checkpoint_manager.add_content_data(url, content_data)
                    
                    # Update progress
                    if self.progress_tracker:
                        self.progress_tracker.update(stage_items=i+1)
                    
                    # Save checkpoint periodically
                    if (i + 1) % self.config.save_interval == 0:
                        self.checkpoint_manager.save_checkpoint()
                    
                except Exception as e:
                    logging.debug(f"Content analysis failed for {url}: {e}")
                    self.error_summary['content_analysis'] = self.error_summary.get('content_analysis', 0) + 1
        
        # Save final checkpoint
        self.checkpoint_manager.save_checkpoint(force=True)
        
        logging.info(f"Content analysis complete: {len(self.content_data)} URLs analyzed")
    
    def _stage_ai_processing(self, resume: bool = False) -> None:
        """Stage 4: AI processing for enhanced descriptions"""
        if not self.ai_processor:
            logging.info("Stage 4: AI processing disabled")
            return
        
        logging.info("Stage 4: AI processing")
        
        self.checkpoint_manager.update_stage(ProcessingStage.AI_PROCESSING)
        
        # Get bookmarks for AI processing
        valid_bookmarks = [b for b in self.bookmarks if b.url in self.validation_results and 
                          self.validation_results[b.url].is_valid]
        
        if resume:
            state = self.checkpoint_manager.current_state
            processed_urls = set(state.ai_results.keys()) if state else set()
            bookmarks_to_process = [b for b in valid_bookmarks if b.url not in processed_urls]
        else:
            bookmarks_to_process = valid_bookmarks
        
        if self.progress_tracker:
            self.progress_tracker.start_stage(PTStage.GENERATING_DESCRIPTIONS, len(bookmarks_to_process))
        
        if bookmarks_to_process:
            logging.info(f"AI processing {len(bookmarks_to_process)} bookmarks")
            
            # Process in batches
            for i in range(0, len(bookmarks_to_process), self.config.batch_size):
                batch = bookmarks_to_process[i:i + self.config.batch_size]
                
                # Process batch
                batch_results = self.ai_processor.batch_process(
                    batch,
                    content_data_map=self.content_data,
                    progress_callback=self._ai_progress_callback
                )
                
                # Store results
                for result in batch_results:
                    self.ai_results[result.original_url] = result
                    self.checkpoint_manager.add_ai_result(result.original_url, result)
                
                # Update progress
                if self.progress_tracker:
                    self.progress_tracker.update(stage_items=i + len(batch))
                
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint()
        
        # Save final checkpoint
        self.checkpoint_manager.save_checkpoint(force=True)
        
        logging.info(f"AI processing complete: {len(self.ai_results)} bookmarks processed")
    
    def _stage_generate_tags(self) -> None:
        """Stage 5: Generate optimized tags"""
        logging.info("Stage 5: Generating optimized tags")
        
        self.checkpoint_manager.update_stage(ProcessingStage.TAG_OPTIMIZATION)
        
        if self.progress_tracker:
            self.progress_tracker.start_stage(PTStage.GENERATING_TAGS, 1)
        
        # Get valid bookmarks
        valid_bookmarks = [b for b in self.bookmarks if b.url in self.validation_results and 
                          self.validation_results[b.url].is_valid]
        
        if valid_bookmarks:
            # Generate corpus-wide tags
            tag_result = self.tag_generator.generate_corpus_tags(
                valid_bookmarks,
                content_data_map=self.content_data,
                ai_results_map=self.ai_results
            )
            
            self.tag_assignments = tag_result.tag_assignments
            
            # Store in checkpoint
            for url, tags in self.tag_assignments.items():
                self.checkpoint_manager.add_tag_assignment(url, tags)
            
            logging.info(f"Tag generation complete: {tag_result.total_unique_tags} unique tags, "
                        f"{tag_result.coverage_percentage:.1f}% coverage")
        
        if self.progress_tracker:
            self.progress_tracker.update(stage_items=1)
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(force=True)
    
    def _stage_generate_output(self) -> None:
        """Stage 6: Generate final output"""
        logging.info("Stage 6: Generating output")
        
        self.checkpoint_manager.update_stage(ProcessingStage.OUTPUT_GENERATION)
        
        if self.progress_tracker:
            self.progress_tracker.start_stage(PTStage.SAVING_RESULTS, 1)
        
        # Prepare output data
        output_bookmarks = []
        
        for bookmark in self.bookmarks:
            # Only include valid bookmarks
            validation_result = self.validation_results.get(bookmark.url)
            if not validation_result or not validation_result.is_valid:
                continue
            
            # Create enhanced bookmark
            enhanced_bookmark = bookmark.copy()
            
            # Update description from AI processing
            ai_result = self.ai_results.get(bookmark.url)
            if ai_result:
                enhanced_bookmark.note = ai_result.enhanced_description
            
            # Update tags
            tags = self.tag_assignments.get(bookmark.url, [])
            if tags:
                enhanced_bookmark.tags = self._format_tags_for_export(tags)
            
            output_bookmarks.append(enhanced_bookmark)
        
        # Generate output CSV
        try:
            self.csv_handler.save_import_csv(output_bookmarks, self.config.output_file)
            logging.info(f"Output saved: {self.config.output_file}")
        except Exception as e:
            logging.error(f"Failed to generate output: {e}")
            raise Exception("Failed to save output file")
        
        if self.progress_tracker:
            self.progress_tracker.update(stage_items=1)
        
        # Mark as completed
        self.checkpoint_manager.update_stage(ProcessingStage.COMPLETED)
        self.checkpoint_manager.save_checkpoint(force=True)
    
    def _restore_from_checkpoint(self, state: ProcessingState) -> None:
        """Restore pipeline state from checkpoint"""
        logging.info(f"Restoring state from {state.current_stage.value}")
        
        # Load bookmarks
        try:
            df = self.csv_handler.load_export_csv(state.input_file)
            self.bookmarks = []
            for _, row in df.iterrows():
                bookmark = Bookmark.from_raindrop_export(row.to_dict())
                self.bookmarks.append(bookmark)
            
            # Remove duplicates
            unique_bookmarks = {}
            for bookmark in self.bookmarks:
                if bookmark.url not in unique_bookmarks:
                    unique_bookmarks[bookmark.url] = bookmark
            self.bookmarks = list(unique_bookmarks.values())
            
        except Exception as e:
            logging.error(f"Failed to restore bookmarks: {e}")
            raise
        
        # Restore validation results
        for bookmark_data in state.validated_bookmarks:
            validation_data = bookmark_data['validation']
            result = ValidationResult(
                url=validation_data['url'],
                is_valid=validation_data['is_valid'],
                status_code=validation_data.get('status_code'),
                final_url=validation_data.get('final_url'),
                response_time=validation_data.get('response_time', 0.0),
                error_message=validation_data.get('error_message'),
                error_type=validation_data.get('error_type')
            )
            self.validation_results[result.url] = result
        
        # Restore failed validations
        for failed_data in state.failed_validations:
            result = ValidationResult(
                url=failed_data['url'],
                is_valid=failed_data['is_valid'],
                status_code=failed_data.get('status_code'),
                error_message=failed_data.get('error_message'),
                error_type=failed_data.get('error_type')
            )
            self.validation_results[result.url] = result
        
        # Restore content data
        for url, content_dict in state.content_data.items():
            content_data = ContentData(url=url)
            content_data.__dict__.update(content_dict)
            self.content_data[url] = content_data
        
        # Restore AI results
        for url, ai_dict in state.ai_results.items():
            ai_result = AIProcessingResult(
                original_url=ai_dict['original_url'],
                enhanced_description=ai_dict['enhanced_description'],
                processing_method=ai_dict['processing_method'],
                processing_time=ai_dict['processing_time']
            )
            self.ai_results[url] = ai_result
        
        # Restore tag assignments
        self.tag_assignments = state.tag_assignments.copy()
    
    def _format_tags_for_export(self, tags: List[str]) -> str:
        """Format tags for raindrop.io export"""
        if not tags:
            return ""
        elif len(tags) == 1:
            return tags[0]
        else:
            return f'"{", ".join(tags)}"'
    
    def _validation_progress_callback(self, message: str) -> None:
        """Callback for URL validation progress"""
        if self.progress_tracker:
            # Just update progress by 1 item, message is ignored for now
            self.progress_tracker.update_progress(items_delta=1)
            self._call_progress_callback()
    
    def _ai_progress_callback(self, message: str) -> None:
        """Callback for AI processing progress"""
        if self.progress_tracker:
            # Just update progress by 1 item, message is ignored for now
            self.progress_tracker.update_progress(items_delta=1)
            self._call_progress_callback()
    
    def _call_progress_callback(self) -> None:
        """Call the stored progress callback if available"""
        if self._progress_callback and self.progress_tracker:
            try:
                snapshot = self.progress_tracker.get_snapshot()
                progress_percentage = snapshot.overall_progress
                self._progress_callback(f"Progress: {progress_percentage:.1f}%")
            except Exception as e:
                logging.debug(f"Progress callback error: {e}")
    
    def _create_results(self) -> PipelineResults:
        """Create final pipeline results"""
        total_bookmarks = len(self.bookmarks)
        valid_bookmarks = sum(1 for r in self.validation_results.values() if r.is_valid)
        invalid_bookmarks = total_bookmarks - valid_bookmarks
        ai_processed = len(self.ai_results)
        tagged_bookmarks = len([url for url, tags in self.tag_assignments.items() if tags])
        unique_tags = len(set(tag for tags in self.tag_assignments.values() for tag in tags))
        
        # Calculate total processing time
        if self.checkpoint_manager.current_state:
            start_time = self.checkpoint_manager.current_state.start_time
            processing_time = (datetime.now() - start_time).total_seconds()
        else:
            processing_time = 0.0
        
        # Get stage names
        stages_completed = ["URL Validation", "Content Analysis", "AI Processing", "Tag Generation", "Output Generation"]
        
        # Collect statistics
        statistics = {
            'url_validation': self.url_validator.get_validation_statistics(),
            'ai_processing': self.ai_processor.get_processing_statistics() if self.ai_processor else {},
            'checkpoint_progress': self.checkpoint_manager.get_processing_progress()
        }
        
        return PipelineResults(
            total_bookmarks=total_bookmarks,
            valid_bookmarks=valid_bookmarks,
            invalid_bookmarks=invalid_bookmarks,
            ai_processed=ai_processed,
            tagged_bookmarks=tagged_bookmarks,
            unique_tags=unique_tags,
            processing_time=processing_time,
            stages_completed=stages_completed,
            error_summary=self.error_summary,
            statistics=statistics
        )
    
    def _cleanup_resources(self) -> None:
        """Clean up resources"""
        try:
            if self.url_validator:
                self.url_validator.close()
            if self.content_analyzer:
                self.content_analyzer.close()
            if self.ai_processor:
                self.ai_processor.close()
            if self.checkpoint_manager:
                self.checkpoint_manager.close()
            if self.progress_tracker:
                self.progress_tracker.complete()
            
            logging.info("Pipeline resources cleaned up")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")


def create_pipeline(input_file: str, 
                   output_file: str,
                   **kwargs) -> BookmarkProcessingPipeline:
    """
    Create a processing pipeline with standard configuration.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        **kwargs: Additional configuration options
        
    Returns:
        Configured BookmarkProcessingPipeline
    """
    config = PipelineConfig(
        input_file=input_file,
        output_file=output_file,
        **kwargs
    )
    
    return BookmarkProcessingPipeline(config)