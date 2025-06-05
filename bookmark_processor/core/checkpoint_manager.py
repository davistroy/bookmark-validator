"""
Checkpoint Manager Module

Provides progress persistence and resume functionality for long-running bookmark processing.
Handles secure checkpoint files with automatic cleanup and recovery mechanisms.
"""

import os
import json
import pickle
import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import threading
import tempfile

from .data_models import Bookmark
from .url_validator import ValidationResult
from .content_analyzer import ContentData
from .ai_processor import AIProcessingResult


class ProcessingStage(Enum):
    """Processing pipeline stages"""
    INITIALIZATION = "initialization"
    LOADING = "loading"
    DEDUPLICATION = "deduplication" 
    URL_VALIDATION = "url_validation"
    CONTENT_ANALYSIS = "content_analysis"
    AI_PROCESSING = "ai_processing"
    TAG_OPTIMIZATION = "tag_optimization"
    OUTPUT_GENERATION = "output_generation"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProcessingState:
    """Complete processing state for checkpointing"""
    # Basic info
    input_file: str
    output_file: str
    total_bookmarks: int
    config_hash: str
    
    # Current state
    current_stage: ProcessingStage
    stage_progress: int
    processed_urls: Set[str] = field(default_factory=set)
    
    # Data storage
    validated_bookmarks: List[Dict[str, Any]] = field(default_factory=list)
    failed_validations: List[Dict[str, Any]] = field(default_factory=list)
    content_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ai_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tag_assignments: Dict[str, List[str]] = field(default_factory=dict)
    
    # Timing info
    start_time: datetime = field(default_factory=datetime.now)
    last_checkpoint_time: datetime = field(default_factory=datetime.now)
    stage_start_time: datetime = field(default_factory=datetime.now)
    
    # Statistics
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        
        # Convert datetime objects to ISO strings
        data['start_time'] = self.start_time.isoformat()
        data['last_checkpoint_time'] = self.last_checkpoint_time.isoformat()
        data['stage_start_time'] = self.stage_start_time.isoformat()
        
        # Convert enum to string
        data['current_stage'] = self.current_stage.value
        
        # Convert set to list for JSON serialization
        data['processed_urls'] = list(self.processed_urls)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingState':
        """Create from dictionary loaded from JSON"""
        # Convert datetime strings back to datetime objects
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        data['last_checkpoint_time'] = datetime.fromisoformat(data['last_checkpoint_time'])
        data['stage_start_time'] = datetime.fromisoformat(data['stage_start_time'])
        
        # Convert string back to enum
        data['current_stage'] = ProcessingStage(data['current_stage'])
        
        # Convert list back to set
        data['processed_urls'] = set(data['processed_urls'])
        
        return cls(**data)


class CheckpointManager:
    """Manages processing state persistence and recovery"""
    
    def __init__(self, 
                 checkpoint_dir: str = ".bookmark_checkpoints",
                 save_interval: int = 50,
                 auto_cleanup: bool = True,
                 compression: bool = True):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
            save_interval: Number of processed items between saves
            auto_cleanup: Whether to automatically clean up old checkpoints
            compression: Whether to compress checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_interval = save_interval
        self.auto_cleanup = auto_cleanup
        self.compression = compression
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # State management
        self.current_state: Optional[ProcessingState] = None
        self.checkpoint_file: Optional[Path] = None
        self.last_save_count = 0
        self.lock = threading.RLock()
        
        logging.info(f"Checkpoint manager initialized (dir={checkpoint_dir}, "
                    f"interval={save_interval})")
    
    def initialize_processing(self, 
                             input_file: str, 
                             output_file: str,
                             total_bookmarks: int,
                             config: Dict[str, Any]) -> ProcessingState:
        """
        Initialize new processing session.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            total_bookmarks: Total number of bookmarks to process
            config: Configuration dictionary
            
        Returns:
            New ProcessingState object
        """
        with self.lock:
            # Create config hash for validation
            config_hash = self._create_config_hash(config)
            
            # Create new processing state
            self.current_state = ProcessingState(
                input_file=input_file,
                output_file=output_file,
                total_bookmarks=total_bookmarks,
                config_hash=config_hash,
                current_stage=ProcessingStage.INITIALIZATION,
                stage_progress=0
            )
            
            # Generate checkpoint filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_{timestamp}.json"
            self.checkpoint_file = self.checkpoint_dir / filename
            
            logging.info(f"Initialized new processing session: {filename}")
            return self.current_state
    
    def has_checkpoint(self, input_file: Optional[str] = None) -> bool:
        """
        Check if a valid checkpoint exists.
        
        Args:
            input_file: Optional input file to match against checkpoint
            
        Returns:
            True if valid checkpoint exists
        """
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
            
            if not checkpoint_files:
                return False
            
            # Find the most recent checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            
            # If input file specified, verify it matches
            if input_file:
                state = self._load_checkpoint_file(latest_checkpoint)
                if state and state.input_file != input_file:
                    return False
            
            return True
            
        except Exception as e:
            logging.debug(f"Error checking for checkpoint: {e}")
            return False
    
    def load_checkpoint(self, input_file: Optional[str] = None) -> Optional[ProcessingState]:
        """
        Load the most recent checkpoint.
        
        Args:
            input_file: Optional input file to match against checkpoint
            
        Returns:
            ProcessingState if found, None otherwise
        """
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
            
            if not checkpoint_files:
                logging.info("No checkpoint files found")
                return None
            
            # Find the most recent checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            
            logging.info(f"Loading checkpoint: {latest_checkpoint.name}")
            
            state = self._load_checkpoint_file(latest_checkpoint)
            
            if not state:
                return None
            
            # Verify input file matches if specified
            if input_file and state.input_file != input_file:
                logging.warning(f"Checkpoint input file mismatch: "
                              f"expected {input_file}, got {state.input_file}")
                return None
            
            # Set current state and checkpoint file
            with self.lock:
                self.current_state = state
                self.checkpoint_file = latest_checkpoint
                self.last_save_count = len(state.processed_urls)
            
            logging.info(f"Loaded checkpoint from stage {state.current_stage.value} "
                        f"with {len(state.processed_urls)} processed items")
            
            return state
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return None
    
    def save_checkpoint(self, force: bool = False) -> bool:
        """
        Save current processing state to checkpoint.
        
        Args:
            force: Force save even if interval hasn't been reached
            
        Returns:
            True if checkpoint was saved
        """
        if not self.current_state:
            return False
        
        with self.lock:
            # Check if we should save based on interval
            current_count = len(self.current_state.processed_urls)
            items_since_save = current_count - self.last_save_count
            
            if not force and items_since_save < self.save_interval:
                return False
            
            try:
                # Update checkpoint time
                self.current_state.last_checkpoint_time = datetime.now()
                
                # Save to temporary file first for atomic operation
                temp_file = self.checkpoint_file.with_suffix('.tmp')
                
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.current_state.to_dict(), f, indent=2)
                
                # Atomic rename
                temp_file.replace(self.checkpoint_file)
                
                self.last_save_count = current_count
                
                logging.debug(f"Checkpoint saved: {current_count} items processed")
                return True
                
            except Exception as e:
                logging.error(f"Failed to save checkpoint: {e}")
                return False
    
    def update_stage(self, stage: ProcessingStage, progress: int = 0) -> None:
        """
        Update current processing stage.
        
        Args:
            stage: New processing stage
            progress: Progress within the stage
        """
        if not self.current_state:
            return
        
        with self.lock:
            old_stage = self.current_state.current_stage
            self.current_state.current_stage = stage
            self.current_stage_progress = progress
            self.current_state.stage_start_time = datetime.now()
            
            logging.info(f"Stage transition: {old_stage.value} → {stage.value}")
    
    def add_validated_bookmark(self, validation_result: ValidationResult,
                              bookmark: Bookmark) -> None:
        """Add validated bookmark to checkpoint data"""
        if not self.current_state:
            return
        
        with self.lock:
            if validation_result.is_valid:
                bookmark_data = {
                    'bookmark': bookmark.to_dict(),
                    'validation': validation_result.to_dict()
                }
                self.current_state.validated_bookmarks.append(bookmark_data)
            else:
                self.current_state.failed_validations.append(validation_result.to_dict())
            
            self.current_state.processed_urls.add(bookmark.url)
    
    def add_content_data(self, url: str, content_data: ContentData) -> None:
        """Add content analysis data to checkpoint"""
        if not self.current_state:
            return
        
        with self.lock:
            self.current_state.content_data[url] = content_data.to_dict()
    
    def add_ai_result(self, url: str, ai_result: AIProcessingResult) -> None:
        """Add AI processing result to checkpoint"""
        if not self.current_state:
            return
        
        with self.lock:
            self.current_state.ai_results[url] = ai_result.to_dict()
    
    def add_tag_assignment(self, url: str, tags: List[str]) -> None:
        """Add tag assignment to checkpoint"""
        if not self.current_state:
            return
        
        with self.lock:
            self.current_state.tag_assignments[url] = tags
    
    def update_statistics(self, stats: Dict[str, Any]) -> None:
        """Update processing statistics"""
        if not self.current_state:
            return
        
        with self.lock:
            self.current_state.processing_stats.update(stats)
    
    def get_processing_progress(self) -> Dict[str, Any]:
        """Get current processing progress information"""
        if not self.current_state:
            return {}
        
        with self.lock:
            elapsed_time = datetime.now() - self.current_state.start_time
            stage_time = datetime.now() - self.current_state.stage_start_time
            
            progress_info = {
                'current_stage': self.current_state.current_stage.value,
                'total_bookmarks': self.current_state.total_bookmarks,
                'processed_count': len(self.current_state.processed_urls),
                'valid_bookmarks': len(self.current_state.validated_bookmarks),
                'failed_validations': len(self.current_state.failed_validations),
                'content_analyzed': len(self.current_state.content_data),
                'ai_processed': len(self.current_state.ai_results),
                'tags_assigned': len(self.current_state.tag_assignments),
                'elapsed_time_seconds': elapsed_time.total_seconds(),
                'stage_time_seconds': stage_time.total_seconds(),
                'last_checkpoint': self.current_state.last_checkpoint_time.isoformat()
            }
            
            # Calculate progress percentage
            if self.current_state.total_bookmarks > 0:
                progress_info['progress_percentage'] = (
                    len(self.current_state.processed_urls) / 
                    self.current_state.total_bookmarks * 100
                )
            else:
                progress_info['progress_percentage'] = 0.0
            
            return progress_info
    
    def clear_checkpoint(self) -> bool:
        """Clear current checkpoint and cleanup files"""
        try:
            with self.lock:
                if self.checkpoint_file and self.checkpoint_file.exists():
                    self.checkpoint_file.unlink()
                    logging.info(f"Cleared checkpoint: {self.checkpoint_file.name}")
                
                self.current_state = None
                self.checkpoint_file = None
                self.last_save_count = 0
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to clear checkpoint: {e}")
            return False
    
    def cleanup_old_checkpoints(self, keep_count: int = 3) -> None:
        """
        Clean up old checkpoint files.
        
        Args:
            keep_count: Number of recent checkpoints to keep
        """
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
            
            if len(checkpoint_files) <= keep_count:
                return
            
            # Sort by modification time (newest first)
            checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            # Remove old files
            for old_file in checkpoint_files[keep_count:]:
                old_file.unlink()
                logging.debug(f"Removed old checkpoint: {old_file.name}")
            
            logging.info(f"Cleaned up {len(checkpoint_files) - keep_count} old checkpoints")
            
        except Exception as e:
            logging.error(f"Failed to cleanup old checkpoints: {e}")
    
    def _load_checkpoint_file(self, checkpoint_file: Path) -> Optional[ProcessingState]:
        """Load checkpoint from specific file"""
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            state = ProcessingState.from_dict(data)
            
            # Validate checkpoint age (don't load if too old)
            age = datetime.now() - state.last_checkpoint_time
            if age > timedelta(days=7):  # 7 days
                logging.warning(f"Checkpoint is too old ({age.days} days), ignoring")
                return None
            
            return state
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint file {checkpoint_file}: {e}")
            return None
    
    def _create_config_hash(self, config: Dict[str, Any]) -> str:
        """Create hash of configuration for validation"""
        try:
            # Sort config for consistent hashing
            config_str = json.dumps(config, sort_keys=True)
            return hashlib.md5(config_str.encode()).hexdigest()[:16]
        except Exception:
            return "unknown"
    
    def close(self) -> None:
        """Clean up resources and perform final checkpoint save"""
        try:
            # Final checkpoint save
            if self.current_state:
                self.save_checkpoint(force=True)
            
            # Cleanup old checkpoints if enabled
            if self.auto_cleanup:
                self.cleanup_old_checkpoints()
            
            logging.info("Checkpoint manager closed")
            
        except Exception as e:
            logging.error(f"Error during checkpoint manager cleanup: {e}")