"""
AI Processing Module

Generates enhanced descriptions using existing bookmark content as input.
Uses local AI models for privacy and offline operation. Implements fallback
hierarchy as required by CLAUDE.md specifications.
"""

import logging
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    # Try importing transformers but catch all errors
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    TRANSFORMERS_AVAILABLE = True
    logging.info("Transformers library available for AI processing")
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    logging.info(
        f"Transformers not available ({e}). Using intelligent fallback methods."
    )

from .content_analyzer import ContentData
from .data_models import Bookmark

# Import cloud clients for module-level access (needed for test patching)
try:
    from .claude_api_client import ClaudeAPIClient
except ImportError:
    ClaudeAPIClient = None

try:
    from .openai_api_client import OpenAIAPIClient
except ImportError:
    OpenAIAPIClient = None


@dataclass
class AIProcessingResult:
    """Result of AI processing"""

    original_url: str
    enhanced_description: str
    processing_method: (
        str  # 'ai_with_context', 'existing_excerpt', 'meta_description', 'title_based'
    )
    processing_time: float
    model_used: Optional[str] = None
    confidence_score: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "original_url": self.original_url,
            "enhanced_description": self.enhanced_description,
            "processing_method": self.processing_method,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "confidence_score": self.confidence_score,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }


class ModelManager:
    """Manage AI models with caching for Linux environment"""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model manager.

        Args:
            cache_dir: Directory to cache models (defaults to ~/.cache/bookmark_processor)
        """
        if cache_dir is None:
            # Use user's cache directory
            cache_dir = os.path.expanduser("~/.cache/bookmark_processor/models")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.summarizer = None
        self.model_name = "facebook/bart-large-cnn"
        self.backup_model = "sshleifer/distilbart-cnn-12-6"  # Smaller backup model

        logging.info(f"Model manager initialized with cache: {self.cache_dir}")

    def _get_mock_summarizer(self):
        """Return mock summarizer for test mode."""
        from unittest.mock import Mock

        mock_summarizer = Mock()
        mock_summarizer.return_value = [
            {"summary_text": "Mock AI-generated summary for testing."}
        ]
        return mock_summarizer

    def get_summarizer(self, force_backup: bool = False):
        """Get or load the summarization model"""
        # Test mode guard - don't load real models during testing
        if os.getenv("BOOKMARK_PROCESSOR_TEST_MODE") == "true":
            return self._get_mock_summarizer()

        if self.summarizer is not None:
            return self.summarizer

        if not TRANSFORMERS_AVAILABLE:
            logging.warning("Transformers not available, AI processing disabled")
            return None

        try:
            model_to_use = self.backup_model if force_backup else self.model_name

            logging.info(f"Loading AI model: {model_to_use}")
            start_time = time.time()

            # Try to load model with caching
            self.summarizer = pipeline(
                "summarization",
                model=model_to_use,
                tokenizer=model_to_use,
                device=-1,  # Use CPU
                model_kwargs={"cache_dir": str(self.cache_dir)},
            )

            load_time = time.time() - start_time
            logging.info(f"Model loaded successfully in {load_time:.2f}s")

            return self.summarizer

        except Exception as e:
            logging.error(f"Failed to load model {model_to_use}: {e}")

            # Try backup model if main model failed
            if not force_backup:
                logging.info("Trying backup model...")
                return self.get_summarizer(force_backup=True)

            logging.error("All AI models failed to load. AI processing disabled.")
            return None

    def clear_cache(self):
        """Clear model cache"""
        try:
            import shutil

            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logging.info("Model cache cleared")
        except Exception as e:
            logging.error(f"Failed to clear cache: {e}")


class EnhancedAIProcessor:
    """AI processor for enhanced descriptions using existing content as input"""

    def __init__(
        self,
        engine: str = "local",
        api_key: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
        max_description_length: int = 150,
        min_content_length: int = 10,
    ):
        """
        Initialize AI processor.

        Args:
            engine: AI engine to use ('local', 'claude', 'openai')
            api_key: API key for cloud services
            model_cache_dir: Directory to cache AI models
            max_description_length: Maximum length of generated descriptions
            min_content_length: Minimum length of content to consider valid
        """
        self.engine = engine
        self.api_key = api_key
        self.max_description_length = max_description_length
        self.min_content_length = min_content_length
        self.cloud_client = None
        self.model_name = "facebook/bart-large-cnn"
        self.processed_count = 0

        # Initialize based on engine type
        if engine == "claude" and api_key:
            if ClaudeAPIClient is not None:
                self.cloud_client = ClaudeAPIClient(api_key=api_key)
                if not self.cloud_client.is_available:
                    logging.warning("Claude API not available, falling back to local")
                    self.engine = "local"
            else:
                logging.warning("Claude client not available, falling back to local")
                self.engine = "local"
        elif engine == "openai" and api_key:
            if OpenAIAPIClient is not None:
                self.cloud_client = OpenAIAPIClient(api_key=api_key)
                if not self.cloud_client.is_available:
                    logging.warning("OpenAI API not available, falling back to local")
                    self.engine = "local"
            else:
                logging.warning("OpenAI client not available, falling back to local")
                self.engine = "local"
        elif engine not in ["local", "claude", "openai"]:
            logging.warning(f"Unknown engine '{engine}', falling back to local")
            self.engine = "local"
        elif engine in ["claude", "openai"] and not api_key:
            logging.warning(f"No API key provided for {engine}, falling back to local")
            self.engine = "local"

        # Initialize model manager for local processing
        self.model_manager = ModelManager(model_cache_dir)

        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "ai_generated": 0,
            "fallback_used": 0,
            "errors": 0,
            "processing_times": [],
        }

        logging.info(
            f"AI processor initialized (engine={self.engine}, max_length={max_description_length})"
        )

    @property
    def is_available(self) -> bool:
        """Check if AI processor is available and ready to use."""
        # In test mode, always consider AI available
        if os.getenv("BOOKMARK_PROCESSOR_TEST_MODE") == "true":
            return True

        if self.engine == "local":
            return TRANSFORMERS_AVAILABLE
        elif self.cloud_client:
            return self.cloud_client.is_available
        return False

    def process_bookmark(self, bookmark: Bookmark) -> Bookmark:
        """
        Process a bookmark to generate enhanced description.

        Args:
            bookmark: Bookmark object to process in place

        Returns:
            The modified bookmark object
        """
        start_time = time.time()
        self.processed_count += 1

        try:
            # Try to generate enhanced description
            enhanced_description = None

            if self.engine == "local":
                enhanced_description = self._process_with_local_ai(bookmark)
            elif self.cloud_client and self.cloud_client.is_available:
                enhanced_description = self._process_with_cloud_ai(bookmark)

            if enhanced_description:
                bookmark.enhanced_description = enhanced_description
                bookmark.processing_status.ai_processed = True
                bookmark.processing_status.ai_processing_error = None
                self.stats["ai_generated"] += 1
            else:
                # Fallback to existing content
                fallback_desc = self._generate_fallback_description(bookmark)
                bookmark.enhanced_description = fallback_desc
                bookmark.processing_status.ai_processed = False
                self.stats["fallback_used"] += 1

        except Exception as e:
            error_msg = str(e)
            logging.error(f"AI processing failed for {bookmark.url}: {error_msg}")

            # Set error state and fallback description
            bookmark.processing_status.ai_processed = False
            bookmark.processing_status.ai_processing_error = error_msg
            bookmark.enhanced_description = self._generate_fallback_description(
                bookmark
            )
            self.stats["errors"] += 1

        finally:
            processing_time = time.time() - start_time
            self.stats["processing_times"].append(processing_time)
            self.stats["total_processed"] += 1
            bookmark.processing_status.processing_attempts += 1
            bookmark.processing_status.last_attempt = datetime.now()

        return bookmark

    def _process_with_local_ai(self, bookmark: Bookmark) -> Optional[str]:
        """Process bookmark with local AI model."""
        summarizer = self.model_manager.get_summarizer()
        if not summarizer:
            return None

        try:
            input_text = self._prepare_input_text(bookmark)
            if not input_text or len(input_text) < self.min_content_length:
                return None

            # Generate with AI
            result = summarizer(
                input_text,
                max_length=self.max_description_length,
                min_length=30,
                do_sample=False,
                truncation=True,
            )

            if result and len(result) > 0:
                enhanced_desc = result[0]["summary_text"]
                return self._clean_ai_output(enhanced_desc)

        except Exception as e:
            logging.debug(f"Local AI processing failed for {bookmark.url}: {e}")

        return None

    def _process_with_cloud_ai(self, bookmark: Bookmark) -> Optional[str]:
        """Process bookmark with cloud AI service."""
        try:
            return self.cloud_client.generate_description(bookmark)
        except Exception as e:
            logging.debug(f"Cloud AI processing failed for {bookmark.url}: {e}")
            return None

    def _prepare_input_text(self, bookmark: Bookmark) -> str:
        """Prepare input text for AI processing."""
        context_parts = []

        # Add title
        if bookmark.title and bookmark.title.strip():
            context_parts.append(f"Title: {bookmark.title.strip()}")

        # Add user note (highest priority)
        if bookmark.note and bookmark.note.strip():
            context_parts.append(f"Note: {bookmark.note.strip()}")

        # Add excerpt
        if bookmark.excerpt and bookmark.excerpt.strip():
            context_parts.append(f"Excerpt: {bookmark.excerpt.strip()}")

        # Add domain context
        if bookmark.url:
            domain = self._extract_domain(bookmark.url)
            context_parts.append(f"Domain: {domain}")

        return " | ".join(context_parts)

    def _generate_fallback_description(self, bookmark: Bookmark) -> str:
        """Generate fallback description using existing content."""
        # Priority: note -> excerpt -> title-based
        if bookmark.note and bookmark.note.strip():
            return bookmark.note.strip()[: self.max_description_length]

        if bookmark.excerpt and bookmark.excerpt.strip():
            return bookmark.excerpt.strip()[: self.max_description_length]

        # Generate title-based description
        title = bookmark.title or "Bookmark"
        domain = self._extract_domain(bookmark.url)
        return f"{title} from {domain}"[: self.max_description_length]

    def process_batch(
        self,
        bookmarks: List[Bookmark],
        content_data_map: Dict = None,
        progress_callback=None,
    ) -> List[Bookmark]:
        """
        Process multiple bookmarks in batch.

        Args:
            bookmarks: List of bookmarks to process
            content_data_map: Optional mapping of URLs to content data
            progress_callback: Optional callback for progress updates

        Returns:
            List of processed Bookmark objects
        """
        results = []

        # Pre-load AI model if available for local processing
        if self.engine == "local" and TRANSFORMERS_AVAILABLE:
            self.model_manager.get_summarizer()

        for bookmark in bookmarks:
            try:
                result = self.process_bookmark(bookmark)
                results.append(result)
            except Exception as e:
                logging.error(f"Batch processing error for {bookmark.url}: {e}")
                # Set error state and continue
                bookmark.processing_status.ai_processed = False
                bookmark.processing_status.ai_processing_error = str(e)
                bookmark.enhanced_description = self._generate_fallback_description(
                    bookmark
                )
                results.append(bookmark)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = {
            "engine": self.engine,
            "model_name": self.model_name,
            "is_available": self.is_available,
            "processed_count": self.processed_count,
            "total_processed": self.stats["total_processed"],
            "ai_generated": self.stats["ai_generated"],
            "fallback_used": self.stats["fallback_used"],
            "errors": self.stats["errors"],
        }

        if self.stats["processing_times"]:
            stats["average_processing_time"] = sum(
                self.stats["processing_times"]
            ) / len(self.stats["processing_times"])
            stats["total_processing_time"] = sum(self.stats["processing_times"])
        else:
            stats["average_processing_time"] = 0.0
            stats["total_processing_time"] = 0.0

        return stats

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics for pipeline compatibility."""
        return self.get_statistics()

    def _clean_ai_output(self, text: str) -> str:
        """Clean and validate AI-generated text"""
        if not text:
            return ""

        # Remove common AI artifacts
        text = re.sub(
            r"^(Summary:|Description:|Enhanced:|The text describes?:?)\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        text = text.strip()

        # Remove quotes if they wrap the entire description
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]

        # Ensure it ends with proper punctuation
        if text and not text[-1] in ".!?":
            text += "."

        return text[: self.max_description_length]

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]

            return domain
        except:
            return "unknown"

    def clear_cache(self):
        """Clear model cache"""
        self.model_manager.clear_cache()

    def close(self):
        """Clean up resources"""
        # Clear model from memory
        self.model_manager.summarizer = None
        logging.info("AI processor closed")
