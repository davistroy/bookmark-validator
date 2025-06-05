"""
AI Processing Module

Generates enhanced descriptions using existing bookmark content as input.
Uses local AI models for privacy and offline operation. Implements fallback
hierarchy as required by CLAUDE.md specifications.
"""

import os
import logging
import time
import re
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import warnings

# Suppress warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    # Try importing transformers but catch all errors
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
    logging.info("Transformers library available for AI processing")
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    logging.info(f"Transformers not available ({e}). Using intelligent fallback methods.")

from .content_analyzer import ContentData
from .data_models import Bookmark


@dataclass
class AIProcessingResult:
    """Result of AI processing"""
    original_url: str
    enhanced_description: str
    processing_method: str  # 'ai_with_context', 'existing_excerpt', 'meta_description', 'title_based'
    processing_time: float
    model_used: Optional[str] = None
    confidence_score: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'original_url': self.original_url,
            'enhanced_description': self.enhanced_description,
            'processing_method': self.processing_method,
            'processing_time': self.processing_time,
            'model_used': self.model_used,
            'confidence_score': self.confidence_score,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat()
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
    
    def get_summarizer(self, force_backup: bool = False):
        """Get or load the summarization model"""
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
                model_kwargs={"cache_dir": str(self.cache_dir)}
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
    
    def __init__(self, 
                 model_cache_dir: Optional[str] = None,
                 max_description_length: int = 150,
                 min_content_length: int = 10):
        """
        Initialize AI processor.
        
        Args:
            model_cache_dir: Directory to cache AI models
            max_description_length: Maximum length of generated descriptions
            min_content_length: Minimum length of content to consider valid
        """
        self.max_description_length = max_description_length
        self.min_content_length = min_content_length
        
        # Initialize model manager
        self.model_manager = ModelManager(model_cache_dir)
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'ai_generated': 0,
            'fallback_used': 0,
            'errors': 0,
            'processing_times': []
        }
        
        logging.info(f"AI processor initialized (max_length={max_description_length})")
    
    def process_bookmark(self, 
                        bookmark: Bookmark,
                        content_data: Optional[ContentData] = None) -> AIProcessingResult:
        """
        Process a bookmark to generate enhanced description.
        
        Args:
            bookmark: Bookmark object with existing data
            content_data: Optional content data from web analysis
            
        Returns:
            AIProcessingResult with enhanced description
        """
        start_time = time.time()
        
        try:
            # Prepare input context from existing data
            input_context = self._prepare_input_context(bookmark, content_data)
            
            # Try AI generation with context
            if input_context and len(input_context) >= self.min_content_length:
                ai_result = self._generate_with_ai(input_context, bookmark.url)
                if ai_result:
                    processing_time = time.time() - start_time
                    self.stats['ai_generated'] += 1
                    self.stats['processing_times'].append(processing_time)
                    
                    return AIProcessingResult(
                        original_url=bookmark.url,
                        enhanced_description=ai_result,
                        processing_method='ai_with_context',
                        processing_time=processing_time,
                        model_used=self.model_manager.model_name,
                        confidence_score=0.8
                    )
            
            # Apply fallback hierarchy
            fallback_result = self._apply_fallback_strategy(bookmark, content_data)
            processing_time = time.time() - start_time
            self.stats['fallback_used'] += 1
            
            return AIProcessingResult(
                original_url=bookmark.url,
                enhanced_description=fallback_result['description'],
                processing_method=fallback_result['method'],
                processing_time=processing_time,
                confidence_score=fallback_result['confidence']
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats['errors'] += 1
            
            logging.error(f"AI processing failed for {bookmark.url}: {e}")
            
            # Emergency fallback
            fallback_desc = self._emergency_fallback(bookmark)
            
            return AIProcessingResult(
                original_url=bookmark.url,
                enhanced_description=fallback_desc,
                processing_method='emergency_fallback',
                processing_time=processing_time,
                error_message=str(e),
                confidence_score=0.1
            )
        finally:
            self.stats['total_processed'] += 1
    
    def batch_process(self, 
                     bookmarks: List[Bookmark],
                     content_data_map: Optional[Dict[str, ContentData]] = None,
                     progress_callback: Optional[callable] = None) -> List[AIProcessingResult]:
        """
        Process multiple bookmarks in batch.
        
        Args:
            bookmarks: List of bookmarks to process
            content_data_map: Optional mapping of URL to ContentData
            progress_callback: Optional progress callback
            
        Returns:
            List of AIProcessingResult objects
        """
        if content_data_map is None:
            content_data_map = {}
        
        results = []
        
        # Pre-load AI model if available
        if TRANSFORMERS_AVAILABLE:
            self.model_manager.get_summarizer()
        
        for i, bookmark in enumerate(bookmarks):
            try:
                content_data = content_data_map.get(bookmark.url)
                result = self.process_bookmark(bookmark, content_data)
                results.append(result)
                
                if progress_callback:
                    progress_callback(f"AI processed {i+1}/{len(bookmarks)}: {bookmark.url}")
                
            except Exception as e:
                logging.error(f"Batch processing error for {bookmark.url}: {e}")
                # Continue with other bookmarks
                error_result = AIProcessingResult(
                    original_url=bookmark.url,
                    enhanced_description=self._emergency_fallback(bookmark),
                    processing_method='error_fallback',
                    processing_time=0.0,
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results
    
    def _prepare_input_context(self, 
                              bookmark: Bookmark,
                              content_data: Optional[ContentData]) -> str:
        """Prepare context from existing bookmark data"""
        context_parts = []
        
        # Priority 1: User's existing note (highest weight)
        if bookmark.note and len(bookmark.note.strip()) > self.min_content_length:
            context_parts.append(f"User Note: {bookmark.note.strip()}")
        
        # Priority 2: Existing excerpt from raindrop
        if bookmark.excerpt and len(bookmark.excerpt.strip()) > self.min_content_length:
            context_parts.append(f"Excerpt: {bookmark.excerpt.strip()}")
        
        # Priority 3: Page title
        if bookmark.title and len(bookmark.title.strip()) > 5:
            context_parts.append(f"Title: {bookmark.title.strip()}")
        
        # Priority 4: Content data if available
        if content_data:
            if content_data.meta_description and len(content_data.meta_description) > 20:
                context_parts.append(f"Description: {content_data.meta_description}")
            
            if content_data.main_content and len(content_data.main_content) > 50:
                # Use first 500 characters of content for context
                truncated_content = content_data.main_content[:500]
                context_parts.append(f"Content: {truncated_content}")
        
        return " | ".join(context_parts)
    
    def _generate_with_ai(self, input_context: str, url: str) -> Optional[str]:
        """Generate enhanced description using AI"""
        summarizer = self.model_manager.get_summarizer()
        
        if not summarizer:
            return None
        
        try:
            # Prepare input for summarization
            # Add instruction to maintain context and enhance
            prompt = f"Enhance this bookmark description while preserving the original context: {input_context}"
            
            # Limit input length for the model
            if len(prompt) > 1000:
                prompt = prompt[:1000] + "..."
            
            # Generate summary
            result = summarizer(
                prompt,
                max_length=self.max_description_length,
                min_length=30,
                do_sample=False,
                truncation=True
            )
            
            if result and len(result) > 0:
                enhanced_desc = result[0]['summary_text']
                
                # Clean and validate the result
                enhanced_desc = self._clean_ai_output(enhanced_desc)
                
                if len(enhanced_desc) >= 20:  # Minimum useful length
                    return enhanced_desc
            
        except Exception as e:
            logging.debug(f"AI generation failed for {url}: {e}")
        
        return None
    
    def _apply_fallback_strategy(self, 
                                bookmark: Bookmark,
                                content_data: Optional[ContentData]) -> Dict[str, Any]:
        """Apply fallback hierarchy for description generation"""
        
        # Fallback 1: Use existing excerpt if good quality
        if bookmark.excerpt and len(bookmark.excerpt.strip()) > 20:
            desc = bookmark.excerpt.strip()[:self.max_description_length]
            return {
                'description': desc,
                'method': 'existing_excerpt',
                'confidence': 0.7
            }
        
        # Fallback 2: Use meta description from content
        if (content_data and content_data.meta_description and 
            len(content_data.meta_description.strip()) > 20):
            desc = content_data.meta_description.strip()[:self.max_description_length]
            return {
                'description': desc,
                'method': 'meta_description',
                'confidence': 0.6
            }
        
        # Fallback 3: Enhanced title + domain
        title = bookmark.title or "Bookmark"
        domain = self._extract_domain(bookmark.url)
        
        # Create a more descriptive fallback
        if content_data and content_data.content_categories:
            category = content_data.content_categories[0]
            desc = f"{title} - {category.title()} content from {domain}"
        else:
            desc = f"{title} - Content from {domain}"
        
        desc = desc[:self.max_description_length]
        
        return {
            'description': desc,
            'method': 'title_based',
            'confidence': 0.4
        }
    
    def _emergency_fallback(self, bookmark: Bookmark) -> str:
        """Emergency fallback for complete failures"""
        title = bookmark.title or "Bookmark"
        domain = self._extract_domain(bookmark.url)
        return f"{title} from {domain}"[:self.max_description_length]
    
    def _clean_ai_output(self, text: str) -> str:
        """Clean and validate AI-generated text"""
        if not text:
            return ""
        
        # Remove common AI artifacts
        text = re.sub(r'^(Summary:|Description:|Enhanced:|The text describes?:?)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Remove quotes if they wrap the entire description
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        
        # Ensure it ends with proper punctuation
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text[:self.max_description_length]
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain
        except:
            return "unknown"
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.stats.copy()
        
        if stats['processing_times']:
            stats['average_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times'])
            stats['total_processing_time'] = sum(stats['processing_times'])
        else:
            stats['average_processing_time'] = 0.0
            stats['total_processing_time'] = 0.0
        
        if stats['total_processed'] > 0:
            stats['ai_success_rate'] = stats['ai_generated'] / stats['total_processed']
            stats['error_rate'] = stats['errors'] / stats['total_processed']
        else:
            stats['ai_success_rate'] = 0.0
            stats['error_rate'] = 0.0
        
        # Don't include raw processing times in output
        del stats['processing_times']
        
        return stats
    
    def clear_cache(self):
        """Clear model cache"""
        self.model_manager.clear_cache()
    
    def close(self):
        """Clean up resources"""
        # Clear model from memory
        self.model_manager.summarizer = None
        logging.info("AI processor closed")