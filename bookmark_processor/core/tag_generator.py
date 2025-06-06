"""
Tag Generation Module

Generates optimized tags for entire bookmark corpus using content analysis,
URL patterns, and intelligent tag optimization to create a coherent tagging system.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from urllib.parse import urlparse
import math

from .data_models import Bookmark
from .content_analyzer import ContentData
from .ai_processor import AIProcessingResult


@dataclass
class TagCandidate:
    """A candidate tag with scoring information"""
    tag: str
    frequency: int = 0
    content_score: float = 0.0
    url_score: float = 0.0
    quality_score: float = 0.0
    bookmarks: Set[str] = field(default_factory=set)
    
    @property
    def total_score(self) -> float:
        """Calculate total weighted score"""
        return (
            self.frequency * 0.3 +
            self.content_score * 0.4 +
            self.url_score * 0.2 +
            self.quality_score * 0.1
        )


@dataclass
class TagOptimizationResult:
    """Result of tag optimization process"""
    optimized_tags: List[str]
    tag_assignments: Dict[str, List[str]]  # URL -> tags
    total_unique_tags: int
    coverage_percentage: float
    optimization_stats: Dict[str, Any]


class CorpusAwareTagGenerator:
    """Generate optimized tags for entire bookmark corpus"""
    
    # Content type indicators for smart tagging
    CONTENT_TYPE_INDICATORS = {
        'tutorial': ['tutorial', 'guide', 'how-to', 'learn', 'course', 'lesson'],
        'documentation': ['docs', 'documentation', 'api', 'reference', 'manual'],
        'article': ['article', 'blog', 'post', 'news', 'story'],
        'tool': ['tool', 'utility', 'app', 'software', 'service'],
        'video': ['video', 'youtube', 'watch', 'stream', 'movie'],
        'code': ['github', 'gitlab', 'code', 'repository', 'source'],
        'research': ['paper', 'research', 'study', 'academic', 'journal'],
        'resource': ['resource', 'list', 'collection', 'awesome', 'curated']
    }
    
    # Technology keywords for tech-focused tagging
    TECH_KEYWORDS = {
        'web': ['html', 'css', 'javascript', 'react', 'vue', 'angular', 'web', 'frontend'],
        'backend': ['node', 'python', 'java', 'go', 'rust', 'backend', 'server', 'api'],
        'database': ['sql', 'mysql', 'postgres', 'mongodb', 'database', 'db'],
        'mobile': ['android', 'ios', 'mobile', 'app', 'react-native', 'flutter'],
        'devops': ['docker', 'kubernetes', 'aws', 'azure', 'cloud', 'devops'],
        'ai': ['ai', 'machine-learning', 'ml', 'deep-learning', 'neural', 'tensorflow'],
        'design': ['design', 'ui', 'ux', 'figma', 'sketch', 'adobe']
    }
    
    # Common stop words to filter out
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'then', 'once', 'here', 'there',
        'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',
        'should', 'now', 'use', 'get', 'make', 'work', 'way', 'new', 'good'
    }
    
    def __init__(self, 
                 target_tag_count: int = 150,
                 max_tags_per_bookmark: int = 5,
                 min_tag_frequency: int = 2,
                 quality_threshold: float = 0.3):
        """
        Initialize tag generator.
        
        Args:
            target_tag_count: Target number of unique tags
            max_tags_per_bookmark: Maximum tags per bookmark
            min_tag_frequency: Minimum frequency for tag inclusion
            quality_threshold: Minimum quality score for tag inclusion
        """
        self.target_tag_count = target_tag_count
        self.max_tags_per_bookmark = max_tags_per_bookmark
        self.min_tag_frequency = min_tag_frequency
        self.quality_threshold = quality_threshold
        
        # Analysis results
        self.tag_candidates: Dict[str, TagCandidate] = {}
        self.bookmark_profiles: Dict[str, Dict[str, Any]] = {}
        
        logging.info(f"Tag generator initialized (target={target_tag_count} tags, "
                    f"max_per_bookmark={max_tags_per_bookmark})")
    
    def generate_corpus_tags(self, 
                           bookmarks: List[Bookmark],
                           content_data_map: Optional[Dict[str, ContentData]] = None,
                           ai_results_map: Optional[Dict[str, AIProcessingResult]] = None) -> TagOptimizationResult:
        """
        Generate optimized tags for entire corpus.
        
        Args:
            bookmarks: List of all bookmarks
            content_data_map: Optional content analysis data
            ai_results_map: Optional AI processing results
            
        Returns:
            TagOptimizationResult with optimized tag assignments
        """
        logging.info(f"Starting corpus tag generation for {len(bookmarks)} bookmarks")
        
        if content_data_map is None:
            content_data_map = {}
        if ai_results_map is None:
            ai_results_map = {}
        
        # Phase 1: Extract all candidate tags
        self._extract_candidate_tags(bookmarks, content_data_map, ai_results_map)
        
        # Phase 2: Score and rank candidates
        self._score_tag_candidates(bookmarks, content_data_map)
        
        # Phase 3: Select optimal tag set
        optimal_tags = self._select_optimal_tags()
        
        # Phase 4: Assign tags to bookmarks
        tag_assignments = self._assign_tags_to_bookmarks(bookmarks, optimal_tags, content_data_map)
        
        # Phase 5: Calculate optimization stats
        stats = self._calculate_optimization_stats(bookmarks, tag_assignments)
        
        # Calculate coverage
        tagged_bookmarks = len([url for url, tags in tag_assignments.items() if tags])
        coverage_percentage = (tagged_bookmarks / len(bookmarks)) * 100 if bookmarks else 0
        
        result = TagOptimizationResult(
            optimized_tags=optimal_tags,
            tag_assignments=tag_assignments,
            total_unique_tags=len(optimal_tags),
            coverage_percentage=coverage_percentage,
            optimization_stats=stats
        )
        
        logging.info(f"Tag optimization complete: {len(optimal_tags)} unique tags, "
                    f"{coverage_percentage:.1f}% coverage")
        
        return result
    
    def _extract_candidate_tags(self, 
                               bookmarks: List[Bookmark],
                               content_data_map: Dict[str, ContentData],
                               ai_results_map: Dict[str, AIProcessingResult]) -> None:
        """Extract all potential tags from various sources"""
        logging.info("Extracting candidate tags from all sources")
        
        for bookmark in bookmarks:
            bookmark_tags = set()
            
            # Extract from existing tags
            if bookmark.tags:
                existing_tags = self._parse_existing_tags(bookmark.tags)
                bookmark_tags.update(existing_tags)
            
            # Extract from URL patterns
            url_tags = self._extract_url_tags(bookmark.url)
            bookmark_tags.update(url_tags)
            
            # Extract from title and descriptions
            title_tags = self._extract_text_tags(bookmark.title or "")
            bookmark_tags.update(title_tags)
            
            if bookmark.note:
                note_tags = self._extract_text_tags(bookmark.note)
                bookmark_tags.update(note_tags)
            
            # Extract from content analysis
            content_data = content_data_map.get(bookmark.url)
            if content_data:
                content_tags = self._extract_content_tags(content_data)
                bookmark_tags.update(content_tags)
            
            # Extract from AI results
            ai_result = ai_results_map.get(bookmark.url)
            if ai_result:
                ai_tags = self._extract_text_tags(ai_result.enhanced_description)
                bookmark_tags.update(ai_tags)
            
            # Clean and validate tags
            cleaned_tags = self._clean_tags(bookmark_tags)
            
            # Store bookmark profile
            self.bookmark_profiles[bookmark.url] = {
                'title': bookmark.title or "",
                'tags': cleaned_tags,
                'domain': self._extract_domain(bookmark.url),
                'content_categories': content_data.content_categories if content_data else []
            }
            
            # Update tag candidates
            for tag in cleaned_tags:
                if tag not in self.tag_candidates:
                    self.tag_candidates[tag] = TagCandidate(tag=tag)
                
                candidate = self.tag_candidates[tag]
                candidate.frequency += 1
                candidate.bookmarks.add(bookmark.url)
    
    def _score_tag_candidates(self, 
                            bookmarks: List[Bookmark],
                            content_data_map: Dict[str, ContentData]) -> None:
        """Score all tag candidates based on various criteria"""
        logging.info(f"Scoring {len(self.tag_candidates)} tag candidates")
        
        total_bookmarks = len(bookmarks)
        
        for tag, candidate in self.tag_candidates.items():
            # Content relevance score
            candidate.content_score = self._calculate_content_score(tag, content_data_map)
            
            # URL pattern score
            candidate.url_score = self._calculate_url_score(tag, bookmarks)
            
            # Quality score based on tag characteristics
            candidate.quality_score = self._calculate_quality_score(tag, candidate, total_bookmarks)
    
    def _select_optimal_tags(self) -> List[str]:
        """Select optimal set of tags based on scores and criteria"""
        logging.info("Selecting optimal tag set")
        
        # Filter candidates by minimum criteria
        qualified_candidates = {
            tag: candidate for tag, candidate in self.tag_candidates.items()
            if (candidate.frequency >= self.min_tag_frequency and 
                candidate.total_score >= self.quality_threshold)
        }
        
        logging.info(f"Qualified candidates: {len(qualified_candidates)}/{len(self.tag_candidates)}")
        
        # Sort by total score
        sorted_candidates = sorted(
            qualified_candidates.items(),
            key=lambda x: x[1].total_score,
            reverse=True
        )
        
        # Select top candidates with diversity considerations
        selected_tags = []
        selected_domains = set()
        selected_categories = set()
        
        for tag, candidate in sorted_candidates:
            if len(selected_tags) >= self.target_tag_count:
                break
            
            # Promote diversity by considering tag categories
            tag_category = self._categorize_tag(tag)
            
            # Include if it adds diversity or is high-scoring enough
            if (tag_category not in selected_categories or 
                candidate.total_score > 0.7 or 
                len(selected_tags) < self.target_tag_count * 0.8):
                
                selected_tags.append(tag)
                selected_categories.add(tag_category)
        
        return selected_tags
    
    def _assign_tags_to_bookmarks(self, 
                                bookmarks: List[Bookmark],
                                optimal_tags: List[str],
                                content_data_map: Dict[str, ContentData]) -> Dict[str, List[str]]:
        """Assign optimal tags to individual bookmarks"""
        logging.info("Assigning optimized tags to bookmarks")
        
        optimal_tags_set = set(optimal_tags)
        tag_assignments = {}
        
        for bookmark in bookmarks:
            bookmark_profile = self.bookmark_profiles.get(bookmark.url, {})
            candidate_tags = bookmark_profile.get('tags', set())
            
            # Filter to only optimal tags
            relevant_tags = [tag for tag in candidate_tags if tag in optimal_tags_set]
            
            # Score tags for this specific bookmark
            tag_scores = []
            for tag in relevant_tags:
                score = self._calculate_bookmark_tag_relevance(bookmark, tag, content_data_map)
                tag_scores.append((tag, score))
            
            # Sort by relevance and select top tags
            tag_scores.sort(key=lambda x: x[1], reverse=True)
            assigned_tags = [tag for tag, score in tag_scores[:self.max_tags_per_bookmark]]
            
            # Ensure at least one tag if possible
            if not assigned_tags and relevant_tags:
                assigned_tags = [relevant_tags[0]]
            
            # Add content-type tags if missing
            content_data = content_data_map.get(bookmark.url)
            if content_data and content_data.content_categories:
                for category in content_data.content_categories[:1]:  # Add primary category
                    if category in optimal_tags_set and category not in assigned_tags:
                        if len(assigned_tags) < self.max_tags_per_bookmark:
                            assigned_tags.append(category)
            
            tag_assignments[bookmark.url] = assigned_tags
        
        return tag_assignments
    
    def _parse_existing_tags(self, tags_input: Union[str, List[str]]) -> Set[str]:
        """Parse existing tags from bookmark"""
        if not tags_input:
            return set()
        
        # Handle different tag formats
        tags = set()
        
        # If it's already a list, use it directly
        if isinstance(tags_input, list):
            tag_parts = tags_input
        else:
            # Remove quotes and split by common separators
            clean_tags = tags_input.strip('"\'')
            for separator in [',', ';', '|']:
                if separator in clean_tags:
                    tag_parts = clean_tags.split(separator)
                    break
            else:
                tag_parts = [clean_tags]
        
        for tag in tag_parts:
            tag = tag.strip().lower()
            if tag and len(tag) > 1:
                tags.add(tag)
        
        return tags
    
    def _extract_url_tags(self, url: str) -> Set[str]:
        """Extract tags from URL patterns"""
        tags = set()
        
        try:
            parsed = urlparse(url)
            
            # Domain-based tags
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Major platforms
            platform_tags = {
                'github.com': ['code', 'repository', 'programming'],
                'stackoverflow.com': ['programming', 'qa', 'development'],
                'youtube.com': ['video', 'tutorial', 'entertainment'],
                'medium.com': ['article', 'blog', 'writing'],
                'reddit.com': ['discussion', 'community', 'social'],
                'wikipedia.org': ['reference', 'encyclopedia', 'knowledge'],
                'docs.google.com': ['documentation', 'google', 'office'],
                'developer.mozilla.org': ['web', 'documentation', 'reference']
            }
            
            for platform, platform_tag_list in platform_tags.items():
                if platform in domain:
                    tags.update(platform_tag_list)
                    break
            
            # Extract from path
            path_parts = [part.lower() for part in parsed.path.split('/') if part]
            for part in path_parts[:3]:  # First few path segments
                if len(part) > 2 and part.isalpha():
                    tags.add(part)
            
        except Exception:
            pass
        
        return tags
    
    def _extract_text_tags(self, text: str) -> Set[str]:
        """Extract tags from text content"""
        if not text:
            return set()
        
        tags = set()
        text_lower = text.lower()
        
        # Extract technology keywords
        for category, keywords in self.TECH_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    tags.add(category)
                    tags.add(keyword)
        
        # Extract content type indicators
        for content_type, indicators in self.CONTENT_TYPE_INDICATORS.items():
            for indicator in indicators:
                if indicator in text_lower:
                    tags.add(content_type)
        
        # Extract capitalized words (potential proper nouns)
        words = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
        for word in words:
            word_lower = word.lower()
            if (len(word_lower) > 3 and 
                word_lower not in self.STOP_WORDS and
                not word_lower.isdigit()):
                tags.add(word_lower)
        
        return tags
    
    def _extract_content_tags(self, content_data: ContentData) -> Set[str]:
        """Extract tags from content analysis data"""
        tags = set()
        
        # Add content categories
        tags.update(content_data.content_categories)
        
        # Extract from meta keywords
        if content_data.meta_keywords:
            keyword_tags = self._parse_existing_tags(content_data.meta_keywords)
            tags.update(keyword_tags)
        
        # Extract from headings
        for heading in content_data.headings[:5]:  # First 5 headings
            heading_tags = self._extract_text_tags(heading)
            tags.update(heading_tags)
        
        return tags
    
    def _clean_tags(self, tags: Set[str]) -> Set[str]:
        """Clean and normalize tags"""
        cleaned = set()
        
        for tag in tags:
            # Basic cleaning
            tag = tag.strip().lower()
            tag = re.sub(r'[^\w\-]', '', tag)  # Remove special chars except hyphens
            
            # Skip if too short, too long, or stop word
            if (len(tag) < 2 or len(tag) > 20 or 
                tag in self.STOP_WORDS or
                tag.isdigit()):
                continue
            
            # Normalize common variations
            tag = self._normalize_tag(tag)
            cleaned.add(tag)
        
        return cleaned
    
    def _normalize_tag(self, tag: str) -> str:
        """Normalize tag variations"""
        # Common normalizations
        normalizations = {
            'js': 'javascript',
            'py': 'python',
            'ml': 'machine-learning',
            'ai': 'artificial-intelligence',
            'ui': 'user-interface',
            'ux': 'user-experience',
            'api': 'api',
            'css': 'css',
            'html': 'html'
        }
        
        return normalizations.get(tag, tag)
    
    def _calculate_content_score(self, tag: str, content_data_map: Dict[str, ContentData]) -> float:
        """Calculate content relevance score for tag"""
        if not content_data_map:
            return 0.0
        
        total_score = 0.0
        count = 0
        
        for content_data in content_data_map.values():
            if tag in content_data.content_categories:
                total_score += 1.0
            elif any(tag in heading.lower() for heading in content_data.headings):
                total_score += 0.5
            elif tag in content_data.main_content.lower():
                total_score += 0.3
            
            count += 1
        
        return total_score / count if count > 0 else 0.0
    
    def _calculate_url_score(self, tag: str, bookmarks: List[Bookmark]) -> float:
        """Calculate URL pattern score for tag"""
        url_matches = 0
        
        for bookmark in bookmarks:
            if tag in bookmark.url.lower():
                url_matches += 1
        
        return url_matches / len(bookmarks) if bookmarks else 0.0
    
    def _calculate_quality_score(self, tag: str, candidate: TagCandidate, total_bookmarks: int) -> float:
        """Calculate overall quality score for tag"""
        # Frequency component (0-1)
        frequency_score = min(1.0, candidate.frequency / (total_bookmarks * 0.1))
        
        # Length component (prefer 3-15 character tags)
        length_score = 1.0 if 3 <= len(tag) <= 15 else 0.5
        
        # Alphabetic component (prefer alphabetic tags)
        alpha_score = 1.0 if tag.isalpha() else 0.8
        
        # Technology relevance
        tech_score = 1.2 if any(tag in keywords for keywords in self.TECH_KEYWORDS.values()) else 1.0
        
        return (frequency_score + length_score + alpha_score) * tech_score / 3.2
    
    def _categorize_tag(self, tag: str) -> str:
        """Categorize tag for diversity"""
        for category, keywords in self.TECH_KEYWORDS.items():
            if tag in keywords:
                return category
        
        for content_type, indicators in self.CONTENT_TYPE_INDICATORS.items():
            if tag in indicators:
                return content_type
        
        return 'general'
    
    def _calculate_bookmark_tag_relevance(self, bookmark: Bookmark, tag: str, 
                                        content_data_map: Dict[str, ContentData]) -> float:
        """Calculate tag relevance for specific bookmark"""
        score = 0.0
        
        # Title relevance
        if bookmark.title and tag in bookmark.title.lower():
            score += 0.3
        
        # URL relevance
        if tag in bookmark.url.lower():
            score += 0.2
        
        # Content relevance
        content_data = content_data_map.get(bookmark.url)
        if content_data:
            if tag in content_data.content_categories:
                score += 0.4
            elif any(tag in heading.lower() for heading in content_data.headings):
                score += 0.2
        
        # Existing tag relevance
        if bookmark.tags and isinstance(bookmark.tags, list):
            existing_tags_str = ' '.join(bookmark.tags).lower()
            if tag.lower() in existing_tags_str:
                score += 0.1
        elif bookmark.tags and isinstance(bookmark.tags, str):
            if tag.lower() in bookmark.tags.lower():
                score += 0.1
        
        return min(1.0, score)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return 'unknown'
    
    def _calculate_optimization_stats(self, bookmarks: List[Bookmark], 
                                    tag_assignments: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate optimization statistics"""
        total_bookmarks = len(bookmarks)
        tagged_bookmarks = len([url for url, tags in tag_assignments.items() if tags])
        
        # Tag distribution
        tag_counter = Counter()
        for tags in tag_assignments.values():
            tag_counter.update(tags)
        
        # Calculate statistics
        stats = {
            'total_bookmarks': total_bookmarks,
            'tagged_bookmarks': tagged_bookmarks,
            'coverage_percentage': (tagged_bookmarks / total_bookmarks) * 100 if total_bookmarks else 0,
            'unique_tags_used': len(tag_counter),
            'average_tags_per_bookmark': sum(len(tags) for tags in tag_assignments.values()) / total_bookmarks if total_bookmarks else 0,
            'most_common_tags': tag_counter.most_common(10),
            'candidate_tags_considered': len(self.tag_candidates),
            'optimization_ratio': len(tag_counter) / len(self.tag_candidates) if self.tag_candidates else 0
        }
        
        return stats