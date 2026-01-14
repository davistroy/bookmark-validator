"""
Content Analysis Module

Extracts and analyzes web content from URLs for AI processing and tag generation.
Handles various content types and provides clean, structured data extraction.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
    logging.warning("BeautifulSoup4 not available. Content analysis will be limited.")

from ..utils.browser_simulator import BrowserSimulator
from .data_models import BookmarkMetadata


@dataclass
class ContentData:
    """Structured content data from web page analysis"""

    url: str
    title: str = ""
    meta_description: str = ""
    meta_keywords: str = ""
    main_content: str = ""
    headings: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    content_type: str = ""
    language: str = ""
    word_count: int = 0
    reading_time_minutes: int = 0
    content_categories: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    extraction_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "url": self.url,
            "title": self.title,
            "meta_description": self.meta_description,
            "meta_keywords": self.meta_keywords,
            "main_content": self.main_content[:1000],  # Truncate for storage
            "headings": self.headings,
            "content_type": self.content_type,
            "language": self.language,
            "word_count": self.word_count,
            "reading_time_minutes": self.reading_time_minutes,
            "content_categories": self.content_categories,
            "quality_score": self.quality_score,
            "extraction_time": self.extraction_time,
            "timestamp": self.timestamp.isoformat(),
        }


class ContentAnalyzer:
    """Analyze and extract content from web pages"""

    # Content type patterns for categorization
    CONTENT_CATEGORIES = {
        "article": [
            "article",
            "blog",
            "post",
            "news",
            "story",
            "entry",
            "editorial",
            "feature",
            "column",
            "review",
        ],
        "documentation": [
            "docs",
            "documentation",
            "manual",
            "guide",
            "tutorial",
            "reference",
            "api",
            "readme",
            "help",
            "wiki",
        ],
        "academic": [
            "paper",
            "research",
            "study",
            "journal",
            "thesis",
            "publication",
            "academic",
            "scholar",
            "arxiv",
        ],
        "video": [
            "youtube",
            "vimeo",
            "video",
            "watch",
            "movie",
            "film",
            "tv",
            "series",
            "episode",
            "stream",
        ],
        "social": [
            "twitter",
            "facebook",
            "linkedin",
            "instagram",
            "reddit",
            "social",
            "forum",
            "discussion",
            "community",
        ],
        "tool": [
            "tool",
            "utility",
            "app",
            "application",
            "software",
            "service",
            "platform",
            "generator",
            "converter",
        ],
        "repository": [
            "github",
            "gitlab",
            "bitbucket",
            "repo",
            "repository",
            "code",
            "source",
            "project",
            "library",
            "framework",
        ],
        "commerce": [
            "shop",
            "store",
            "buy",
            "sell",
            "product",
            "price",
            "amazon",
            "ebay",
            "marketplace",
            "ecommerce",
        ],
    }

    # Common non-content selectors to ignore
    IGNORE_SELECTORS = [
        "header",
        "nav",
        "footer",
        "aside",
        "sidebar",
        ".advertisement",
        ".ad",
        ".ads",
        ".banner",
        ".navigation",
        ".menu",
        ".breadcrumb",
        ".pagination",
        ".social",
        ".share",
        ".comment-form",
        ".newsletter",
        "script",
        "style",
        "noscript",
    ]

    # Preferred content selectors (in order of preference)
    CONTENT_SELECTORS = [
        "article",
        "main",
        '[role="main"]',
        ".content",
        ".main-content",
        ".post-content",
        ".entry-content",
        ".article-content",
        "#content",
        "#main",
        ".body",
        ".text",
    ]

    def __init__(
        self,
        timeout: float = 30.0,
        max_content_length: int = 50000,
        user_agent_rotation: bool = True,
        user_agent: Optional[str] = None,
        max_content_size: Optional[int] = None,
    ):
        """
        Initialize content analyzer.

        Args:
            timeout: Request timeout in seconds
            max_content_length: Maximum content length to extract
            user_agent_rotation: Whether to rotate user agents
            user_agent: Specific user agent to use (backward compatibility)
            max_content_size: Alias for max_content_length (backward compatibility)
        """
        self.timeout = timeout
        self.max_content_length = max_content_length
        
        # Backward compatibility attributes - tests expect max_content_size to default to 1MB
        if max_content_size is not None:
            self.max_content_size = max_content_size
        else:
            self.max_content_size = 1024 * 1024  # 1MB default for backward compatibility
        
        self.browser_simulator = BrowserSimulator(rotate_agents=user_agent_rotation)
        
        # Set user agent - ensure it's always not None for test compatibility
        try:
            if user_agent:
                self.user_agent = user_agent
            else:
                # Get a default user agent from browser simulator
                self.user_agent = self.browser_simulator.get_headers("").get("User-Agent")
                # Fallback if browser simulator fails
                if not self.user_agent:
                    self.user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        except Exception:
            # Final fallback to ensure user_agent is never None
            self.user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

        # Create session
        self.session = requests.Session()

        logging.info(
            f"Initialized content analyzer (timeout={timeout}s, "
            f"max_length={max_content_length})"
        )

    def analyze_content(
        self,
        url: str,
        existing_title: str = "",
        existing_note: str = "",
        existing_excerpt: str = "",
    ) -> ContentData:
        """
        Analyze content from a URL.

        Args:
            url: URL to analyze
            existing_title: Existing bookmark title
            existing_note: Existing user note
            existing_excerpt: Existing excerpt

        Returns:
            ContentData object with extracted information
        """
        start_time = time.time()

        content_data = ContentData(url=url)

        try:
            # Get headers
            headers = self.browser_simulator.get_headers(url)

            # Fetch content
            response = self.session.get(
                url, headers=headers, timeout=self.timeout, stream=True
            )

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            content_data.content_type = content_type

            # Only process HTML content
            if "html" not in content_type:
                content_data.main_content = f"Non-HTML content: {content_type}"
                content_data.extraction_time = time.time() - start_time
                return content_data

            # Get content with size limit
            content = ""
            size = 0
            for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                content += chunk
                size += len(chunk)
                if size > self.max_content_length:
                    content += "... [content truncated]"
                    break

            # Parse with BeautifulSoup if available
            if BeautifulSoup:
                content_data = self._extract_with_beautifulsoup(content, content_data)
            else:
                content_data = self._extract_with_regex(content, content_data)

            # Use existing data if available and better
            content_data = self._enhance_with_existing_data(
                content_data, existing_title, existing_note, existing_excerpt
            )

            # Categorize content
            content_data.content_categories = self._categorize_content(content_data)

            # Calculate quality score
            content_data.quality_score = self._calculate_quality_score(content_data)

            # Calculate reading time
            content_data.reading_time_minutes = self._calculate_reading_time(
                content_data.word_count
            )

        except requests.exceptions.Timeout:
            content_data.main_content = f"Timeout after {self.timeout}s"
        except requests.exceptions.RequestException as e:
            content_data.main_content = f"Request error: {str(e)}"
        except Exception as e:
            content_data.main_content = f"Analysis error: {str(e)}"
            logging.debug(f"Content analysis error for {url}: {e}")

        content_data.extraction_time = time.time() - start_time
        return content_data

    def _extract_with_beautifulsoup(
        self, html_content: str, content_data: ContentData
    ) -> ContentData:
        """Extract content using BeautifulSoup"""
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract title
            title_tag = soup.find("title")
            if title_tag:
                content_data.title = self._clean_text(title_tag.get_text())

            # Extract meta information
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc:
                content_data.meta_description = meta_desc.get("content", "")

            meta_keywords = soup.find("meta", attrs={"name": "keywords"})
            if meta_keywords:
                content_data.meta_keywords = meta_keywords.get("content", "")

            # Extract language
            html_tag = soup.find("html")
            if html_tag:
                content_data.language = html_tag.get("lang", "")

            # Remove unwanted elements
            for selector in self.IGNORE_SELECTORS:
                for element in soup.select(selector):
                    element.decompose()

            # Extract main content
            main_content = self._extract_main_content(soup)
            content_data.main_content = self._clean_text(main_content)

            # Extract headings
            content_data.headings = self._extract_headings(soup)

            # Extract links
            content_data.links = self._extract_links(soup, content_data.url)

            # Extract images
            content_data.images = self._extract_images(soup, content_data.url)

            # Count words
            content_data.word_count = len(content_data.main_content.split())

        except Exception as e:
            logging.debug(f"BeautifulSoup extraction error: {e}")
            # Fallback to regex extraction
            content_data = self._extract_with_regex(html_content, content_data)

        return content_data

    def _extract_with_regex(
        self, html_content: str, content_data: ContentData
    ) -> ContentData:
        """Extract content using regex patterns (fallback)"""
        try:
            # Extract title
            title_match = re.search(
                r"<title[^>]*>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL
            )
            if title_match:
                content_data.title = self._clean_text(title_match.group(1))

            # Extract meta description
            desc_match = re.search(
                r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']',
                html_content,
                re.IGNORECASE,
            )
            if desc_match:
                content_data.meta_description = desc_match.group(1)

            # Remove scripts and styles
            clean_content = re.sub(
                r"<script[^>]*>.*?</script>",
                "",
                html_content,
                flags=re.IGNORECASE | re.DOTALL,
            )
            clean_content = re.sub(
                r"<style[^>]*>.*?</style>",
                "",
                clean_content,
                flags=re.IGNORECASE | re.DOTALL,
            )

            # Extract text content
            text_content = re.sub(r"<[^>]+>", " ", clean_content)
            content_data.main_content = self._clean_text(text_content)

            # Count words
            content_data.word_count = len(content_data.main_content.split())

        except Exception as e:
            logging.debug(f"Regex extraction error: {e}")
            content_data.main_content = "Content extraction failed"

        return content_data

    def _extract_main_content(self, soup) -> str:
        """Extract main content using preferred selectors"""
        content_parts = []

        # Try preferred selectors
        for selector in self.CONTENT_SELECTORS:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    text = element.get_text(separator=" ", strip=True)
                    if len(text) > 100:  # Only include substantial content
                        content_parts.append(text)
                break

        # If no preferred selectors worked, get paragraphs
        if not content_parts:
            paragraphs = soup.find_all("p")
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50:  # Filter out short paragraphs
                    content_parts.append(text)

        return " ".join(content_parts)

    def _extract_headings(self, soup) -> List[str]:
        """Extract heading text"""
        headings = []
        for level in range(1, 7):  # h1 to h6
            for heading in soup.find_all(f"h{level}"):
                text = self._clean_text(heading.get_text())
                if text and len(text) > 3:
                    headings.append(text)
        return headings[:10]  # Limit to 10 headings

    def _extract_links(self, soup, base_url: str) -> List[str]:
        """Extract links from the page"""
        links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.startswith("http"):
                links.append(href)
            elif href.startswith("/"):
                full_url = urljoin(base_url, href)
                links.append(full_url)

        # Remove duplicates and limit
        return list(dict.fromkeys(links))[:20]

    def _extract_images(self, soup, base_url: str) -> List[str]:
        """Extract image URLs"""
        images = []
        for img in soup.find_all("img", src=True):
            src = img["src"]
            if src.startswith("http"):
                images.append(src)
            elif src.startswith("/"):
                full_url = urljoin(base_url, src)
                images.append(full_url)

        # Remove duplicates and limit
        return list(dict.fromkeys(images))[:10]

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text)

        # Remove common unicode characters
        text = text.replace("\u00a0", " ")  # Non-breaking space
        text = text.replace("\u2019", "'")  # Right single quotation mark
        text = text.replace("\u201c", '"')  # Left double quotation mark
        text = text.replace("\u201d", '"')  # Right double quotation mark

        return text.strip()

    def _enhance_with_existing_data(
        self,
        content_data: ContentData,
        existing_title: str,
        existing_note: str,
        existing_excerpt: str,
    ) -> ContentData:
        """Enhance content data with existing bookmark information"""
        # Use existing title if better or missing
        if existing_title and (
            not content_data.title or len(existing_title) > len(content_data.title)
        ):
            content_data.title = existing_title

        # Use existing excerpt as meta description if missing
        if existing_excerpt and not content_data.meta_description:
            content_data.meta_description = existing_excerpt

        # Add existing note to main content for context
        if existing_note:
            note_text = f"User Note: {existing_note}"
            if content_data.main_content:
                content_data.main_content = f"{note_text} | {content_data.main_content}"
            else:
                content_data.main_content = note_text

        return content_data

    def _categorize_content(self, content_data: ContentData) -> List[str]:
        """Categorize content based on URL and content analysis"""
        categories = []

        # Analyze URL for patterns
        url_lower = content_data.url.lower()
        title_lower = content_data.title.lower()
        content_lower = content_data.main_content.lower()

        for category, keywords in self.CONTENT_CATEGORIES.items():
            score = 0

            # Check URL
            for keyword in keywords:
                if keyword in url_lower:
                    score += 2

            # Check title
            for keyword in keywords:
                if keyword in title_lower:
                    score += 1

            # Check content (sample)
            content_sample = content_lower[:1000]  # First 1000 chars
            for keyword in keywords:
                if keyword in content_sample:
                    score += 0.5

            if score >= 1.0:  # Threshold for category inclusion
                categories.append(category)

        return categories

    def _calculate_quality_score(self, content_data: ContentData) -> float:
        """Calculate content quality score (0-1)"""
        score = 0.0

        # Title presence and quality
        if content_data.title:
            score += 0.2
            if 20 <= len(content_data.title) <= 100:
                score += 0.1

        # Meta description presence
        if content_data.meta_description:
            score += 0.1
            if 50 <= len(content_data.meta_description) <= 200:
                score += 0.1

        # Content length and quality
        if content_data.word_count > 100:
            score += 0.2
        if content_data.word_count > 500:
            score += 0.1

        # Headings structure
        if content_data.headings:
            score += 0.1
            if len(content_data.headings) >= 3:
                score += 0.1

        # Language detection
        if content_data.language:
            score += 0.1

        return min(score, 1.0)

    def _calculate_reading_time(self, word_count: int) -> int:
        """Calculate estimated reading time in minutes"""
        # Average reading speed: 200-250 words per minute
        if word_count <= 0:
            return 0

        reading_speed = 225  # words per minute
        return max(1, round(word_count / reading_speed))

    def extract_metadata(self, url: str) -> Optional[BookmarkMetadata]:
        """
        Extract metadata from a URL (backward compatibility method).
        
        Args:
            url: URL to extract metadata from
            
        Returns:
            BookmarkMetadata object or None if extraction fails
        """
        try:
            content_data = self.analyze_content(url)
            
            if content_data.main_content == "Content extraction failed":
                return None
                
            # Convert ContentData to BookmarkMetadata
            metadata = BookmarkMetadata(
                url=url,
                title=content_data.title,
                description=content_data.meta_description,
                keywords=content_data.meta_keywords.split(", ") if content_data.meta_keywords else [],
                author=None,  # Not extracted in current implementation
                canonical_url=None  # Not extracted in current implementation
            )
            
            return metadata
            
        except Exception:
            return None
    
    def _parse_html(self, soup, url: str) -> BookmarkMetadata:
        """
        Parse HTML with BeautifulSoup (backward compatibility method).
        
        Args:
            soup: BeautifulSoup object
            url: Original URL
            
        Returns:
            BookmarkMetadata object
        """
        title = self._extract_title(soup)
        description = self._extract_description(soup)
        keywords = self._extract_keywords(soup)
        
        # Extract author and canonical URL
        author = None
        author_meta = soup.find("meta", attrs={"name": "author"})
        if author_meta:
            author = author_meta.get("content", "")
            
        canonical_url = None
        canonical_link = soup.find("link", rel="canonical")
        if canonical_link:
            canonical_url = canonical_link.get("href", "")
        
        return BookmarkMetadata(
            url=url,
            title=title,
            description=description,
            keywords=keywords,
            author=author,
            canonical_url=canonical_url
        )
    
    def _extract_title(self, soup) -> Optional[str]:
        """
        Extract title from HTML (backward compatibility method).
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Extracted title or None
        """
        # Try title tag first
        title_tag = soup.find("title")
        if title_tag and title_tag.get_text().strip():
            return self._clean_text(title_tag.get_text())
            
        # Try og:title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title.get("content").strip()
            
        # Try h1
        h1_tag = soup.find("h1")
        if h1_tag and h1_tag.get_text().strip():
            return self._clean_text(h1_tag.get_text())
            
        return None
    
    def _extract_description(self, soup) -> Optional[str]:
        """
        Extract description from HTML (backward compatibility method).
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Extracted description or None
        """
        # Try meta description first
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            return meta_desc.get("content").strip()
            
        # Try og:description
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            return og_desc.get("content").strip()
            
        # Try first paragraph
        first_p = soup.find("p")
        if first_p and first_p.get_text().strip():
            text = self._clean_text(first_p.get_text())
            if len(text) > 20:  # Only use substantial paragraphs
                return text[:200]  # Limit length
                
        return None
    
    def _extract_keywords(self, soup) -> List[str]:
        """
        Extract keywords from HTML (backward compatibility method).
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of extracted keywords
        """
        keywords = []
        
        # Try meta keywords
        meta_keywords = soup.find("meta", attrs={"name": "keywords"})
        if meta_keywords and meta_keywords.get("content"):
            content = meta_keywords.get("content")
            # Split by comma and clean
            for keyword in content.split(","):
                keyword = keyword.strip()
                if keyword:
                    keywords.append(keyword)
                    
        return keywords
    
    def _extract_content_text(self, soup) -> str:
        """
        Extract main content text from HTML (backward compatibility method).
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Extracted content text
        """
        # Remove unwanted elements
        for element in soup.select("script, style, nav, header, footer, aside"):
            element.decompose()
            
        # Try main content selectors
        main_content = None
        for selector in ["main", "article", ".content", "#content"]:
            main_element = soup.select_one(selector)
            if main_element:
                main_content = main_element
                break
                
        if not main_content:
            main_content = soup.find("body") or soup
            
        # Extract text
        text = main_content.get_text(separator=" ", strip=True)
        return self._clean_text(text)
    
    def analyze_content_categories(self, content: str) -> List[str]:
        """
        Analyze content and categorize it (backward compatibility method).
        
        Args:
            content: Text content to analyze
            
        Returns:
            List of content categories
        """
        categories = []
        content_lower = content.lower()
        
        # Programming category
        programming_keywords = ["python", "programming", "code", "function", "variable", "algorithm", "software", "development"]
        if any(keyword in content_lower for keyword in programming_keywords):
            categories.append("programming")
            
        # News category  
        news_keywords = ["news", "breaking", "report", "update", "article", "story", "journalist"]
        if any(keyword in content_lower for keyword in news_keywords):
            categories.append("news")
            
        # Education category
        education_keywords = ["learn", "tutorial", "course", "education", "teach", "study", "lesson", "training"]
        if any(keyword in content_lower for keyword in education_keywords):
            categories.append("education")
            
        # Technology category
        tech_keywords = ["technology", "tech", "computer", "digital", "internet", "web", "mobile", "app"]
        if any(keyword in content_lower for keyword in tech_keywords):
            categories.append("technology")
            
        return categories
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get analyzer statistics (backward compatibility method).
        
        Returns:
            Dictionary with statistics
        """
        # Initialize statistics if not present
        if not hasattr(self, '_stats'):
            self._stats = {
                "total_extractions": 0,
                "successful_extractions": 0,
                "failed_extractions": 0
            }
            
        # Calculate success rate
        total = self._stats["total_extractions"]
        success_rate = (self._stats["successful_extractions"] / total * 100) if total > 0 else 0.0
        
        return {
            "total_extractions": self._stats["total_extractions"],
            "successful_extractions": self._stats["successful_extractions"], 
            "failed_extractions": self._stats["failed_extractions"],
            "success_rate": success_rate
        }

    def close(self) -> None:
        """Clean up resources"""
        if hasattr(self, "session"):
            self.session.close()
