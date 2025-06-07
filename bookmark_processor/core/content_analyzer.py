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
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
    logging.warning("BeautifulSoup4 not available. Content analysis will be limited.")

from ..utils.browser_simulator import BrowserSimulator


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
    ):
        """
        Initialize content analyzer.

        Args:
            timeout: Request timeout in seconds
            max_content_length: Maximum content length to extract
            user_agent_rotation: Whether to rotate user agents
        """
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.browser_simulator = BrowserSimulator(rotate_agents=user_agent_rotation)

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

    def close(self) -> None:
        """Clean up resources"""
        if hasattr(self, "session"):
            self.session.close()
