"""
Data models for the Bookmark Processor.

This module defines the internal data structures used to represent
bookmarks throughout the processing pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


@dataclass
class BookmarkMetadata:
    """Metadata extracted from webpage content."""

    title: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    canonical_url: Optional[str] = None


@dataclass
class ProcessingStatus:
    """Status information for bookmark processing."""

    url_validated: bool = False
    url_validation_error: Optional[str] = None
    content_extracted: bool = False
    content_extraction_error: Optional[str] = None
    ai_processed: bool = False
    ai_processing_error: Optional[str] = None
    tags_optimized: bool = False
    processing_attempts: int = 0
    last_attempt: Optional[datetime] = None


@dataclass
class Bookmark:
    """
    Internal representation of a bookmark with all processing data.

    This class represents the complete bookmark data structure used
    throughout the processing pipeline, from initial import to final export.
    """

    # Original raindrop.io export data
    id: Optional[str] = None
    title: str = ""
    note: str = ""
    excerpt: str = ""
    url: str = ""
    folder: str = ""
    tags: List[str] = field(default_factory=list)
    created: Optional[datetime] = None
    cover: str = ""
    highlights: str = ""
    favorite: bool = False

    # Enhanced data from processing
    enhanced_description: str = ""
    optimized_tags: List[str] = field(default_factory=list)
    extracted_metadata: Optional[BookmarkMetadata] = None
    processing_status: ProcessingStatus = field(default_factory=ProcessingStatus)

    # Additional fields for processing
    normalized_url: str = ""
    folder_hierarchy: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize computed fields after creation."""
        if not self.normalized_url and self.url:
            self.normalized_url = self._normalize_url(self.url)

        if not self.folder_hierarchy and self.folder:
            self.folder_hierarchy = self._parse_folder_hierarchy(self.folder)

    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL for consistent processing.

        Args:
            url: Raw URL string

        Returns:
            Normalized URL string
        """
        if not url or not url.strip():
            return ""

        url = url.strip()

        # Don't modify special protocols
        if url.startswith(("javascript:", "data:", "mailto:", "ftp:", "file:")):
            return url

        # Add protocol if missing
        if not url.startswith(("http://", "https://")):
            if url.startswith("www."):
                url = f"https://{url}"
            elif "." in url:
                url = f"https://{url}"
            else:
                return url  # Don't modify if it doesn't look like a URL

        try:
            parsed = urlparse(url)

            # Normalize the URL
            normalized = f"{parsed.scheme}://{parsed.netloc.lower()}"

            if parsed.path and parsed.path != "/":
                # Remove trailing slash unless it's the root
                path = parsed.path.rstrip("/")
                if path:
                    normalized += path
                else:
                    normalized += "/"

            if parsed.query:
                # Sort query parameters for consistency
                from urllib.parse import parse_qs, urlencode

                params = parse_qs(parsed.query, keep_blank_values=True)
                sorted_params = sorted(params.items())
                normalized += "?" + urlencode(sorted_params, doseq=True)

            if parsed.fragment:
                normalized += f"#{parsed.fragment}"

            return normalized

        except Exception:
            # If URL parsing fails, return the original
            return url

    def _parse_folder_hierarchy(self, folder: str) -> List[str]:
        """
        Parse folder hierarchy from folder string.

        Args:
            folder: Folder path string (e.g., "Tech/AI/MachineLearning")

        Returns:
            List of folder components
        """
        if not folder or not folder.strip():
            return []

        # Split on forward slash and clean components
        components = []
        for component in folder.split("/"):
            component = component.strip()
            if component:
                components.append(component)

        return components

    def get_folder_path(self) -> str:
        """
        Get the folder path as a string.

        Returns:
            Folder path string
        """
        return "/".join(self.folder_hierarchy) if self.folder_hierarchy else ""

    def get_effective_title(self) -> str:
        """
        Get the most appropriate title for this bookmark.

        Returns:
            Best available title
        """
        # Priority: title -> extracted metadata title -> URL domain
        if self.title and self.title.strip():
            return self.title.strip()

        if (
            self.extracted_metadata
            and self.extracted_metadata.title
            and self.extracted_metadata.title.strip()
        ):
            return self.extracted_metadata.title.strip()

        # Fallback to URL domain
        if self.url:
            try:
                parsed = urlparse(self.url)
                return parsed.netloc or self.url
            except Exception:
                return self.url

        return "Untitled Bookmark"

    def get_effective_description(self) -> str:
        """
        Get the most appropriate description for this bookmark.

        Returns:
            Best available description
        """
        # Priority: enhanced description -> note -> excerpt -> metadata description
        if self.enhanced_description and self.enhanced_description.strip():
            return self.enhanced_description.strip()

        if self.note and self.note.strip():
            return self.note.strip()

        if self.excerpt and self.excerpt.strip():
            return self.excerpt.strip()

        if (
            self.extracted_metadata
            and self.extracted_metadata.description
            and self.extracted_metadata.description.strip()
        ):
            return self.extracted_metadata.description.strip()

        return ""

    def get_all_tags(self) -> List[str]:
        """
        Get all available tags (original + optimized).

        Returns:
            Combined list of unique tags
        """
        all_tags = set()

        # Add original tags
        all_tags.update(self.tags)

        # Add optimized tags
        all_tags.update(self.optimized_tags)

        # Add metadata keywords as potential tags
        if self.extracted_metadata:
            all_tags.update(self.extracted_metadata.keywords)

        # Clean and filter tags
        cleaned_tags = []
        for tag in all_tags:
            if isinstance(tag, str) and tag.strip():
                cleaned_tag = tag.strip().lower()
                if cleaned_tag and len(cleaned_tag) > 1:
                    cleaned_tags.append(cleaned_tag)

        return sorted(list(set(cleaned_tags)))

    def get_final_tags(self, max_tags: int = 5) -> List[str]:
        """
        Get the final optimized tags for export.

        Args:
            max_tags: Maximum number of tags to return

        Returns:
            List of final tags for this bookmark
        """
        # Use optimized tags if available, otherwise fall back to original
        if self.optimized_tags:
            tags = self.optimized_tags[:max_tags]
        else:
            tags = self.tags[:max_tags]

        # Clean tags
        cleaned_tags = []
        for tag in tags:
            if isinstance(tag, str) and tag.strip():
                cleaned_tag = tag.strip()
                if cleaned_tag:
                    cleaned_tags.append(cleaned_tag)

        return cleaned_tags

    def is_valid(self) -> bool:
        """
        Check if this bookmark has minimum required data.

        Returns:
            True if bookmark is valid for processing
        """
        return bool(self.url and self.url.strip() and self.get_effective_title())

    def to_export_dict(self) -> Dict[str, Any]:
        """
        Convert bookmark to dictionary for CSV export.

        Returns:
            Dictionary suitable for raindrop.io import format
        """
        tags = self.get_final_tags()

        # Format tags according to raindrop.io requirements
        if len(tags) == 0:
            formatted_tags = ""
        elif len(tags) == 1:
            formatted_tags = tags[0]
        else:
            formatted_tags = ", ".join(tags)

        return {
            "url": self.url,
            "folder": self.get_folder_path(),
            "title": self.get_effective_title(),
            "note": self.get_effective_description(),
            "tags": formatted_tags,
            "created": self.created.isoformat() if self.created else "",
        }

    @classmethod
    def from_raindrop_export(cls, row_data: Dict[str, Any]) -> "Bookmark":
        """
        Create bookmark from raindrop.io export CSV row.

        Args:
            row_data: Dictionary from CSV row

        Returns:
            Bookmark object
        """
        # Parse tags from string format
        tags = []
        tags_str = str(row_data.get("tags", "")).strip()
        if tags_str:
            # Handle both quoted and unquoted tag formats
            if tags_str.startswith('"') and tags_str.endswith('"'):
                tags_str = tags_str[1:-1]  # Remove quotes

            # Split by comma and clean
            for tag in tags_str.split(","):
                tag = tag.strip()
                if tag:
                    tags.append(tag)

        # Parse created date
        created = None
        created_str = str(row_data.get("created", "")).strip()
        if created_str:
            try:
                from datetime import datetime

                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        # Parse favorite boolean
        favorite = False
        favorite_str = str(row_data.get("favorite", "")).strip().lower()
        if favorite_str in ("true", "1", "yes"):
            favorite = True

        return cls(
            id=str(row_data.get("id", "")).strip(),
            title=str(row_data.get("title", "")).strip(),
            note=str(row_data.get("note", "")).strip(),
            excerpt=str(row_data.get("excerpt", "")).strip(),
            url=str(row_data.get("url", "")).strip(),
            folder=str(row_data.get("folder", "")).strip(),
            tags=tags,
            created=created,
            cover=str(row_data.get("cover", "")).strip(),
            highlights=str(row_data.get("highlights", "")).strip(),
            favorite=favorite,
        )

    def copy(self) -> "Bookmark":
        """Create a copy of this bookmark"""
        return Bookmark(
            id=self.id,
            title=self.title,
            note=self.note,
            excerpt=self.excerpt,
            url=self.url,
            folder=self.folder,
            tags=self.tags.copy(),
            created=self.created,
            cover=self.cover,
            highlights=self.highlights,
            favorite=self.favorite,
            enhanced_description=self.enhanced_description,
            optimized_tags=self.optimized_tags.copy(),
            extracted_metadata=self.extracted_metadata,
            processing_status=self.processing_status,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert bookmark to dictionary for serialization.

        Returns:
            Dictionary representation of bookmark
        """
        return {
            "id": self.id,
            "title": self.title,
            "note": self.note,
            "excerpt": self.excerpt,
            "url": self.url,
            "folder": self.folder,
            "tags": self.tags,
            "created": self.created.isoformat() if self.created else "",
            "cover": self.cover,
            "highlights": self.highlights,
            "favorite": self.favorite,
            "enhanced_description": self.enhanced_description,
            "optimized_tags": self.optimized_tags,
            "normalized_url": self.normalized_url,
            "folder_hierarchy": self.folder_hierarchy,
        }


@dataclass
class ProcessingResults:
    """Container for batch processing results and statistics."""

    total_bookmarks: int = 0
    processed_bookmarks: int = 0
    valid_bookmarks: int = 0
    invalid_bookmarks: int = 0
    errors: List[str] = field(default_factory=list)
    processing_time: float = 0.0

    # Detailed statistics
    url_validation_success: int = 0
    url_validation_failed: int = 0
    ai_processing_success: int = 0
    ai_processing_failed: int = 0
    tags_optimized: int = 0
    duplicates_removed: int = 0

    def __str__(self) -> str:
        return (
            f"ProcessingResults("
            f"total={self.total_bookmarks}, "
            f"processed={self.processed_bookmarks}, "
            f"valid={self.valid_bookmarks}, "
            f"invalid={self.invalid_bookmarks}, "
            f"errors={len(self.errors)}, "
            f"time={self.processing_time:.2f}s)"
        )

    def get_success_rate(self) -> float:
        """Get overall processing success rate as percentage."""
        if self.total_bookmarks == 0:
            return 0.0
        return (self.valid_bookmarks / self.total_bookmarks) * 100

    def get_url_validation_rate(self) -> float:
        """Get URL validation success rate as percentage."""
        total_attempts = self.url_validation_success + self.url_validation_failed
        if total_attempts == 0:
            return 0.0
        return (self.url_validation_success / total_attempts) * 100

    def get_ai_processing_rate(self) -> float:
        """Get AI processing success rate as percentage."""
        total_attempts = self.ai_processing_success + self.ai_processing_failed
        if total_attempts == 0:
            return 0.0
        return (self.ai_processing_success / total_attempts) * 100
