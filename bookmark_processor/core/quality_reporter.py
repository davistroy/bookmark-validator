"""
Quality Assessment Reporter for Bookmark Processing.

This module provides comprehensive quality assessment reporting for
processed bookmarks, including metrics calculation, report generation,
and export of items needing manual review.
"""

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..utils.report_generator import ReportGenerator, ReportSection
from ..utils.report_styles import ReportStyle, ICONS
from .data_models import Bookmark, ProcessingResults


@dataclass
class DescriptionMetrics:
    """Metrics for description enhancement quality."""

    ai_enhanced_count: int = 0
    excerpt_used_count: int = 0
    title_fallback_count: int = 0
    meta_description_count: int = 0
    no_description_count: int = 0
    total_count: int = 0
    confidence_scores: List[float] = field(default_factory=list)

    @property
    def ai_enhanced_percentage(self) -> float:
        """Get percentage of bookmarks enhanced by AI."""
        if self.total_count == 0:
            return 0.0
        return (self.ai_enhanced_count / self.total_count) * 100

    @property
    def excerpt_used_percentage(self) -> float:
        """Get percentage using existing excerpt."""
        if self.total_count == 0:
            return 0.0
        return (self.excerpt_used_count / self.total_count) * 100

    @property
    def title_fallback_percentage(self) -> float:
        """Get percentage falling back to title."""
        if self.total_count == 0:
            return 0.0
        return (self.title_fallback_count / self.total_count) * 100

    @property
    def average_confidence(self) -> float:
        """Get average confidence score."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)


@dataclass
class TagMetrics:
    """Metrics for tag analysis."""

    unique_tags: Set[str] = field(default_factory=set)
    bookmarks_with_tags: int = 0
    bookmarks_without_tags: int = 0
    total_bookmarks: int = 0
    tag_counts: List[int] = field(default_factory=list)
    tag_frequency: Dict[str, int] = field(default_factory=dict)

    @property
    def unique_tag_count(self) -> int:
        """Get count of unique tags."""
        return len(self.unique_tags)

    @property
    def tagged_percentage(self) -> float:
        """Get percentage of bookmarks with tags."""
        if self.total_bookmarks == 0:
            return 0.0
        return (self.bookmarks_with_tags / self.total_bookmarks) * 100

    @property
    def avg_tags_per_bookmark(self) -> float:
        """Get average number of tags per bookmark."""
        if not self.tag_counts:
            return 0.0
        return sum(self.tag_counts) / len(self.tag_counts)

    @property
    def tag_coverage_score(self) -> float:
        """
        Calculate tag coverage score (0-1).

        Higher score means better tag distribution.
        """
        if self.total_bookmarks == 0 or self.unique_tag_count == 0:
            return 0.0

        # Coverage is based on:
        # 1. Percentage of bookmarks with tags (40%)
        # 2. Average tags per bookmark normalized to target of 3-5 (30%)
        # 3. Diversity of tags (30%)

        tagged_ratio = self.bookmarks_with_tags / self.total_bookmarks

        avg_tags = self.avg_tags_per_bookmark
        # Optimal is 3-5 tags, score drops outside this range
        if 3 <= avg_tags <= 5:
            tag_count_score = 1.0
        elif avg_tags < 3:
            tag_count_score = avg_tags / 3
        else:  # avg_tags > 5
            tag_count_score = max(0.0, 1.0 - (avg_tags - 5) / 5)

        # Diversity: ratio of unique tags to total tag assignments
        total_tags = sum(self.tag_counts)
        if total_tags == 0:
            diversity_score = 0.0
        else:
            diversity_score = min(1.0, self.unique_tag_count / (total_tags * 0.3))

        return (tagged_ratio * 0.4) + (tag_count_score * 0.3) + (diversity_score * 0.3)


@dataclass
class FolderMetrics:
    """Metrics for folder organization."""

    unique_folders: Set[str] = field(default_factory=set)
    folder_depths: List[int] = field(default_factory=list)
    bookmarks_reorganized: int = 0
    total_bookmarks: int = 0
    folder_distribution: Dict[str, int] = field(default_factory=dict)

    @property
    def total_folders(self) -> int:
        """Get total number of unique folders."""
        return len(self.unique_folders)

    @property
    def max_depth(self) -> int:
        """Get maximum folder depth."""
        if not self.folder_depths:
            return 0
        return max(self.folder_depths)

    @property
    def avg_depth(self) -> float:
        """Get average folder depth."""
        if not self.folder_depths:
            return 0.0
        return sum(self.folder_depths) / len(self.folder_depths)

    @property
    def reorganized_percentage(self) -> float:
        """Get percentage of bookmarks reorganized."""
        if self.total_bookmarks == 0:
            return 0.0
        return (self.bookmarks_reorganized / self.total_bookmarks) * 100

    @property
    def organization_coherence(self) -> float:
        """
        Calculate organization coherence score (0-1).

        Higher score means better folder organization.
        """
        if self.total_bookmarks == 0:
            return 0.0

        # Coherence based on:
        # 1. Reasonable folder count (not too many, not too few) (40%)
        # 2. Balanced distribution (30%)
        # 3. Reasonable depth (30%)

        # Optimal folder count is roughly sqrt(total_bookmarks) * 2
        optimal_folders = (self.total_bookmarks ** 0.5) * 2
        if self.total_folders == 0:
            folder_count_score = 0.0
        else:
            ratio = self.total_folders / optimal_folders
            folder_count_score = 1.0 - abs(1.0 - ratio) * 0.5
            folder_count_score = max(0.0, min(1.0, folder_count_score))

        # Distribution balance
        if self.folder_distribution:
            counts = list(self.folder_distribution.values())
            avg_count = sum(counts) / len(counts)
            if avg_count > 0:
                variance = sum((c - avg_count) ** 2 for c in counts) / len(counts)
                cv = (variance ** 0.5) / avg_count  # Coefficient of variation
                distribution_score = max(0.0, 1.0 - cv * 0.3)
            else:
                distribution_score = 0.0
        else:
            distribution_score = 0.0

        # Depth score (optimal depth is 2-3)
        if self.folder_depths:
            avg_d = self.avg_depth
            if 2 <= avg_d <= 3:
                depth_score = 1.0
            elif avg_d < 2:
                depth_score = avg_d / 2
            else:
                depth_score = max(0.0, 1.0 - (avg_d - 3) / 3)
        else:
            depth_score = 0.0

        return (folder_count_score * 0.4) + (distribution_score * 0.3) + (depth_score * 0.3)


@dataclass
class AttentionItems:
    """Items that need manual attention."""

    low_confidence_descriptions: List[Bookmark] = field(default_factory=list)
    untagged_bookmarks: List[Bookmark] = field(default_factory=list)
    invalid_urls: List[Bookmark] = field(default_factory=list)
    missing_titles: List[Bookmark] = field(default_factory=list)
    processing_errors: List[Tuple[Bookmark, str]] = field(default_factory=list)

    @property
    def total_review_items(self) -> int:
        """Get total items needing review."""
        return (
            len(self.low_confidence_descriptions) +
            len(self.untagged_bookmarks) +
            len(self.invalid_urls) +
            len(self.missing_titles) +
            len(self.processing_errors)
        )

    def get_all_items_for_review(self) -> List[Bookmark]:
        """Get all unique bookmarks needing review."""
        seen_urls: Set[str] = set()
        items: List[Bookmark] = []

        for bookmark in (
            self.low_confidence_descriptions +
            self.untagged_bookmarks +
            self.invalid_urls +
            self.missing_titles +
            [b for b, _ in self.processing_errors]
        ):
            if bookmark.url not in seen_urls:
                seen_urls.add(bookmark.url)
                items.append(bookmark)

        return items


@dataclass
class QualityMetrics:
    """Complete quality metrics for processed bookmarks."""

    description_metrics: DescriptionMetrics = field(default_factory=DescriptionMetrics)
    tag_metrics: TagMetrics = field(default_factory=TagMetrics)
    folder_metrics: FolderMetrics = field(default_factory=FolderMetrics)
    attention_items: AttentionItems = field(default_factory=AttentionItems)

    # Processing statistics
    total_processed: int = 0
    successful_count: int = 0
    failed_count: int = 0
    processing_time_seconds: float = 0.0

    # Validation statistics
    urls_validated: int = 0
    urls_valid: int = 0
    urls_invalid: int = 0

    @property
    def overall_quality_score(self) -> float:
        """
        Calculate overall quality score (0-1).

        Combines description, tag, and folder quality.
        """
        desc_score = self.description_metrics.average_confidence
        tag_score = self.tag_metrics.tag_coverage_score
        folder_score = self.folder_metrics.organization_coherence

        # Weight: descriptions 40%, tags 35%, folders 25%
        return (desc_score * 0.4) + (tag_score * 0.35) + (folder_score * 0.25)

    @property
    def success_rate(self) -> float:
        """Get processing success rate."""
        if self.total_processed == 0:
            return 0.0
        return (self.successful_count / self.total_processed) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "description": {
                "ai_enhanced_count": self.description_metrics.ai_enhanced_count,
                "ai_enhanced_percentage": self.description_metrics.ai_enhanced_percentage,
                "excerpt_used_count": self.description_metrics.excerpt_used_count,
                "excerpt_used_percentage": self.description_metrics.excerpt_used_percentage,
                "title_fallback_count": self.description_metrics.title_fallback_count,
                "title_fallback_percentage": self.description_metrics.title_fallback_percentage,
                "average_confidence": self.description_metrics.average_confidence,
            },
            "tags": {
                "unique_tag_count": self.tag_metrics.unique_tag_count,
                "bookmarks_with_tags": self.tag_metrics.bookmarks_with_tags,
                "tagged_percentage": self.tag_metrics.tagged_percentage,
                "avg_tags_per_bookmark": self.tag_metrics.avg_tags_per_bookmark,
                "tag_coverage_score": self.tag_metrics.tag_coverage_score,
            },
            "folders": {
                "total_folders": self.folder_metrics.total_folders,
                "max_depth": self.folder_metrics.max_depth,
                "avg_depth": self.folder_metrics.avg_depth,
                "bookmarks_reorganized": self.folder_metrics.bookmarks_reorganized,
                "reorganized_percentage": self.folder_metrics.reorganized_percentage,
                "organization_coherence": self.folder_metrics.organization_coherence,
            },
            "attention": {
                "low_confidence_descriptions": len(self.attention_items.low_confidence_descriptions),
                "untagged_bookmarks": len(self.attention_items.untagged_bookmarks),
                "invalid_urls": len(self.attention_items.invalid_urls),
                "missing_titles": len(self.attention_items.missing_titles),
                "processing_errors": len(self.attention_items.processing_errors),
                "total_review_items": self.attention_items.total_review_items,
            },
            "overall": {
                "total_processed": self.total_processed,
                "successful_count": self.successful_count,
                "failed_count": self.failed_count,
                "success_rate": self.success_rate,
                "overall_quality_score": self.overall_quality_score,
                "processing_time_seconds": self.processing_time_seconds,
            },
        }


class QualityReporter:
    """
    Generate quality assessment reports for processed bookmarks.

    Uses the ReportGenerator infrastructure from Phase 0 to produce
    formatted reports in terminal, markdown, and JSON formats.
    """

    # Confidence threshold for flagging low-confidence descriptions
    LOW_CONFIDENCE_THRESHOLD = 0.5

    def __init__(
        self,
        bookmarks: Optional[List[Bookmark]] = None,
        processing_results: Optional[ProcessingResults] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
        original_bookmarks: Optional[List[Bookmark]] = None,
    ):
        """
        Initialize the quality reporter.

        Args:
            bookmarks: List of processed bookmarks
            processing_results: ProcessingResults from pipeline
            confidence_scores: Optional mapping of URL to AI confidence score
            original_bookmarks: Original bookmarks before processing (for comparison)
        """
        self.bookmarks = bookmarks or []
        self.processing_results = processing_results
        self.confidence_scores = confidence_scores or {}
        self.original_bookmarks = original_bookmarks or []
        self._metrics: Optional[QualityMetrics] = None

        # Build lookup for original bookmarks
        self._original_lookup: Dict[str, Bookmark] = {
            b.url: b for b in self.original_bookmarks
        }

    @property
    def metrics(self) -> QualityMetrics:
        """Get calculated quality metrics (cached)."""
        if self._metrics is None:
            self._metrics = self._calculate_metrics()
        return self._metrics

    def _calculate_metrics(self) -> QualityMetrics:
        """Calculate all quality metrics from bookmarks."""
        metrics = QualityMetrics()

        if not self.bookmarks:
            return metrics

        metrics.total_processed = len(self.bookmarks)

        # Calculate description metrics
        desc_metrics = self._calculate_description_metrics()
        metrics.description_metrics = desc_metrics

        # Calculate tag metrics
        tag_metrics = self._calculate_tag_metrics()
        metrics.tag_metrics = tag_metrics

        # Calculate folder metrics
        folder_metrics = self._calculate_folder_metrics()
        metrics.folder_metrics = folder_metrics

        # Identify attention items
        attention = self._identify_attention_items()
        metrics.attention_items = attention

        # Copy processing results statistics if available
        if self.processing_results:
            metrics.urls_validated = (
                self.processing_results.url_validation_success +
                self.processing_results.url_validation_failed
            )
            metrics.urls_valid = self.processing_results.url_validation_success
            metrics.urls_invalid = self.processing_results.url_validation_failed
            metrics.successful_count = self.processing_results.valid_bookmarks
            metrics.failed_count = self.processing_results.invalid_bookmarks
            metrics.processing_time_seconds = self.processing_results.processing_time
        else:
            # Calculate from bookmarks
            metrics.successful_count = sum(
                1 for b in self.bookmarks
                if b.processing_status.url_validated and not b.processing_status.url_validation_error
            )
            metrics.failed_count = metrics.total_processed - metrics.successful_count

        return metrics

    def _calculate_description_metrics(self) -> DescriptionMetrics:
        """Calculate description enhancement metrics."""
        desc = DescriptionMetrics()
        desc.total_count = len(self.bookmarks)

        for bookmark in self.bookmarks:
            # Determine description source
            if bookmark.enhanced_description:
                # Check if it was AI enhanced
                confidence = self.confidence_scores.get(bookmark.url, 0.8)
                desc.confidence_scores.append(confidence)

                # Compare to original to determine source
                original = self._original_lookup.get(bookmark.url)

                if original:
                    if (bookmark.enhanced_description != original.note and
                        bookmark.enhanced_description != original.excerpt):
                        # Description was changed - likely AI enhanced
                        desc.ai_enhanced_count += 1
                    elif bookmark.enhanced_description == original.excerpt:
                        desc.excerpt_used_count += 1
                    elif bookmark.enhanced_description == original.note:
                        # Used existing note
                        desc.excerpt_used_count += 1
                    else:
                        desc.ai_enhanced_count += 1
                else:
                    # No original to compare, assume AI enhanced
                    desc.ai_enhanced_count += 1
            elif bookmark.excerpt:
                desc.excerpt_used_count += 1
                desc.confidence_scores.append(0.7)  # Default confidence for excerpts
            elif bookmark.note:
                desc.excerpt_used_count += 1
                desc.confidence_scores.append(0.6)
            elif bookmark.title:
                desc.title_fallback_count += 1
                desc.confidence_scores.append(0.3)
            else:
                desc.no_description_count += 1
                desc.confidence_scores.append(0.0)

        return desc

    def _calculate_tag_metrics(self) -> TagMetrics:
        """Calculate tag analysis metrics."""
        tags = TagMetrics()
        tags.total_bookmarks = len(self.bookmarks)

        for bookmark in self.bookmarks:
            # Use optimized tags if available, otherwise original
            bookmark_tags = bookmark.optimized_tags if bookmark.optimized_tags else bookmark.tags

            if bookmark_tags:
                tags.bookmarks_with_tags += 1
                tags.tag_counts.append(len(bookmark_tags))

                for tag in bookmark_tags:
                    tags.unique_tags.add(tag.lower())
                    tags.tag_frequency[tag.lower()] = tags.tag_frequency.get(tag.lower(), 0) + 1
            else:
                tags.bookmarks_without_tags += 1
                tags.tag_counts.append(0)

        return tags

    def _calculate_folder_metrics(self) -> FolderMetrics:
        """Calculate folder organization metrics."""
        folders = FolderMetrics()
        folders.total_bookmarks = len(self.bookmarks)

        for bookmark in self.bookmarks:
            folder = bookmark.folder or ""

            if folder:
                folders.unique_folders.add(folder)
                folders.folder_distribution[folder] = folders.folder_distribution.get(folder, 0) + 1

                # Calculate depth
                depth = len(folder.split("/"))
                folders.folder_depths.append(depth)

                # Check if reorganized
                original = self._original_lookup.get(bookmark.url)
                if original and original.folder != folder:
                    folders.bookmarks_reorganized += 1
            else:
                folders.folder_depths.append(0)

        return folders

    def _identify_attention_items(self) -> AttentionItems:
        """Identify items needing manual attention."""
        attention = AttentionItems()

        for bookmark in self.bookmarks:
            # Low confidence descriptions
            confidence = self.confidence_scores.get(bookmark.url, 0.8)
            if confidence < self.LOW_CONFIDENCE_THRESHOLD:
                attention.low_confidence_descriptions.append(bookmark)

            # Untagged bookmarks
            has_tags = bool(bookmark.optimized_tags or bookmark.tags)
            if not has_tags:
                attention.untagged_bookmarks.append(bookmark)

            # Invalid URLs
            if bookmark.processing_status.url_validation_error:
                attention.invalid_urls.append(bookmark)

            # Missing titles (only count if explicit title is missing and we're using URL fallback)
            effective_title = bookmark.get_effective_title()
            has_explicit_title = bool(bookmark.title and bookmark.title.strip())
            if not has_explicit_title and (effective_title == "Untitled Bookmark" or not effective_title):
                attention.missing_titles.append(bookmark)

            # Processing errors
            errors = []
            if bookmark.processing_status.content_extraction_error:
                errors.append(bookmark.processing_status.content_extraction_error)
            if bookmark.processing_status.ai_processing_error:
                errors.append(bookmark.processing_status.ai_processing_error)

            if errors:
                attention.processing_errors.append((bookmark, "; ".join(errors)))

        return attention

    def generate_report(self, style: Union[str, ReportStyle] = "rich") -> str:
        """
        Generate a quality assessment report.

        Args:
            style: Output style - "rich", "markdown", "json", or "plain"

        Returns:
            Formatted report string
        """
        if isinstance(style, str):
            style_map = {
                "rich": ReportStyle.RICH,
                "terminal": ReportStyle.RICH,
                "markdown": ReportStyle.MARKDOWN,
                "md": ReportStyle.MARKDOWN,
                "json": ReportStyle.JSON,
                "plain": ReportStyle.PLAIN,
            }
            report_style = style_map.get(style.lower(), ReportStyle.RICH)
        else:
            report_style = style

        generator = ReportGenerator(style=report_style)
        generator.set_title(
            "QUALITY ASSESSMENT REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Description Enhancement section
        self._add_description_section(generator)

        # Tag Analysis section
        self._add_tag_section(generator)

        # Folder Organization section
        self._add_folder_section(generator)

        # Items Needing Attention section
        self._add_attention_section(generator)

        # Overall Summary section
        self._add_summary_section(generator)

        return generator.render()

    def _add_description_section(self, generator: ReportGenerator) -> None:
        """Add description enhancement section to report."""
        m = self.metrics.description_metrics

        generator.add_metrics(
            "DESCRIPTION ENHANCEMENT",
            {
                "Enhanced by AI": f"{m.ai_enhanced_count:,} ({m.ai_enhanced_percentage:.1f}%)",
                "Used existing excerpt": f"{m.excerpt_used_count:,} ({m.excerpt_used_percentage:.1f}%)",
                "Fallback to title": f"{m.title_fallback_count:,} ({m.title_fallback_percentage:.1f}%)",
                "Average confidence": f"{m.average_confidence:.2f}",
            },
            icon="chart",
        )

    def _add_tag_section(self, generator: ReportGenerator) -> None:
        """Add tag analysis section to report."""
        m = self.metrics.tag_metrics

        generator.add_metrics(
            "TAG ANALYSIS",
            {
                "Total unique tags": f"{m.unique_tag_count:,}",
                "Bookmarks with tags": f"{m.bookmarks_with_tags:,} ({m.tagged_percentage:.1f}%)",
                "Avg tags per bookmark": f"{m.avg_tags_per_bookmark:.1f}",
                "Tag coverage score": f"{m.tag_coverage_score:.2f}",
            },
            icon="tags",
        )

    def _add_folder_section(self, generator: ReportGenerator) -> None:
        """Add folder organization section to report."""
        m = self.metrics.folder_metrics

        generator.add_metrics(
            "FOLDER ORGANIZATION",
            {
                "Total folders": f"{m.total_folders:,}",
                "Max depth": f"{m.max_depth}",
                "Bookmarks reorganized": f"{m.bookmarks_reorganized:,} ({m.reorganized_percentage:.1f}%)",
                "Organization coherence": f"{m.organization_coherence:.2f}",
            },
            icon="folder",
        )

    def _add_attention_section(self, generator: ReportGenerator) -> None:
        """Add attention items section to report."""
        a = self.metrics.attention_items

        generator.add_metrics(
            "ITEMS NEEDING ATTENTION",
            {
                "Low-confidence descriptions": f"{len(a.low_confidence_descriptions):,}",
                "Untagged bookmarks": f"{len(a.untagged_bookmarks):,}",
                "Invalid URLs": f"{len(a.invalid_urls):,}",
                "Missing titles": f"{len(a.missing_titles):,}",
                "Processing errors": f"{len(a.processing_errors):,}",
                "Suggested for manual review": f"{a.total_review_items:,}",
            },
            icon="warning",
        )

    def _add_summary_section(self, generator: ReportGenerator) -> None:
        """Add overall summary section to report."""
        m = self.metrics

        generator.add_metrics(
            "OVERALL SUMMARY",
            {
                "Total processed": f"{m.total_processed:,}",
                "Successful": f"{m.successful_count:,}",
                "Failed": f"{m.failed_count:,}",
                "Success rate": f"{m.success_rate:.1f}%",
                "Overall quality score": f"{m.overall_quality_score:.2f}",
                "Processing time": f"{m.processing_time_seconds:.1f}s",
            },
            icon="metrics",
        )

    def get_items_for_review(self) -> List[Bookmark]:
        """
        Get all bookmarks that need manual attention.

        Returns:
            List of bookmarks needing review
        """
        return self.metrics.attention_items.get_all_items_for_review()

    def export_review_csv(
        self,
        path: Union[str, Path],
        include_reasons: bool = True,
    ) -> int:
        """
        Export items needing review to a separate CSV file.

        Args:
            path: Path to save the CSV file
            include_reasons: Whether to include reason column

        Returns:
            Number of items exported
        """
        items = self.get_items_for_review()

        if not items:
            return 0

        path = Path(path)

        # Build rows with reasons
        rows = []
        attention = self.metrics.attention_items

        # Create lookup for reasons
        reasons_lookup: Dict[str, List[str]] = {}

        for b in attention.low_confidence_descriptions:
            reasons_lookup.setdefault(b.url, []).append("Low confidence description")

        for b in attention.untagged_bookmarks:
            reasons_lookup.setdefault(b.url, []).append("No tags")

        for b in attention.invalid_urls:
            reasons_lookup.setdefault(b.url, []).append("Invalid URL")

        for b in attention.missing_titles:
            reasons_lookup.setdefault(b.url, []).append("Missing title")

        for b, error in attention.processing_errors:
            reasons_lookup.setdefault(b.url, []).append(f"Error: {error}")

        # Build export rows
        fieldnames = ["url", "title", "folder", "tags", "description"]
        if include_reasons:
            fieldnames.append("review_reasons")

        for bookmark in items:
            row = {
                "url": bookmark.url,
                "title": bookmark.get_effective_title(),
                "folder": bookmark.folder or "",
                "tags": ", ".join(bookmark.optimized_tags or bookmark.tags or []),
                "description": bookmark.get_effective_description()[:200],
            }

            if include_reasons:
                row["review_reasons"] = "; ".join(reasons_lookup.get(bookmark.url, ["Unknown"]))

            rows.append(row)

        # Write CSV
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return len(rows)

    def get_metrics_json(self) -> str:
        """
        Get metrics as JSON string.

        Returns:
            JSON string of metrics
        """
        return json.dumps(self.metrics.to_dict(), indent=2)

    def save_report(
        self,
        path: Union[str, Path],
        style: Optional[str] = None,
    ) -> None:
        """
        Save the report to a file.

        Args:
            path: File path to save to
            style: Output format (auto-detected from extension if not provided)
        """
        path = Path(path)

        # Auto-detect format from extension
        if style is None:
            ext_map = {
                ".md": "markdown",
                ".markdown": "markdown",
                ".json": "json",
                ".txt": "plain",
            }
            style = ext_map.get(path.suffix.lower(), "plain")

        report = self.generate_report(style=style)
        path.write_text(report, encoding="utf-8")

    def print_report(self) -> None:
        """Print the report to console using Rich formatting."""
        generator = ReportGenerator(style=ReportStyle.RICH)
        generator.set_title(
            "QUALITY ASSESSMENT REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self._add_description_section(generator)
        self._add_tag_section(generator)
        self._add_folder_section(generator)
        self._add_attention_section(generator)
        self._add_summary_section(generator)

        generator.print_to_console()


def create_quality_report(
    bookmarks: List[Bookmark],
    processing_results: Optional[ProcessingResults] = None,
    confidence_scores: Optional[Dict[str, float]] = None,
    original_bookmarks: Optional[List[Bookmark]] = None,
    style: str = "rich",
) -> str:
    """
    Convenience function to create a quality report.

    Args:
        bookmarks: List of processed bookmarks
        processing_results: Optional processing results
        confidence_scores: Optional confidence score mapping
        original_bookmarks: Optional original bookmarks for comparison
        style: Output style

    Returns:
        Formatted report string
    """
    reporter = QualityReporter(
        bookmarks=bookmarks,
        processing_results=processing_results,
        confidence_scores=confidence_scores,
        original_bookmarks=original_bookmarks,
    )
    return reporter.generate_report(style=style)
