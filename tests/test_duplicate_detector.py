"""
Unit tests for the duplicate detector module.

Tests the duplicate URL detection and resolution functionality
for bookmark collections.
"""

from datetime import datetime, timedelta

import pytest

from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.duplicate_detector import (
    DuplicateDetectionResult,
    DuplicateDetector,
    DuplicateGroup,
)


@pytest.fixture
def sample_bookmarks():
    """Create sample bookmarks with some duplicates"""
    base_time = datetime.now()

    bookmarks = [
        # Exact duplicates
        Bookmark(
            url="https://example.com/page1",
            title="Example Page 1",
            note="First note",
            created=base_time,
        ),
        Bookmark(
            url="https://example.com/page1",
            title="Example Page 1 Updated",
            note="Updated note with more content",
            tags="test, example",
            created=base_time + timedelta(days=1),
        ),
        # HTTP vs HTTPS duplicates
        Bookmark(
            url="http://test.com/article",
            title="Test Article",
            note="HTTP version",
            created=base_time,
        ),
        Bookmark(
            url="https://test.com/article",
            title="Test Article",
            note="HTTPS version",
            created=base_time + timedelta(hours=1),
        ),
        # WWW vs non-WWW duplicates
        Bookmark(
            url="https://www.example.org/blog",
            title="Blog Post",
            note="WWW version",
            created=base_time,
        ),
        Bookmark(
            url="https://example.org/blog",
            title="Blog Post",
            note="Non-WWW version",
            created=base_time + timedelta(hours=2),
        ),
        # Trailing slash duplicates
        Bookmark(
            url="https://news.com/article",
            title="News Article",
            note="No trailing slash",
            created=base_time,
        ),
        Bookmark(
            url="https://news.com/article/",
            title="News Article",
            note="With trailing slash",
            created=base_time + timedelta(minutes=30),
        ),
        # Query parameter duplicates
        Bookmark(
            url="https://shop.com/product?id=123&ref=email",
            title="Product Page",
            note="With tracking",
            created=base_time,
        ),
        Bookmark(
            url="https://shop.com/product?ref=email&id=123",
            title="Product Page",
            note="Different param order",
            created=base_time + timedelta(minutes=10),
        ),
        # Unique bookmark (no duplicates)
        Bookmark(
            url="https://unique.com/special",
            title="Unique Page",
            note="This is unique",
            created=base_time,
        ),
    ]

    return bookmarks


@pytest.fixture
def duplicate_detector():
    """Create a duplicate detector with default settings"""
    return DuplicateDetector()


class TestDuplicateDetector:
    """Test the DuplicateDetector class"""

    def test_normalize_url_basic(self, duplicate_detector):
        """Test basic URL normalization"""
        # Test protocol normalization
        assert (
            duplicate_detector.normalize_url("http://example.com")
            == "https://example.com"
        )
        assert (
            duplicate_detector.normalize_url("https://example.com")
            == "https://example.com"
        )

        # Test WWW normalization
        assert (
            duplicate_detector.normalize_url("https://www.example.com")
            == "https://example.com"
        )

        # Test trailing slash normalization
        assert (
            duplicate_detector.normalize_url("https://example.com/page/")
            == "https://example.com/page"
        )
        assert (
            duplicate_detector.normalize_url("https://example.com/")
            == "https://example.com/"
        )  # Root path preserved

        # Test case sensitivity
        assert (
            duplicate_detector.normalize_url("https://Example.COM/Page")
            == "https://example.com/page"
        )

    def test_normalize_url_query_params(self, duplicate_detector):
        """Test query parameter normalization"""
        url1 = "https://example.com/search?q=test&sort=date&limit=10"
        url2 = "https://example.com/search?limit=10&q=test&sort=date"

        normalized1 = duplicate_detector.normalize_url(url1)
        normalized2 = duplicate_detector.normalize_url(url2)

        assert normalized1 == normalized2

    def test_normalize_url_edge_cases(self, duplicate_detector):
        """Test edge cases in URL normalization"""
        # Empty URL
        assert duplicate_detector.normalize_url("") == ""

        # Malformed URL (should return original)
        malformed = "not-a-url"
        assert duplicate_detector.normalize_url(malformed) == malformed

        # URL with fragment (should be removed)
        assert (
            duplicate_detector.normalize_url("https://example.com/page#section")
            == "https://example.com/page"
        )

    def test_detect_duplicates(self, duplicate_detector, sample_bookmarks):
        """Test duplicate detection"""
        duplicate_groups = duplicate_detector.detect_duplicates(sample_bookmarks)

        # Should find 5 duplicate groups (each with 2 bookmarks)
        assert len(duplicate_groups) == 5

        # Each group should have exactly 2 bookmarks
        for group in duplicate_groups.values():
            assert len(group.bookmarks) == 2

    def test_resolve_duplicates_newest(self, duplicate_detector, sample_bookmarks):
        """Test duplicate resolution with 'newest' strategy"""
        duplicate_groups = duplicate_detector.detect_duplicates(sample_bookmarks)
        resolved_groups = duplicate_detector.resolve_duplicates(
            duplicate_groups, "newest"
        )

        for group in resolved_groups.values():
            kept_bookmark = group.get_kept_bookmark()
            removed_bookmarks = group.get_removed_bookmarks()

            # Should keep the newest bookmark
            assert kept_bookmark is not None
            assert len(removed_bookmarks) == 1

            # Kept bookmark should be newer than removed ones
            for removed in removed_bookmarks:
                if kept_bookmark.created and removed.created:
                    assert kept_bookmark.created >= removed.created

    def test_resolve_duplicates_oldest(self, duplicate_detector, sample_bookmarks):
        """Test duplicate resolution with 'oldest' strategy"""
        duplicate_groups = duplicate_detector.detect_duplicates(sample_bookmarks)
        resolved_groups = duplicate_detector.resolve_duplicates(
            duplicate_groups, "oldest"
        )

        for group in resolved_groups.values():
            kept_bookmark = group.get_kept_bookmark()
            removed_bookmarks = group.get_removed_bookmarks()

            # Should keep the oldest bookmark
            assert kept_bookmark is not None
            assert len(removed_bookmarks) == 1

            # Kept bookmark should be older than removed ones
            for removed in removed_bookmarks:
                if kept_bookmark.created and removed.created:
                    assert kept_bookmark.created <= removed.created

    def test_resolve_duplicates_most_complete(
        self, duplicate_detector, sample_bookmarks
    ):
        """Test duplicate resolution with 'most_complete' strategy"""
        duplicate_groups = duplicate_detector.detect_duplicates(sample_bookmarks)
        resolved_groups = duplicate_detector.resolve_duplicates(
            duplicate_groups, "most_complete"
        )

        for group in resolved_groups.values():
            kept_bookmark = group.get_kept_bookmark()
            removed_bookmarks = group.get_removed_bookmarks()

            # Should keep the most complete bookmark
            assert kept_bookmark is not None
            assert len(removed_bookmarks) == 1

            # Calculate completeness scores
            kept_score = self._calculate_completeness_score(kept_bookmark)
            for removed in removed_bookmarks:
                removed_score = self._calculate_completeness_score(removed)
                assert kept_score >= removed_score

    def test_resolve_duplicates_highest_quality(
        self, duplicate_detector, sample_bookmarks
    ):
        """Test duplicate resolution with 'highest_quality' strategy"""
        duplicate_groups = duplicate_detector.detect_duplicates(sample_bookmarks)
        resolved_groups = duplicate_detector.resolve_duplicates(
            duplicate_groups, "highest_quality"
        )

        for group in resolved_groups.values():
            kept_bookmark = group.get_kept_bookmark()
            removed_bookmarks = group.get_removed_bookmarks()

            # Should keep the highest quality bookmark
            assert kept_bookmark is not None
            assert len(removed_bookmarks) == 1

            # Quality strategy should prefer bookmarks with longer notes and more tags
            for removed in removed_bookmarks:
                if kept_bookmark.note and removed.note:
                    # If notes differ significantly, longer should be kept
                    if abs(len(kept_bookmark.note) - len(removed.note)) > 10:
                        assert len(kept_bookmark.note) >= len(removed.note)

    def test_process_bookmarks(self, duplicate_detector, sample_bookmarks):
        """Test complete duplicate processing workflow"""
        original_count = len(sample_bookmarks)

        deduplicated, result = duplicate_detector.process_bookmarks(
            sample_bookmarks, strategy="highest_quality", dry_run=False
        )

        # Should have fewer bookmarks after deduplication
        assert len(deduplicated) < original_count

        # Should have 6 unique bookmarks (5 duplicate groups + 1 unique)
        assert len(deduplicated) == 6

        # Result should have correct statistics
        assert result.total_bookmarks == original_count
        assert result.unique_urls == 6
        assert result.duplicates_count == 5  # 5 duplicates removed
        assert result.removed_count == 5
        assert len(result.duplicate_groups) == 5

    def test_process_bookmarks_dry_run(self, duplicate_detector, sample_bookmarks):
        """Test dry run mode"""
        original_count = len(sample_bookmarks)

        deduplicated, result = duplicate_detector.process_bookmarks(
            sample_bookmarks, strategy="newest", dry_run=True
        )

        # Should not actually remove duplicates in dry run
        assert len(deduplicated) == original_count
        assert result.removed_count == 0
        assert result.duplicates_count == 5  # Still detects duplicates

    def test_generate_report(self, duplicate_detector, sample_bookmarks):
        """Test duplicate detection report generation"""
        _, result = duplicate_detector.process_bookmarks(
            sample_bookmarks, strategy="highest_quality", dry_run=False
        )

        report = duplicate_detector.generate_report(result)

        # Report should contain key information
        assert "DUPLICATE URL DETECTION REPORT" in report
        assert str(result.total_bookmarks) in report
        assert str(result.unique_urls) in report
        assert str(result.duplicates_count) in report
        assert "DUPLICATE GROUPS" in report

    def test_custom_normalization_settings(self):
        """Test duplicate detector with custom normalization settings"""
        # Test with case-sensitive URLs
        detector = DuplicateDetector(
            normalize_www=False, normalize_protocol=False, case_sensitive=True
        )

        bookmarks = [
            Bookmark(url="http://Example.com/Page", title="Page 1"),
            Bookmark(url="https://example.com/page", title="Page 2"),
            Bookmark(url="https://www.example.com/page", title="Page 3"),
        ]

        duplicate_groups = detector.detect_duplicates(bookmarks)

        # With these settings, all URLs should be considered unique
        assert len(duplicate_groups) == 0

    def _calculate_completeness_score(self, bookmark):
        """Helper method to calculate completeness score like the detector does"""
        score = 0
        if bookmark.title and len(bookmark.title.strip()) > 0:
            score += 2
        if bookmark.note and len(bookmark.note.strip()) > 0:
            score += 3
        if bookmark.excerpt and len(bookmark.excerpt.strip()) > 0:
            score += 1
        if bookmark.tags and len(bookmark.tags) > 0:
            score += 2
        if bookmark.folder and bookmark.folder != "/":
            score += 1
        if bookmark.created:
            score += 1
        return score


class TestDuplicateGroup:
    """Test the DuplicateGroup class"""

    def test_duplicate_group_basic(self):
        """Test basic DuplicateGroup functionality"""
        group = DuplicateGroup(normalized_url="https://example.com")

        bookmark1 = Bookmark(url="https://example.com", title="Title 1")
        bookmark2 = Bookmark(url="https://example.com", title="Title 2")

        group.add_bookmark(bookmark1)
        group.add_bookmark(bookmark2)

        assert len(group.bookmarks) == 2
        assert group.get_kept_bookmark() is None  # No keep_index set yet
        assert group.get_removed_bookmarks() == []

    def test_duplicate_group_resolution(self):
        """Test DuplicateGroup after resolution"""
        group = DuplicateGroup(normalized_url="https://example.com")

        bookmark1 = Bookmark(url="https://example.com", title="Title 1")
        bookmark2 = Bookmark(url="https://example.com", title="Title 2")

        group.add_bookmark(bookmark1)
        group.add_bookmark(bookmark2)
        group.keep_index = 1
        group.resolution_reason = "Test resolution"

        assert group.get_kept_bookmark() == bookmark2
        assert group.get_removed_bookmarks() == [bookmark1]


class TestDuplicateDetectionResult:
    """Test the DuplicateDetectionResult class"""

    def test_duplicate_detection_result(self):
        """Test DuplicateDetectionResult functionality"""
        result = DuplicateDetectionResult(
            total_bookmarks=10,
            unique_urls=8,
            duplicate_groups=[],
            duplicates_count=2,
            removed_count=2,
            processing_time=1.5,
        )

        # Test to_dict
        result_dict = result.to_dict()
        assert result_dict["total_bookmarks"] == 10
        assert result_dict["unique_urls"] == 8
        assert result_dict["duplicates_count"] == 2
        assert result_dict["removed_count"] == 2
        assert result_dict["processing_time"] == 1.5

        # Test get_summary
        summary = result.get_summary()
        assert "Duplicate Detection Summary" in summary
        assert "Total bookmarks: 10" in summary
        assert "Unique URLs: 8" in summary
        assert "Total duplicates: 2" in summary
        assert "Removed: 2" in summary
