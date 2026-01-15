"""
Unit tests for bookmark filter infrastructure.

Tests the BookmarkFilter classes and FilterChain for filtering
bookmarks by various criteria.
"""

from datetime import datetime, timedelta

import pytest

from bookmark_processor.core.data_models import Bookmark, ProcessingStatus
from bookmark_processor.core.filters import (
    BookmarkFilter,
    CompositeFilter,
    CustomFilter,
    DateRangeFilter,
    DomainFilter,
    FilterChain,
    FolderFilter,
    NotFilter,
    StatusFilter,
    TagFilter,
    URLPatternFilter,
    date_filter,
    domain_filter,
    folder_filter,
    status_filter,
    tag_filter,
    url_pattern_filter,
)


class TestFolderFilter:
    """Test FolderFilter class."""

    def test_exact_match(self):
        """Test exact folder matching."""
        filter_obj = FolderFilter("Tech")

        assert filter_obj.matches(Bookmark(url="http://test.com", folder="Tech"))
        assert not filter_obj.matches(Bookmark(url="http://test.com", folder="Personal"))
        assert not filter_obj.matches(Bookmark(url="http://test.com", folder="Tech/Python"))

    def test_glob_pattern_star(self):
        """Test glob pattern with star wildcard."""
        filter_obj = FolderFilter("Tech/*")

        assert filter_obj.matches(Bookmark(url="http://test.com", folder="Tech/Python"))
        assert filter_obj.matches(Bookmark(url="http://test.com", folder="Tech/JavaScript"))
        assert not filter_obj.matches(Bookmark(url="http://test.com", folder="Tech"))
        assert not filter_obj.matches(Bookmark(url="http://test.com", folder="Personal/Finance"))

    def test_glob_pattern_double_star(self):
        """Test glob pattern with double star (any depth)."""
        filter_obj = FolderFilter("Tech/*")

        assert filter_obj.matches(Bookmark(url="http://test.com", folder="Tech/Python/Django"))

        # Also test with pattern at any depth
        filter_obj2 = FolderFilter("*/Python/*")
        assert filter_obj2.matches(Bookmark(url="http://test.com", folder="Tech/Python/Django"))

    def test_case_insensitive_default(self):
        """Test that matching is case-insensitive by default."""
        filter_obj = FolderFilter("Tech")

        assert filter_obj.matches(Bookmark(url="http://test.com", folder="tech"))
        assert filter_obj.matches(Bookmark(url="http://test.com", folder="TECH"))
        assert filter_obj.matches(Bookmark(url="http://test.com", folder="Tech"))

    def test_case_sensitive(self):
        """Test case-sensitive matching."""
        filter_obj = FolderFilter("Tech", case_sensitive=True)

        assert filter_obj.matches(Bookmark(url="http://test.com", folder="Tech"))
        assert not filter_obj.matches(Bookmark(url="http://test.com", folder="tech"))
        assert not filter_obj.matches(Bookmark(url="http://test.com", folder="TECH"))

    def test_empty_folder(self):
        """Test matching bookmarks with empty folders."""
        filter_obj = FolderFilter("")

        assert filter_obj.matches(Bookmark(url="http://test.com", folder=""))
        assert not filter_obj.matches(Bookmark(url="http://test.com", folder="Tech"))

    def test_question_mark_wildcard(self):
        """Test single character wildcard."""
        filter_obj = FolderFilter("Tech?")

        assert filter_obj.matches(Bookmark(url="http://test.com", folder="Tech1"))
        assert filter_obj.matches(Bookmark(url="http://test.com", folder="Techs"))
        assert not filter_obj.matches(Bookmark(url="http://test.com", folder="Tech"))
        assert not filter_obj.matches(Bookmark(url="http://test.com", folder="Tech12"))


class TestTagFilter:
    """Test TagFilter class."""

    def test_single_tag_any_mode(self):
        """Test filtering by single tag in 'any' mode."""
        filter_obj = TagFilter("python")

        assert filter_obj.matches(Bookmark(url="http://test.com", tags=["python", "programming"]))
        assert filter_obj.matches(Bookmark(url="http://test.com", tags=["python"]))
        assert not filter_obj.matches(Bookmark(url="http://test.com", tags=["javascript"]))
        assert not filter_obj.matches(Bookmark(url="http://test.com", tags=[]))

    def test_multiple_tags_any_mode(self):
        """Test filtering by multiple tags in 'any' mode."""
        filter_obj = TagFilter(["python", "javascript"], mode="any")

        assert filter_obj.matches(Bookmark(url="http://test.com", tags=["python"]))
        assert filter_obj.matches(Bookmark(url="http://test.com", tags=["javascript"]))
        assert filter_obj.matches(Bookmark(url="http://test.com", tags=["python", "javascript"]))
        assert not filter_obj.matches(Bookmark(url="http://test.com", tags=["rust"]))

    def test_multiple_tags_all_mode(self):
        """Test filtering by multiple tags in 'all' mode."""
        filter_obj = TagFilter(["python", "django"], mode="all")

        assert filter_obj.matches(Bookmark(url="http://test.com", tags=["python", "django"]))
        assert filter_obj.matches(Bookmark(url="http://test.com", tags=["python", "django", "web"]))
        assert not filter_obj.matches(Bookmark(url="http://test.com", tags=["python"]))
        assert not filter_obj.matches(Bookmark(url="http://test.com", tags=["django"]))

    def test_case_insensitive_default(self):
        """Test that tag matching is case-insensitive by default."""
        filter_obj = TagFilter("Python")

        assert filter_obj.matches(Bookmark(url="http://test.com", tags=["python"]))
        assert filter_obj.matches(Bookmark(url="http://test.com", tags=["PYTHON"]))
        assert filter_obj.matches(Bookmark(url="http://test.com", tags=["Python"]))

    def test_case_sensitive(self):
        """Test case-sensitive tag matching."""
        filter_obj = TagFilter("Python", case_sensitive=True)

        assert filter_obj.matches(Bookmark(url="http://test.com", tags=["Python"]))
        assert not filter_obj.matches(Bookmark(url="http://test.com", tags=["python"]))
        assert not filter_obj.matches(Bookmark(url="http://test.com", tags=["PYTHON"]))

    def test_string_tag_input(self):
        """Test that single string tag is handled correctly."""
        filter_obj = TagFilter("python")

        assert filter_obj.matches(Bookmark(url="http://test.com", tags=["python", "web"]))

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            TagFilter("python", mode="invalid")


class TestDateRangeFilter:
    """Test DateRangeFilter class."""

    def test_start_date_only(self):
        """Test filtering with only start date."""
        start = datetime(2024, 1, 1)
        filter_obj = DateRangeFilter(start=start)

        assert filter_obj.matches(Bookmark(url="http://test.com", created=datetime(2024, 6, 15)))
        assert filter_obj.matches(Bookmark(url="http://test.com", created=datetime(2024, 1, 1)))
        assert not filter_obj.matches(Bookmark(url="http://test.com", created=datetime(2023, 12, 31)))

    def test_end_date_only(self):
        """Test filtering with only end date."""
        end = datetime(2024, 12, 31)
        filter_obj = DateRangeFilter(end=end)

        assert filter_obj.matches(Bookmark(url="http://test.com", created=datetime(2024, 6, 15)))
        assert filter_obj.matches(Bookmark(url="http://test.com", created=datetime(2024, 12, 31)))
        assert not filter_obj.matches(Bookmark(url="http://test.com", created=datetime(2025, 1, 1)))

    def test_date_range(self):
        """Test filtering with both start and end dates."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        filter_obj = DateRangeFilter(start=start, end=end)

        assert filter_obj.matches(Bookmark(url="http://test.com", created=datetime(2024, 6, 15)))
        assert filter_obj.matches(Bookmark(url="http://test.com", created=datetime(2024, 1, 1)))
        assert filter_obj.matches(Bookmark(url="http://test.com", created=datetime(2024, 12, 31)))
        assert not filter_obj.matches(Bookmark(url="http://test.com", created=datetime(2023, 12, 31)))
        assert not filter_obj.matches(Bookmark(url="http://test.com", created=datetime(2025, 1, 1)))

    def test_no_created_date(self):
        """Test that bookmarks without created date don't match."""
        filter_obj = DateRangeFilter(start=datetime(2024, 1, 1))

        assert not filter_obj.matches(Bookmark(url="http://test.com", created=None))

    def test_no_dates_raises_error(self):
        """Test that creating filter without dates raises error."""
        with pytest.raises(ValueError, match="At least one"):
            DateRangeFilter(start=None, end=None)

    def test_invalid_range_raises_error(self):
        """Test that start > end raises error."""
        with pytest.raises(ValueError, match="Start date must be before"):
            DateRangeFilter(
                start=datetime(2024, 12, 31),
                end=datetime(2024, 1, 1)
            )

    def test_from_string_full_range(self):
        """Test creating filter from string with full range."""
        filter_obj = DateRangeFilter.from_string("2024-01-01:2024-12-31")

        assert filter_obj.start == datetime(2024, 1, 1)
        assert filter_obj.end.date() == datetime(2024, 12, 31).date()

    def test_from_string_start_only(self):
        """Test creating filter from string with start only."""
        filter_obj = DateRangeFilter.from_string("2024-01-01:")

        assert filter_obj.start == datetime(2024, 1, 1)
        assert filter_obj.end is None

    def test_from_string_end_only(self):
        """Test creating filter from string with end only."""
        filter_obj = DateRangeFilter.from_string(":2024-12-31")

        assert filter_obj.start is None
        assert filter_obj.end.date() == datetime(2024, 12, 31).date()

    def test_from_string_invalid_format(self):
        """Test that invalid string format raises error."""
        with pytest.raises(ValueError, match="Invalid date range format"):
            DateRangeFilter.from_string("2024-01-01")

    def test_from_string_invalid_date(self):
        """Test that invalid date raises error."""
        with pytest.raises(ValueError, match="Invalid start date"):
            DateRangeFilter.from_string("not-a-date:2024-12-31")


class TestDomainFilter:
    """Test DomainFilter class."""

    def test_single_domain(self):
        """Test filtering by single domain."""
        filter_obj = DomainFilter("github.com")

        assert filter_obj.matches(Bookmark(url="https://github.com/user/repo"))
        assert filter_obj.matches(Bookmark(url="http://github.com/"))
        assert not filter_obj.matches(Bookmark(url="https://gitlab.com/user/repo"))

    def test_multiple_domains(self):
        """Test filtering by multiple domains."""
        filter_obj = DomainFilter(["github.com", "gitlab.com"])

        assert filter_obj.matches(Bookmark(url="https://github.com/user/repo"))
        assert filter_obj.matches(Bookmark(url="https://gitlab.com/user/repo"))
        assert not filter_obj.matches(Bookmark(url="https://bitbucket.org/user/repo"))

    def test_subdomain_included_by_default(self):
        """Test that subdomains are included by default."""
        filter_obj = DomainFilter("github.com")

        assert filter_obj.matches(Bookmark(url="https://api.github.com/users"))
        assert filter_obj.matches(Bookmark(url="https://raw.github.com/file"))
        assert filter_obj.matches(Bookmark(url="https://github.com/"))

    def test_subdomain_excluded(self):
        """Test excluding subdomains."""
        filter_obj = DomainFilter("github.com", include_subdomains=False)

        assert filter_obj.matches(Bookmark(url="https://github.com/user/repo"))
        assert not filter_obj.matches(Bookmark(url="https://api.github.com/users"))

    def test_string_with_commas(self):
        """Test domain string with comma separation."""
        filter_obj = DomainFilter("github.com, gitlab.com, bitbucket.org")

        assert filter_obj.matches(Bookmark(url="https://github.com/"))
        assert filter_obj.matches(Bookmark(url="https://gitlab.com/"))
        assert filter_obj.matches(Bookmark(url="https://bitbucket.org/"))

    def test_empty_url(self):
        """Test handling of empty URL."""
        filter_obj = DomainFilter("github.com")

        assert not filter_obj.matches(Bookmark(url=""))
        assert not filter_obj.matches(Bookmark(url=None))

    def test_url_with_port(self):
        """Test URL with port number."""
        filter_obj = DomainFilter("localhost")

        assert filter_obj.matches(Bookmark(url="http://localhost:8080/api"))
        assert filter_obj.matches(Bookmark(url="http://localhost/"))


class TestStatusFilter:
    """Test StatusFilter class."""

    def test_validated_status(self):
        """Test filtering by validated status."""
        filter_obj = StatusFilter("validated")

        validated_bookmark = Bookmark(url="http://test.com")
        validated_bookmark.processing_status.url_validated = True

        unvalidated_bookmark = Bookmark(url="http://test.com")

        assert filter_obj.matches(validated_bookmark)
        assert not filter_obj.matches(unvalidated_bookmark)

    def test_invalid_status(self):
        """Test filtering by invalid status."""
        filter_obj = StatusFilter("invalid")

        invalid_bookmark = Bookmark(url="http://test.com")
        invalid_bookmark.processing_status.url_validation_error = "Connection timeout"

        valid_bookmark = Bookmark(url="http://test.com")

        assert filter_obj.matches(invalid_bookmark)
        assert not filter_obj.matches(valid_bookmark)

    def test_processed_status(self):
        """Test filtering by AI processed status."""
        filter_obj = StatusFilter("processed")

        processed_bookmark = Bookmark(url="http://test.com")
        processed_bookmark.processing_status.ai_processed = True

        unprocessed_bookmark = Bookmark(url="http://test.com")

        assert filter_obj.matches(processed_bookmark)
        assert not filter_obj.matches(unprocessed_bookmark)

    def test_unprocessed_status(self):
        """Test filtering by unprocessed status."""
        filter_obj = StatusFilter("unprocessed")

        unprocessed_bookmark = Bookmark(url="http://test.com")

        processed_bookmark = Bookmark(url="http://test.com")
        processed_bookmark.processing_status.ai_processed = True

        assert filter_obj.matches(unprocessed_bookmark)
        assert not filter_obj.matches(processed_bookmark)

    def test_error_status(self):
        """Test filtering by any error status."""
        filter_obj = StatusFilter("error")

        # URL validation error
        bookmark1 = Bookmark(url="http://test.com")
        bookmark1.processing_status.url_validation_error = "Error"
        assert filter_obj.matches(bookmark1)

        # Content extraction error
        bookmark2 = Bookmark(url="http://test.com")
        bookmark2.processing_status.content_extraction_error = "Error"
        assert filter_obj.matches(bookmark2)

        # AI processing error
        bookmark3 = Bookmark(url="http://test.com")
        bookmark3.processing_status.ai_processing_error = "Error"
        assert filter_obj.matches(bookmark3)

        # No errors
        bookmark4 = Bookmark(url="http://test.com")
        assert not filter_obj.matches(bookmark4)

    def test_multiple_statuses(self):
        """Test filtering by multiple statuses (OR logic)."""
        filter_obj = StatusFilter(["validated", "processed"])

        validated = Bookmark(url="http://test.com")
        validated.processing_status.url_validated = True

        processed = Bookmark(url="http://test.com")
        processed.processing_status.ai_processed = True

        neither = Bookmark(url="http://test.com")

        assert filter_obj.matches(validated)
        assert filter_obj.matches(processed)
        assert not filter_obj.matches(neither)

    def test_invalid_status_raises_error(self):
        """Test that invalid status raises error."""
        with pytest.raises(ValueError, match="Invalid status"):
            StatusFilter("invalid_status_name")


class TestURLPatternFilter:
    """Test URLPatternFilter class."""

    def test_simple_pattern(self):
        """Test simple regex pattern."""
        filter_obj = URLPatternFilter(r"github\.com")

        assert filter_obj.matches(Bookmark(url="https://github.com/user/repo"))
        assert not filter_obj.matches(Bookmark(url="https://gitlab.com/user/repo"))

    def test_complex_pattern(self):
        """Test complex regex pattern."""
        filter_obj = URLPatternFilter(r"github\.com/[^/]+/[^/]+$")

        assert filter_obj.matches(Bookmark(url="https://github.com/user/repo"))
        assert not filter_obj.matches(Bookmark(url="https://github.com/user/repo/issues"))

    def test_case_insensitive_default(self):
        """Test that pattern matching is case-insensitive by default."""
        filter_obj = URLPatternFilter(r"github\.com")

        assert filter_obj.matches(Bookmark(url="https://GITHUB.COM/user/repo"))
        assert filter_obj.matches(Bookmark(url="https://GitHub.com/user/repo"))


class TestCustomFilter:
    """Test CustomFilter class."""

    def test_custom_predicate(self):
        """Test custom filter with predicate function."""
        # Filter bookmarks with title longer than 10 characters
        filter_obj = CustomFilter(
            predicate=lambda b: len(b.title) > 10,
            name="long_title"
        )

        assert filter_obj.matches(Bookmark(url="http://test.com", title="This is a long title"))
        assert not filter_obj.matches(Bookmark(url="http://test.com", title="Short"))

    def test_custom_filter_complex_logic(self):
        """Test custom filter with complex logic."""
        # Filter bookmarks from tech folders with python tag
        filter_obj = CustomFilter(
            predicate=lambda b: (
                b.folder.startswith("Tech") and "python" in [t.lower() for t in b.tags]
            ),
            name="tech_python"
        )

        assert filter_obj.matches(Bookmark(
            url="http://test.com",
            folder="Tech/Programming",
            tags=["Python", "web"]
        ))
        assert not filter_obj.matches(Bookmark(
            url="http://test.com",
            folder="Personal",
            tags=["Python"]
        ))


class TestCompositeFilter:
    """Test CompositeFilter class."""

    def test_and_operator(self):
        """Test AND combination of filters."""
        folder_f = FolderFilter("Tech/*")
        tag_f = TagFilter("python")

        composite = CompositeFilter([folder_f, tag_f], operator="and")

        # Matches both
        assert composite.matches(Bookmark(
            url="http://test.com",
            folder="Tech/Python",
            tags=["python"]
        ))

        # Matches folder only
        assert not composite.matches(Bookmark(
            url="http://test.com",
            folder="Tech/Python",
            tags=["javascript"]
        ))

        # Matches tag only
        assert not composite.matches(Bookmark(
            url="http://test.com",
            folder="Personal",
            tags=["python"]
        ))

    def test_or_operator(self):
        """Test OR combination of filters."""
        folder_f = FolderFilter("Tech/*")
        tag_f = TagFilter("python")

        composite = CompositeFilter([folder_f, tag_f], operator="or")

        # Matches both
        assert composite.matches(Bookmark(
            url="http://test.com",
            folder="Tech/Python",
            tags=["python"]
        ))

        # Matches folder only
        assert composite.matches(Bookmark(
            url="http://test.com",
            folder="Tech/JavaScript",
            tags=["javascript"]
        ))

        # Matches tag only
        assert composite.matches(Bookmark(
            url="http://test.com",
            folder="Personal",
            tags=["python"]
        ))

        # Matches neither
        assert not composite.matches(Bookmark(
            url="http://test.com",
            folder="Personal",
            tags=["javascript"]
        ))

    def test_invalid_operator_raises_error(self):
        """Test that invalid operator raises error."""
        with pytest.raises(ValueError, match="Invalid operator"):
            CompositeFilter([], operator="xor")

    def test_empty_filters(self):
        """Test composite with no filters matches everything."""
        composite = CompositeFilter([], operator="and")

        assert composite.matches(Bookmark(url="http://test.com"))


class TestNotFilter:
    """Test NotFilter class."""

    def test_negation(self):
        """Test filter negation."""
        tag_f = TagFilter("python")
        not_f = NotFilter(tag_f)

        assert not_f.matches(Bookmark(url="http://test.com", tags=["javascript"]))
        assert not not_f.matches(Bookmark(url="http://test.com", tags=["python"]))

    def test_invert_operator(self):
        """Test using ~ operator for negation."""
        tag_f = TagFilter("python")
        not_f = ~tag_f

        assert not_f.matches(Bookmark(url="http://test.com", tags=["javascript"]))
        assert not not_f.matches(Bookmark(url="http://test.com", tags=["python"]))


class TestFilterOperators:
    """Test filter operators (& and |)."""

    def test_and_operator(self):
        """Test & operator creates AND composite."""
        folder_f = FolderFilter("Tech/*")
        tag_f = TagFilter("python")

        combined = folder_f & tag_f

        assert isinstance(combined, CompositeFilter)
        assert combined.operator == "and"

        # Test functionality
        assert combined.matches(Bookmark(
            url="http://test.com",
            folder="Tech/Python",
            tags=["python"]
        ))
        assert not combined.matches(Bookmark(
            url="http://test.com",
            folder="Personal",
            tags=["python"]
        ))

    def test_or_operator(self):
        """Test | operator creates OR composite."""
        folder_f = FolderFilter("Tech/*")
        tag_f = TagFilter("python")

        combined = folder_f | tag_f

        assert isinstance(combined, CompositeFilter)
        assert combined.operator == "or"

        # Test functionality
        assert combined.matches(Bookmark(
            url="http://test.com",
            folder="Personal",
            tags=["python"]
        ))

    def test_chained_operators(self):
        """Test chaining multiple operators."""
        f1 = FolderFilter("Tech/*")
        f2 = TagFilter("python")
        f3 = DomainFilter("github.com")

        # (folder AND tag) OR domain
        combined = (f1 & f2) | f3

        assert combined.matches(Bookmark(
            url="http://test.com",
            folder="Tech/Python",
            tags=["python"]
        ))
        assert combined.matches(Bookmark(
            url="https://github.com/user/repo",
            folder="Personal",
            tags=["javascript"]
        ))


class TestFilterChain:
    """Test FilterChain class."""

    def test_empty_chain(self):
        """Test empty filter chain matches everything."""
        chain = FilterChain()

        assert chain.matches(Bookmark(url="http://test.com"))
        assert chain.apply([Bookmark(url="http://test.com")]) == [Bookmark(url="http://test.com")]

    def test_add_filters(self):
        """Test adding filters to chain."""
        chain = FilterChain()
        chain.add(FolderFilter("Tech/*"))
        chain.add(TagFilter("python"))

        assert len(chain) == 2

    def test_apply_and_logic(self):
        """Test applying chain with AND logic."""
        chain = FilterChain(operator="and")
        chain.add(FolderFilter("Tech/*"))
        chain.add(TagFilter("python"))

        bookmarks = [
            Bookmark(url="http://1.com", folder="Tech/Python", tags=["python"]),
            Bookmark(url="http://2.com", folder="Tech/Python", tags=["javascript"]),
            Bookmark(url="http://3.com", folder="Personal", tags=["python"]),
        ]

        result = chain.apply(bookmarks)

        assert len(result) == 1
        assert result[0].url == "http://1.com"

    def test_apply_or_logic(self):
        """Test applying chain with OR logic."""
        chain = FilterChain(operator="or")
        chain.add(FolderFilter("Tech/*"))
        chain.add(TagFilter("python"))

        bookmarks = [
            Bookmark(url="http://1.com", folder="Tech/Python", tags=["python"]),
            Bookmark(url="http://2.com", folder="Tech/Python", tags=["javascript"]),
            Bookmark(url="http://3.com", folder="Personal", tags=["python"]),
            Bookmark(url="http://4.com", folder="Personal", tags=["javascript"]),
        ]

        result = chain.apply(bookmarks)

        assert len(result) == 3
        urls = {b.url for b in result}
        assert "http://1.com" in urls
        assert "http://2.com" in urls
        assert "http://3.com" in urls

    def test_count_matching(self):
        """Test counting matching bookmarks."""
        chain = FilterChain()
        chain.add(TagFilter("python"))

        bookmarks = [
            Bookmark(url="http://1.com", tags=["python"]),
            Bookmark(url="http://2.com", tags=["python"]),
            Bookmark(url="http://3.com", tags=["javascript"]),
        ]

        assert chain.count_matching(bookmarks) == 2

    def test_from_cli_args(self):
        """Test creating chain from CLI arguments."""
        args = {
            "filter_folder": "Tech/*",
            "filter_tag": "python,django",
            "filter_domain": "github.com",
        }

        chain = FilterChain.from_cli_args(args)

        assert len(chain) == 3

    def test_from_cli_args_date_range(self):
        """Test creating chain from CLI args with date range."""
        args = {
            "filter_date": "2024-01-01:2024-12-31",
        }

        chain = FilterChain.from_cli_args(args)

        assert len(chain) == 1

        assert chain.matches(Bookmark(
            url="http://test.com",
            created=datetime(2024, 6, 15)
        ))
        assert not chain.matches(Bookmark(
            url="http://test.com",
            created=datetime(2023, 6, 15)
        ))

    def test_from_cli_args_retry_invalid(self):
        """Test retry_invalid shortcut."""
        args = {"retry_invalid": True}

        chain = FilterChain.from_cli_args(args)

        assert len(chain) == 1

        invalid_bookmark = Bookmark(url="http://test.com")
        invalid_bookmark.processing_status.url_validation_error = "Error"

        assert chain.matches(invalid_bookmark)

    def test_method_chaining(self):
        """Test fluent interface."""
        chain = (
            FilterChain()
            .add(FolderFilter("Tech/*"))
            .add(TagFilter("python"))
        )

        assert len(chain) == 2

    def test_bool_conversion(self):
        """Test boolean conversion."""
        empty_chain = FilterChain()
        assert not empty_chain

        non_empty_chain = FilterChain()
        non_empty_chain.add(TagFilter("python"))
        assert non_empty_chain


class TestFactoryFunctions:
    """Test convenience factory functions."""

    def test_folder_filter(self):
        """Test folder_filter factory function."""
        f = folder_filter("Tech/*")
        assert isinstance(f, FolderFilter)
        assert f.matches(Bookmark(url="http://test.com", folder="Tech/Python"))

    def test_tag_filter(self):
        """Test tag_filter factory function."""
        f = tag_filter(["python", "django"], mode="all")
        assert isinstance(f, TagFilter)
        assert f.mode == "all"

    def test_date_filter(self):
        """Test date_filter factory function."""
        f = date_filter(start=datetime(2024, 1, 1))
        assert isinstance(f, DateRangeFilter)

    def test_domain_filter(self):
        """Test domain_filter factory function."""
        f = domain_filter("github.com")
        assert isinstance(f, DomainFilter)

    def test_status_filter(self):
        """Test status_filter factory function."""
        f = status_filter("validated")
        assert isinstance(f, StatusFilter)

    def test_url_pattern_filter(self):
        """Test url_pattern_filter factory function."""
        f = url_pattern_filter(r"github\.com")
        assert isinstance(f, URLPatternFilter)


class TestFilterIntegration:
    """Integration tests for filter combinations."""

    def test_complex_filter_scenario(self):
        """Test complex real-world filtering scenario."""
        # Build a complex filter:
        # (Tech folder AND python tag) OR (github domain AND validated)
        tech_python = FolderFilter("Tech/*") & TagFilter("python")
        github_validated = DomainFilter("github.com") & StatusFilter("validated")

        complex_filter = tech_python | github_validated

        bookmarks = [
            # Matches tech_python
            Bookmark(url="http://example.com", folder="Tech/Python", tags=["python"]),
            # Matches github_validated
            Bookmark(url="https://github.com/user/repo", folder="Personal", tags=["git"]),
            # Matches neither
            Bookmark(url="http://example.com", folder="Personal", tags=["javascript"]),
        ]

        # Set validated status for github bookmark
        bookmarks[1].processing_status.url_validated = True

        result = complex_filter.filter(bookmarks)

        assert len(result) == 2
        urls = {b.url for b in result}
        assert "http://example.com" in urls
        assert "https://github.com/user/repo" in urls

    def test_filter_chain_with_all_filter_types(self):
        """Test filter chain using all filter types."""
        chain = FilterChain()
        chain.add(FolderFilter("Tech/*"))
        chain.add(TagFilter("python"))
        chain.add(DomainFilter("github.com"))
        chain.add(DateRangeFilter(start=datetime(2024, 1, 1)))

        bookmark = Bookmark(
            url="https://github.com/user/repo",
            folder="Tech/Python",
            tags=["python", "web"],
            created=datetime(2024, 6, 15),
        )

        # All filters must match
        assert chain.matches(bookmark)

        # Change one attribute to fail a filter
        bookmark.folder = "Personal"
        assert not chain.matches(bookmark)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
