"""
Tests for Streaming/Incremental Processing Module (Phase 8.1).

Tests cover:
- StreamingBookmarkReader: Generator-based reading
- StreamingBookmarkWriter: Incremental writing
- StreamingPipeline: Streaming pipeline execution
"""

import csv
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.streaming import (
    StreamingBookmarkReader,
    StreamingBookmarkWriter,
    StreamingPipeline,
    StreamingPipelineConfig,
    StreamingPipelineResults,
)
from bookmark_processor.core.streaming.writer import AppendingBookmarkWriter


# ============ Fixtures ============


@pytest.fixture
def sample_csv_content():
    """Sample CSV content in raindrop.io export format."""
    return [
        ["id", "title", "note", "excerpt", "url", "folder", "tags", "created", "cover", "highlights", "favorite"],
        ["1", "Test Site 1", "Note 1", "Excerpt 1", "https://example.com/1", "Tech", "test, example", "2024-01-01T00:00:00Z", "", "", "false"],
        ["2", "Test Site 2", "Note 2", "Excerpt 2", "https://example.com/2", "Tech/AI", "ai, ml", "2024-01-02T00:00:00Z", "", "", "true"],
        ["3", "Test Site 3", "", "", "https://example.com/3", "Science", "science", "2024-01-03T00:00:00Z", "", "", "false"],
    ]


@pytest.fixture
def sample_csv_file(sample_csv_content, tmp_path):
    """Create a temporary CSV file with sample content."""
    csv_file = tmp_path / "test_bookmarks.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(sample_csv_content)
    return csv_file


@pytest.fixture
def large_csv_file(tmp_path):
    """Create a larger CSV file for batch testing."""
    csv_file = tmp_path / "large_bookmarks.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "note", "excerpt", "url", "folder", "tags", "created", "cover", "highlights", "favorite"])
        for i in range(250):
            writer.writerow([
                str(i),
                f"Test Site {i}",
                f"Note {i}",
                f"Excerpt {i}",
                f"https://example.com/{i}",
                f"Folder{i % 5}",
                f"tag{i % 10}",
                "2024-01-01T00:00:00Z",
                "",
                "",
                "false"
            ])
    return csv_file


@pytest.fixture
def sample_bookmarks():
    """Create sample Bookmark objects."""
    return [
        Bookmark(
            id="1",
            url="https://example.com/1",
            title="Test Site 1",
            note="Note 1",
            folder="Tech",
            tags=["test", "example"]
        ),
        Bookmark(
            id="2",
            url="https://example.com/2",
            title="Test Site 2",
            note="Note 2",
            folder="Tech/AI",
            tags=["ai", "ml"]
        ),
        Bookmark(
            id="3",
            url="https://example.com/3",
            title="Test Site 3",
            folder="Science",
            tags=["science"]
        ),
    ]


# ============ StreamingBookmarkReader Tests ============


class TestStreamingBookmarkReader:
    """Tests for StreamingBookmarkReader."""

    def test_init_valid_file(self, sample_csv_file):
        """Test initialization with valid file."""
        reader = StreamingBookmarkReader(sample_csv_file)
        assert reader.input_path == sample_csv_file
        assert reader.total_count is None  # Not counted yet

    def test_init_missing_file(self, tmp_path):
        """Test initialization with missing file raises error."""
        with pytest.raises(FileNotFoundError):
            StreamingBookmarkReader(tmp_path / "nonexistent.csv")

    def test_init_directory_raises_error(self, tmp_path):
        """Test initialization with directory raises error."""
        with pytest.raises(ValueError):
            StreamingBookmarkReader(tmp_path)

    def test_stream_yields_bookmarks(self, sample_csv_file):
        """Test stream() yields Bookmark objects."""
        reader = StreamingBookmarkReader(sample_csv_file)
        bookmarks = list(reader.stream())

        assert len(bookmarks) == 3
        assert all(isinstance(b, Bookmark) for b in bookmarks)
        assert bookmarks[0].url == "https://example.com/1"
        assert bookmarks[1].url == "https://example.com/2"
        assert bookmarks[2].url == "https://example.com/3"

    def test_stream_parses_tags(self, sample_csv_file):
        """Test stream correctly parses tags."""
        reader = StreamingBookmarkReader(sample_csv_file)
        bookmarks = list(reader.stream())

        assert bookmarks[0].tags == ["test", "example"]
        assert bookmarks[1].tags == ["ai", "ml"]

    def test_stream_parses_datetime(self, sample_csv_file):
        """Test stream correctly parses datetime."""
        reader = StreamingBookmarkReader(sample_csv_file)
        bookmarks = list(reader.stream())

        assert bookmarks[0].created is not None
        assert bookmarks[0].created.year == 2024
        assert bookmarks[0].created.month == 1
        assert bookmarks[0].created.day == 1

    def test_stream_parses_boolean(self, sample_csv_file):
        """Test stream correctly parses boolean fields."""
        reader = StreamingBookmarkReader(sample_csv_file)
        bookmarks = list(reader.stream())

        assert bookmarks[0].favorite is False
        assert bookmarks[1].favorite is True

    def test_stream_batches(self, large_csv_file):
        """Test stream_batches() yields batches of correct size."""
        reader = StreamingBookmarkReader(large_csv_file)
        batches = list(reader.stream_batches(batch_size=100))

        # 250 items / 100 batch_size = 3 batches (100, 100, 50)
        assert len(batches) == 3
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100
        assert len(batches[2]) == 50

    def test_stream_batches_small_size(self, sample_csv_file):
        """Test stream_batches with small batch size."""
        reader = StreamingBookmarkReader(sample_csv_file)
        batches = list(reader.stream_batches(batch_size=1))

        assert len(batches) == 3
        assert all(len(b) == 1 for b in batches)

    def test_stream_batches_invalid_size(self, sample_csv_file):
        """Test stream_batches raises error for invalid batch size."""
        reader = StreamingBookmarkReader(sample_csv_file)
        with pytest.raises(ValueError):
            list(reader.stream_batches(batch_size=0))

    def test_stream_with_index(self, sample_csv_file):
        """Test stream_with_index yields index with bookmark."""
        reader = StreamingBookmarkReader(sample_csv_file)
        items = list(reader.stream_with_index())

        assert len(items) == 3
        assert items[0] == (0, items[0][1])
        assert items[1] == (1, items[1][1])
        assert items[2] == (2, items[2][1])

    def test_count_rows(self, sample_csv_file):
        """Test count_rows() returns correct count."""
        reader = StreamingBookmarkReader(sample_csv_file)
        count = reader.count_rows()

        assert count == 3
        assert reader.total_count == 3

    def test_peek(self, sample_csv_file):
        """Test peek() returns first N bookmarks."""
        reader = StreamingBookmarkReader(sample_csv_file)
        preview = reader.peek(count=2)

        assert len(preview) == 2
        assert preview[0].url == "https://example.com/1"
        assert preview[1].url == "https://example.com/2"

    def test_get_sample(self, sample_csv_file):
        """Test get_sample() with skip parameter."""
        reader = StreamingBookmarkReader(sample_csv_file)
        sample = reader.get_sample(count=2, skip=1)

        assert len(sample) == 2
        assert sample[0].url == "https://example.com/2"
        assert sample[1].url == "https://example.com/3"

    def test_iterator_protocol(self, sample_csv_file):
        """Test reader implements iterator protocol."""
        reader = StreamingBookmarkReader(sample_csv_file)
        bookmarks = []
        for bookmark in reader:
            bookmarks.append(bookmark)

        assert len(bookmarks) == 3

    def test_skip_invalid_rows(self, tmp_path):
        """Test skipping rows with missing URL."""
        csv_file = tmp_path / "invalid.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows([
                ["id", "title", "note", "excerpt", "url", "folder", "tags", "created", "cover", "highlights", "favorite"],
                ["1", "Valid", "", "", "https://valid.com", "", "", "", "", "", ""],
                ["2", "Invalid", "", "", "", "", "", "", "", "", ""],  # Missing URL
                ["3", "Valid 2", "", "", "https://valid2.com", "", "", "", "", "", ""],
            ])

        reader = StreamingBookmarkReader(csv_file, skip_invalid=True)
        bookmarks = list(reader.stream())

        assert len(bookmarks) == 2
        assert bookmarks[0].url == "https://valid.com"
        assert bookmarks[1].url == "https://valid2.com"

    def test_repr(self, sample_csv_file):
        """Test string representation."""
        reader = StreamingBookmarkReader(sample_csv_file)
        assert "StreamingBookmarkReader" in repr(reader)


# ============ StreamingBookmarkWriter Tests ============


class TestStreamingBookmarkWriter:
    """Tests for StreamingBookmarkWriter."""

    def test_init(self, tmp_path):
        """Test initialization."""
        output_path = tmp_path / "output.csv"
        writer = StreamingBookmarkWriter(output_path)

        assert writer.output_path == output_path
        assert writer.written_count == 0

    def test_context_manager(self, tmp_path, sample_bookmarks):
        """Test context manager usage."""
        output_path = tmp_path / "output.csv"

        with StreamingBookmarkWriter(output_path) as writer:
            writer.write(sample_bookmarks[0])

        assert output_path.exists()

    def test_write_single_bookmark(self, tmp_path, sample_bookmarks):
        """Test writing a single bookmark."""
        output_path = tmp_path / "output.csv"

        with StreamingBookmarkWriter(output_path) as writer:
            writer.write(sample_bookmarks[0])
            assert writer.written_count == 1

        # Verify output
        with open(output_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["url"] == "https://example.com/1"

    def test_write_batch(self, tmp_path, sample_bookmarks):
        """Test writing a batch of bookmarks."""
        output_path = tmp_path / "output.csv"

        with StreamingBookmarkWriter(output_path) as writer:
            written = writer.write_batch(sample_bookmarks)
            assert written == 3
            assert writer.written_count == 3

        # Verify output
        with open(output_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3

    def test_write_skips_invalid(self, tmp_path):
        """Test that invalid bookmarks (no URL) are skipped."""
        output_path = tmp_path / "output.csv"
        invalid_bookmark = Bookmark(url="")

        with StreamingBookmarkWriter(output_path) as writer:
            writer.write(invalid_bookmark)
            assert writer.written_count == 0

    def test_header_written(self, tmp_path, sample_bookmarks):
        """Test CSV header is written."""
        output_path = tmp_path / "output.csv"

        with StreamingBookmarkWriter(output_path) as writer:
            writer.write(sample_bookmarks[0])

        with open(output_path, "r", encoding="utf-8-sig") as f:
            first_line = f.readline().strip()

        assert "url" in first_line
        assert "folder" in first_line
        assert "title" in first_line

    def test_flush(self, tmp_path, sample_bookmarks):
        """Test flush method."""
        output_path = tmp_path / "output.csv"

        with StreamingBookmarkWriter(output_path) as writer:
            writer.write(sample_bookmarks[0])
            writer.flush()

        assert output_path.exists()

    def test_get_statistics(self, tmp_path, sample_bookmarks):
        """Test get_statistics method."""
        output_path = tmp_path / "output.csv"

        with StreamingBookmarkWriter(output_path) as writer:
            writer.write_batch(sample_bookmarks)
            stats = writer.get_statistics()

        assert stats["written_count"] == 3
        assert stats["is_open"] is True

    def test_creates_parent_directories(self, tmp_path, sample_bookmarks):
        """Test that parent directories are created."""
        output_path = tmp_path / "subdir" / "deep" / "output.csv"

        with StreamingBookmarkWriter(output_path) as writer:
            writer.write(sample_bookmarks[0])

        assert output_path.exists()

    def test_write_without_open_raises_error(self, tmp_path):
        """Test writing without opening raises error."""
        writer = StreamingBookmarkWriter(tmp_path / "output.csv")

        with pytest.raises(RuntimeError):
            writer.write(Bookmark(url="https://test.com"))

    def test_repr(self, tmp_path):
        """Test string representation."""
        writer = StreamingBookmarkWriter(tmp_path / "output.csv")
        assert "StreamingBookmarkWriter" in repr(writer)


# ============ AppendingBookmarkWriter Tests ============


class TestAppendingBookmarkWriter:
    """Tests for AppendingBookmarkWriter."""

    def test_append_to_existing_file(self, tmp_path, sample_bookmarks):
        """Test appending to existing file."""
        output_path = tmp_path / "output.csv"

        # Create initial file
        with StreamingBookmarkWriter(output_path) as writer:
            writer.write(sample_bookmarks[0])

        # Append to it
        with AppendingBookmarkWriter(output_path) as writer:
            writer.write(sample_bookmarks[1])

        # Verify both entries exist
        with open(output_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2

    def test_creates_new_file_with_header(self, tmp_path, sample_bookmarks):
        """Test creating new file when appending to non-existent."""
        output_path = tmp_path / "new_output.csv"

        with AppendingBookmarkWriter(output_path) as writer:
            writer.write(sample_bookmarks[0])

        with open(output_path, "r", encoding="utf-8-sig") as f:
            first_line = f.readline()

        assert "url" in first_line


# ============ StreamingPipeline Tests ============


class TestStreamingPipeline:
    """Tests for StreamingPipeline."""

    def test_init(self, sample_csv_file, tmp_path):
        """Test pipeline initialization."""
        config = StreamingPipelineConfig(
            input_file=sample_csv_file,
            output_file=tmp_path / "output.csv"
        )
        pipeline = StreamingPipeline(config)

        assert pipeline.config == config

    @patch('bookmark_processor.core.streaming.pipeline.StreamingPipeline._get_url_validator')
    def test_execute_basic(self, mock_validator, sample_csv_file, tmp_path):
        """Test basic pipeline execution."""
        # Mock validator to always return valid
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_url.return_value = MagicMock(is_valid=True)
        mock_validator.return_value = mock_validator_instance

        config = StreamingPipelineConfig(
            input_file=sample_csv_file,
            output_file=tmp_path / "output.csv",
            ai_enabled=False,
            use_state_tracker=False
        )
        pipeline = StreamingPipeline(config)

        # Mock other components
        pipeline._get_content_analyzer = MagicMock(return_value=None)
        pipeline._get_tag_generator = MagicMock(return_value=MagicMock(
            generate_for_single_bookmark=MagicMock(return_value=["test"])
        ))

        results = pipeline.execute()

        assert isinstance(results, StreamingPipelineResults)
        assert results.stats.total_read == 3

    def test_execute_streaming_with_mocked_components(self, sample_csv_file, tmp_path):
        """Test execute_streaming with mocked components."""
        config = StreamingPipelineConfig(
            input_file=sample_csv_file,
            output_file=tmp_path / "output.csv",
            ai_enabled=False,
            use_state_tracker=False
        )

        reader = StreamingBookmarkReader(sample_csv_file)
        writer = StreamingBookmarkWriter(tmp_path / "output.csv")

        pipeline = StreamingPipeline(config)

        # Mock all processing methods to just pass through
        pipeline._validate_url = MagicMock(return_value=True)
        pipeline._analyze_content = MagicMock(return_value=None)
        pipeline._generate_tags = MagicMock(return_value=["test"])

        with writer:
            results = pipeline.execute_streaming(reader, writer)

        assert results.completed is True
        assert results.stats.total_processed == 3

    def test_statistics(self, sample_csv_file, tmp_path):
        """Test statistics collection."""
        config = StreamingPipelineConfig(
            input_file=sample_csv_file,
            output_file=tmp_path / "output.csv"
        )
        pipeline = StreamingPipeline(config)
        stats = pipeline.get_statistics()

        assert "total_read" in stats
        assert "total_processed" in stats

    def test_repr(self, sample_csv_file, tmp_path):
        """Test string representation."""
        config = StreamingPipelineConfig(
            input_file=sample_csv_file,
            output_file=tmp_path / "output.csv"
        )
        pipeline = StreamingPipeline(config)

        assert "StreamingPipeline" in repr(pipeline)


# ============ ProcessingStats Tests ============


class TestProcessingStats:
    """Tests for ProcessingStats dataclass."""

    def test_processing_time(self):
        """Test processing_time calculation."""
        from bookmark_processor.core.streaming.pipeline import ProcessingStats

        stats = ProcessingStats()
        stats.start_time = datetime(2024, 1, 1, 0, 0, 0)
        stats.end_time = datetime(2024, 1, 1, 0, 1, 30)

        assert stats.processing_time.total_seconds() == 90

    def test_success_rate(self):
        """Test success_rate calculation."""
        from bookmark_processor.core.streaming.pipeline import ProcessingStats

        stats = ProcessingStats()
        stats.total_read = 100
        stats.total_processed = 80

        assert stats.success_rate == 80.0

    def test_success_rate_zero_division(self):
        """Test success_rate with zero total_read."""
        from bookmark_processor.core.streaming.pipeline import ProcessingStats

        stats = ProcessingStats()
        assert stats.success_rate == 0.0

    def test_throughput(self):
        """Test throughput calculation."""
        from bookmark_processor.core.streaming.pipeline import ProcessingStats

        stats = ProcessingStats()
        stats.total_processed = 100
        stats.start_time = datetime(2024, 1, 1, 0, 0, 0)
        stats.end_time = datetime(2024, 1, 1, 0, 0, 10)  # 10 seconds

        assert stats.throughput == 10.0  # 100 items / 10 seconds

    def test_to_dict(self):
        """Test to_dict conversion."""
        from bookmark_processor.core.streaming.pipeline import ProcessingStats

        stats = ProcessingStats()
        stats.total_read = 100
        stats.total_processed = 80

        d = stats.to_dict()

        assert d["total_read"] == 100
        assert d["total_processed"] == 80
        assert "success_rate" in d
        assert "throughput" in d
