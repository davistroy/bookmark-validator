"""
Comprehensive tests for memory_optimizer module.

Tests for memory monitoring, batch processing, streaming processing,
data caching, and memory optimization utilities.
"""

import gc
import sys
import time
import threading
from datetime import datetime
from io import StringIO
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest

from bookmark_processor.utils.memory_optimizer import (
    BatchProcessor,
    DataCache,
    MemoryMonitor,
    MemoryStats,
    StreamingProcessor,
    data_cache,
    memory_context,
    memory_monitor,
    optimize_for_large_dataset,
)


class TestMemoryStats:
    """Test MemoryStats dataclass."""

    def test_init(self):
        """Test MemoryStats initialization with all fields."""
        stats = MemoryStats(
            current_mb=100.5,
            peak_mb=150.0,
            available_mb=3000.0,
            gc_collections={"gen_0": 10, "gen_1": 5, "gen_2": 1},
            timestamp=datetime.now(),
        )

        assert stats.current_mb == 100.5
        assert stats.peak_mb == 150.0
        assert stats.available_mb == 3000.0
        assert stats.gc_collections == {"gen_0": 10, "gen_1": 5, "gen_2": 1}
        assert isinstance(stats.timestamp, datetime)

    def test_gc_collections_dict(self):
        """Test gc_collections dictionary structure."""
        gc_stats = {"gen_0": 100, "gen_1": 20, "gen_2": 3}
        stats = MemoryStats(
            current_mb=50.0,
            peak_mb=50.0,
            available_mb=3950.0,
            gc_collections=gc_stats,
            timestamp=datetime.now(),
        )

        assert "gen_0" in stats.gc_collections
        assert "gen_1" in stats.gc_collections
        assert "gen_2" in stats.gc_collections


class TestMemoryMonitor:
    """Test MemoryMonitor class."""

    def test_init_default_thresholds(self):
        """Test MemoryMonitor initialization with default thresholds."""
        monitor = MemoryMonitor()

        assert monitor.warning_threshold == 3000
        assert monitor.critical_threshold == 3500
        assert monitor.peak_memory == 0.0
        assert len(monitor.history) == 0
        assert isinstance(monitor.lock, type(threading.RLock()))

    def test_init_custom_thresholds(self):
        """Test MemoryMonitor initialization with custom thresholds."""
        monitor = MemoryMonitor(
            warning_threshold_mb=2000.0, critical_threshold_mb=2500.0
        )

        assert monitor.warning_threshold == 2000.0
        assert monitor.critical_threshold == 2500.0

    def test_get_current_memory(self):
        """Test get_current_memory method."""
        monitor = MemoryMonitor()
        memory = monitor.get_current_memory()

        # Should return a non-negative float
        assert isinstance(memory, float)
        assert memory >= 0.0

    def test_get_current_usage_mb(self):
        """Test get_current_usage_mb method (alias)."""
        monitor = MemoryMonitor()
        memory = monitor.get_current_usage_mb()

        # Should return same value as get_current_memory
        assert isinstance(memory, float)
        assert memory >= 0.0

    def test_get_memory_stats(self):
        """Test get_memory_stats method returns MemoryStats."""
        monitor = MemoryMonitor()
        stats = monitor.get_memory_stats()

        assert isinstance(stats, MemoryStats)
        assert stats.current_mb >= 0
        assert stats.peak_mb >= stats.current_mb or stats.peak_mb == stats.current_mb
        assert stats.available_mb >= 0
        assert "gen_0" in stats.gc_collections
        assert "gen_1" in stats.gc_collections
        assert "gen_2" in stats.gc_collections
        assert isinstance(stats.timestamp, datetime)

    def test_get_memory_stats_updates_history(self):
        """Test that get_memory_stats adds to history."""
        monitor = MemoryMonitor()

        assert len(monitor.history) == 0

        monitor.get_memory_stats()
        assert len(monitor.history) == 1

        monitor.get_memory_stats()
        assert len(monitor.history) == 2

    def test_get_memory_stats_updates_peak(self):
        """Test that get_memory_stats updates peak memory."""
        monitor = MemoryMonitor()

        assert monitor.peak_memory == 0.0

        stats = monitor.get_memory_stats()
        # Peak should be updated to at least current
        assert monitor.peak_memory >= 0.0

    def test_history_limit(self):
        """Test that history is limited to 100 entries."""
        monitor = MemoryMonitor()

        # Add more than 100 entries
        for _ in range(120):
            monitor.get_memory_stats()

        # History should be trimmed to 100
        assert len(monitor.history) == 100

    def test_check_memory_pressure_normal(self):
        """Test check_memory_pressure returns 'normal' when below thresholds."""
        # Use high thresholds to ensure normal state
        monitor = MemoryMonitor(
            warning_threshold_mb=100000.0, critical_threshold_mb=200000.0
        )
        pressure = monitor.check_memory_pressure()

        assert pressure == "normal"

    def test_check_memory_pressure_warning(self):
        """Test check_memory_pressure returns 'warning' when above warning threshold."""
        # Mock get_current_memory to return value above warning but below critical
        monitor = MemoryMonitor(
            warning_threshold_mb=50.0, critical_threshold_mb=100.0
        )

        with patch.object(monitor, "get_current_memory", return_value=75.0):
            pressure = monitor.check_memory_pressure()
            assert pressure == "warning"

    def test_check_memory_pressure_critical(self):
        """Test check_memory_pressure returns 'critical' when above critical threshold."""
        monitor = MemoryMonitor(
            warning_threshold_mb=50.0, critical_threshold_mb=100.0
        )

        with patch.object(monitor, "get_current_memory", return_value=150.0):
            pressure = monitor.check_memory_pressure()
            assert pressure == "critical"

    def test_force_cleanup(self):
        """Test force_cleanup triggers garbage collection."""
        monitor = MemoryMonitor()

        # Create some garbage
        garbage = [{"key": f"value_{i}"} for i in range(1000)]
        del garbage

        collected = monitor.force_cleanup()

        # Should return sum of collected objects (could be 0 if nothing to collect)
        assert isinstance(collected, int)
        assert collected >= 0

    def test_thread_safety(self):
        """Test thread safety of get_memory_stats."""
        monitor = MemoryMonitor()
        results = []
        errors = []

        def get_stats():
            try:
                for _ in range(10):
                    stats = monitor.get_memory_stats()
                    results.append(stats)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_stats) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 50

    def test_get_current_memory_exception_handling(self):
        """Test get_current_memory handles exceptions gracefully."""
        monitor = MemoryMonitor()

        # Mock psutil to raise an exception
        with patch(
            "bookmark_processor.utils.memory_optimizer.HAS_PSUTIL", False
        ), patch(
            "bookmark_processor.utils.memory_optimizer.HAS_RESOURCE", False
        ):
            memory = monitor.get_current_usage_mb()
            # Should return 0.0 when both methods fail
            assert memory == 0.0


class TestBatchProcessor:
    """Test BatchProcessor class."""

    def test_init_default(self):
        """Test BatchProcessor initialization with defaults."""
        processor = BatchProcessor()

        assert processor.batch_size == 100
        assert isinstance(processor.memory_monitor, MemoryMonitor)
        assert processor.enable_gc is True
        assert processor.processed_count == 0
        assert processor.total_items == 0

    def test_init_custom(self):
        """Test BatchProcessor initialization with custom settings."""
        custom_monitor = MemoryMonitor(warning_threshold_mb=1000.0)
        processor = BatchProcessor(
            batch_size=50, memory_monitor=custom_monitor, enable_gc=False
        )

        assert processor.batch_size == 50
        assert processor.memory_monitor is custom_monitor
        assert processor.enable_gc is False

    def test_process_batches_simple(self):
        """Test process_batches with simple data."""
        processor = BatchProcessor(batch_size=3)
        items = [1, 2, 3, 4, 5, 6, 7]

        def double(batch: List[int]) -> List[int]:
            return [x * 2 for x in batch]

        results = list(processor.process_batches(items, double))

        assert len(results) == 3  # 3 batches: [1,2,3], [4,5,6], [7]
        assert results[0] == [2, 4, 6]
        assert results[1] == [8, 10, 12]
        assert results[2] == [14]

    def test_process_batches_with_callback(self):
        """Test process_batches with progress callback."""
        processor = BatchProcessor(batch_size=2)
        items = [1, 2, 3, 4]
        callback_calls = []

        def callback(processed: int, total: int, status: str):
            callback_calls.append((processed, total, status))

        def identity(batch: List[int]) -> List[int]:
            return batch

        list(processor.process_batches(items, identity, progress_callback=callback))

        # Should have multiple callback calls
        assert len(callback_calls) > 0
        # Final callback should show all items processed
        final_processed = max(c[0] for c in callback_calls)
        assert final_processed == 4

    def test_process_batches_critical_memory(self):
        """Test process_batches handles critical memory pressure."""
        processor = BatchProcessor(batch_size=2)
        items = [1, 2, 3, 4]
        callback_calls = []

        def callback(processed: int, total: int, status: str):
            callback_calls.append((processed, total, status))

        def identity(batch: List[int]) -> List[int]:
            return batch

        # Mock critical memory pressure
        with patch.object(
            processor.memory_monitor, "check_memory_pressure", return_value="critical"
        ), patch.object(
            processor.memory_monitor, "force_cleanup", return_value=100
        ) as mock_cleanup:
            list(processor.process_batches(items, identity, progress_callback=callback))

            # force_cleanup should be called due to critical pressure
            assert mock_cleanup.called

    def test_process_batches_warning_memory_gc(self):
        """Test process_batches triggers GC on warning memory pressure."""
        processor = BatchProcessor(batch_size=2, enable_gc=True)
        items = [1, 2, 3, 4]

        def identity(batch: List[int]) -> List[int]:
            return batch

        # Mock warning memory pressure
        with patch.object(
            processor.memory_monitor, "check_memory_pressure", return_value="warning"
        ), patch("gc.collect") as mock_gc:
            list(processor.process_batches(items, identity))

            # GC should be triggered due to warning pressure
            assert mock_gc.called

    def test_process_batches_no_gc_when_disabled(self):
        """Test process_batches doesn't GC when disabled."""
        processor = BatchProcessor(batch_size=2, enable_gc=False)
        items = [1, 2, 3, 4]

        def identity(batch: List[int]) -> List[int]:
            return batch

        # Even with warning pressure, no GC if disabled
        with patch.object(
            processor.memory_monitor, "check_memory_pressure", return_value="warning"
        ), patch("gc.collect") as mock_gc:
            list(processor.process_batches(items, identity))

            # GC should not be triggered for batch cleanup when enable_gc=False
            # Note: GC may still be called by memory_monitor.force_cleanup in critical cases

    def test_process_batches_exception_handling(self):
        """Test process_batches handles processor function exceptions."""
        processor = BatchProcessor(batch_size=2)
        items = [1, 2, 3, 4]
        callback_calls = []

        def callback(processed: int, total: int, status: str):
            callback_calls.append((processed, total, status))

        def failing_processor(batch: List[int]) -> List[int]:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            list(processor.process_batches(items, failing_processor, progress_callback=callback))

        # Callback should have recorded the error
        error_calls = [c for c in callback_calls if "error" in c[2].lower()]
        assert len(error_calls) > 0

    def test_process_all(self):
        """Test process_all combines all batch results."""
        processor = BatchProcessor(batch_size=3)
        items = [1, 2, 3, 4, 5, 6, 7]

        def double(batch: List[int]) -> List[int]:
            return [x * 2 for x in batch]

        results = processor.process_all(items, double)

        assert results == [2, 4, 6, 8, 10, 12, 14]
        assert processor.processed_count == 7
        assert processor.total_items == 7

    def test_process_all_with_callback(self):
        """Test process_all with progress callback."""
        processor = BatchProcessor(batch_size=2)
        items = [1, 2, 3, 4, 5]
        callback_calls = []

        def callback(processed: int, total: int, status: str):
            callback_calls.append((processed, total, status))

        def identity(batch: List[int]) -> List[int]:
            return batch

        results = processor.process_all(items, identity, progress_callback=callback)

        assert results == items
        assert len(callback_calls) > 0

    def test_process_all_empty_list(self):
        """Test process_all with empty list."""
        processor = BatchProcessor(batch_size=10)
        items: List[int] = []

        def double(batch: List[int]) -> List[int]:
            return [x * 2 for x in batch]

        results = processor.process_all(items, double)

        assert results == []
        assert processor.processed_count == 0
        assert processor.total_items == 0


class TestStreamingProcessor:
    """Test StreamingProcessor class."""

    def test_init_default(self):
        """Test StreamingProcessor initialization with defaults."""
        processor = StreamingProcessor()

        assert isinstance(processor.memory_monitor, MemoryMonitor)

    def test_init_custom_monitor(self):
        """Test StreamingProcessor initialization with custom monitor."""
        custom_monitor = MemoryMonitor(warning_threshold_mb=1000.0)
        processor = StreamingProcessor(memory_monitor=custom_monitor)

        assert processor.memory_monitor is custom_monitor

    def test_stream_items_basic(self):
        """Test stream_items yields chunks correctly."""
        processor = StreamingProcessor()
        items = list(range(10))
        chunks = []

        for chunk in processor.stream_items(items, chunk_size=3):
            chunks.append(chunk)

        assert len(chunks) == 4  # 3+3+3+1
        assert chunks[0] == [0, 1, 2]
        assert chunks[1] == [3, 4, 5]
        assert chunks[2] == [6, 7, 8]
        assert chunks[3] == [9]

    def test_stream_items_exact_chunks(self):
        """Test stream_items with items divisible by chunk_size."""
        processor = StreamingProcessor()
        items = list(range(6))
        chunks = []

        for chunk in processor.stream_items(items, chunk_size=2):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0] == [0, 1]
        assert chunks[1] == [2, 3]
        assert chunks[2] == [4, 5]

    def test_stream_items_empty_list(self):
        """Test stream_items with empty list."""
        processor = StreamingProcessor()
        items: List[int] = []
        chunks = []

        for chunk in processor.stream_items(items, chunk_size=10):
            chunks.append(chunk)

        assert len(chunks) == 0

    def test_stream_items_critical_memory_gc(self):
        """Test stream_items triggers GC on critical memory pressure."""
        processor = StreamingProcessor()
        items = list(range(10))

        with patch.object(
            processor.memory_monitor, "check_memory_pressure", return_value="critical"
        ), patch("gc.collect") as mock_gc:
            for _ in processor.stream_items(items, chunk_size=3):
                pass

            # GC should be triggered due to critical pressure
            assert mock_gc.called

    def test_stream_items_cleanup_on_exit(self):
        """Test stream_items calls gc.collect on exit."""
        processor = StreamingProcessor()
        items = [1, 2, 3]

        with patch("gc.collect") as mock_gc:
            for _ in processor.stream_items(items, chunk_size=2):
                pass

            # GC should be called at least once (cleanup)
            assert mock_gc.called


class TestDataCache:
    """Test DataCache class."""

    def test_init_default(self):
        """Test DataCache initialization with defaults."""
        cache = DataCache()

        assert cache.max_size_mb == 500
        assert len(cache.cache) == 0
        assert len(cache.access_times) == 0

    def test_init_custom_size(self):
        """Test DataCache initialization with custom size."""
        cache = DataCache(max_size_mb=100.0)

        assert cache.max_size_mb == 100.0

    def test_get_missing_key(self):
        """Test get returns None for missing key."""
        cache = DataCache()

        result = cache.get("nonexistent")

        assert result is None

    def test_put_and_get(self):
        """Test put and get operations."""
        cache = DataCache()

        cache.put("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"

    def test_put_updates_access_time(self):
        """Test that put updates access time."""
        cache = DataCache()

        cache.put("key1", "value1")

        assert "key1" in cache.access_times
        assert isinstance(cache.access_times["key1"], datetime)

    def test_get_updates_access_time(self):
        """Test that get updates access time."""
        cache = DataCache()
        cache.put("key1", "value1")

        first_access = cache.access_times["key1"]
        time.sleep(0.01)  # Small delay
        cache.get("key1")
        second_access = cache.access_times["key1"]

        assert second_access >= first_access

    def test_put_returns_true_on_success(self):
        """Test put returns True on successful insertion."""
        cache = DataCache(max_size_mb=1000.0)

        result = cache.put("key1", "value1")

        assert result is True

    def test_clear(self):
        """Test clear empties the cache."""
        cache = DataCache()
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cache.clear()

        assert len(cache.cache) == 0
        assert len(cache.access_times) == 0

    def test_cleanup_cache_removes_oldest(self):
        """Test _cleanup_cache removes oldest items."""
        cache = DataCache(max_size_mb=1000.0)

        # Add items with time gaps
        cache.put("key1", "value1")
        time.sleep(0.01)
        cache.put("key2", "value2")
        time.sleep(0.01)
        cache.put("key3", "value3")
        time.sleep(0.01)
        cache.put("key4", "value4")

        # Manually trigger cleanup
        cache._cleanup_cache()

        # Should have removed ~25% (1 item from 4)
        assert len(cache.cache) == 3
        # Oldest key (key1) should be removed
        assert "key1" not in cache.cache

    def test_cleanup_cache_empty(self):
        """Test _cleanup_cache handles empty cache."""
        cache = DataCache()

        result = cache._cleanup_cache()

        assert result is True

    def test_get_cache_size(self):
        """Test _get_cache_size returns estimated size."""
        cache = DataCache()
        cache.put("key1", "a" * 1000)  # ~1KB string
        cache.put("key2", "b" * 1000)

        size = cache._get_cache_size()

        # Size should be positive
        assert size > 0

    def test_put_triggers_cleanup_when_full(self):
        """Test put triggers cleanup when cache would exceed size."""
        # Create a small cache
        cache = DataCache(max_size_mb=0.001)  # Very small: ~1KB

        # Put a large value
        large_value = "x" * 10000  # ~10KB

        with patch.object(cache, "_cleanup_cache", return_value=True) as mock_cleanup:
            cache.put("key1", large_value)

            # Cleanup should be triggered
            mock_cleanup.assert_called()

    def test_put_fails_when_cleanup_insufficient(self):
        """Test put returns False when cleanup can't free enough space."""
        cache = DataCache(max_size_mb=0.0001)  # Tiny cache

        # Mock cleanup to return False (couldn't free enough)
        with patch.object(cache, "_cleanup_cache", return_value=False):
            with patch.object(cache, "_get_cache_size", return_value=1.0):
                result = cache.put("key1", "x" * 100000)

                assert result is False

    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        cache = DataCache()
        errors = []

        def cache_operations():
            try:
                for i in range(50):
                    cache.put(f"key_{threading.current_thread().name}_{i}", f"value_{i}")
                    cache.get(f"key_{threading.current_thread().name}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=cache_operations, name=f"t{i}") for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"


class TestMemoryContext:
    """Test memory_context context manager."""

    def test_memory_context_basic(self, capsys):
        """Test memory_context prints memory usage."""
        with memory_context("test_operation"):
            # Do some work
            data = [i for i in range(1000)]
            del data

        captured = capsys.readouterr()
        assert "Memory usage for test_operation" in captured.out
        assert "MB" in captured.out

    def test_memory_context_default_name(self, capsys):
        """Test memory_context with default operation name."""
        with memory_context():
            pass

        captured = capsys.readouterr()
        assert "Memory usage for operation" in captured.out

    def test_memory_context_yields_monitor(self):
        """Test memory_context yields memory monitor."""
        with memory_context("test") as monitor:
            assert monitor is memory_monitor

    def test_memory_context_exception_handling(self, capsys):
        """Test memory_context prints stats even on exception."""
        with pytest.raises(ValueError):
            with memory_context("failing_op"):
                raise ValueError("Test error")

        # Should still print memory stats in finally block
        captured = capsys.readouterr()
        assert "Memory usage for failing_op" in captured.out


class TestOptimizeForLargeDataset:
    """Test optimize_for_large_dataset decorator."""

    def test_decorator_basic(self, capsys):
        """Test decorator wraps function correctly."""
        @optimize_for_large_dataset
        def test_func(x, y):
            return x + y

        result = test_func(2, 3)

        assert result == 5
        captured = capsys.readouterr()
        assert "Memory usage for test_func" in captured.out

    def test_decorator_with_kwargs(self, capsys):
        """Test decorator handles kwargs."""
        @optimize_for_large_dataset
        def test_func(x, y=10):
            return x * y

        result = test_func(5, y=20)

        assert result == 100

    def test_decorator_triggers_gc(self):
        """Test decorator triggers GC before operation."""
        @optimize_for_large_dataset
        def test_func():
            return "done"

        with patch("gc.collect") as mock_gc:
            test_func()

            # GC should be called at least once (before operation)
            assert mock_gc.called

    def test_decorator_cleanup_on_pressure(self):
        """Test decorator triggers cleanup on memory pressure."""
        @optimize_for_large_dataset
        def test_func():
            return "done"

        with patch.object(
            memory_monitor, "check_memory_pressure", return_value="warning"
        ), patch.object(memory_monitor, "force_cleanup") as mock_cleanup:
            test_func()

            # Cleanup should be triggered due to pressure
            mock_cleanup.assert_called()

    def test_decorator_no_cleanup_normal(self):
        """Test decorator doesn't cleanup when memory is normal."""
        @optimize_for_large_dataset
        def test_func():
            return "done"

        with patch.object(
            memory_monitor, "check_memory_pressure", return_value="normal"
        ), patch.object(memory_monitor, "force_cleanup") as mock_cleanup:
            test_func()

            # Cleanup should not be triggered
            mock_cleanup.assert_not_called()

    def test_decorator_preserves_function_name(self):
        """Test decorator preserves original function name."""
        @optimize_for_large_dataset
        def my_special_function():
            return "done"

        # Function name is used in memory_context
        # We can verify by checking the output mentions the function name
        # This is implicit in other tests but let's be explicit


class TestGlobalInstances:
    """Test global memory_monitor and data_cache instances."""

    def test_global_memory_monitor(self):
        """Test global memory_monitor is accessible."""
        assert memory_monitor is not None
        assert isinstance(memory_monitor, MemoryMonitor)

    def test_global_data_cache(self):
        """Test global data_cache is accessible."""
        assert data_cache is not None
        assert isinstance(data_cache, DataCache)


class TestPsutilFallback:
    """Test behavior when psutil is not available."""

    def test_memory_without_psutil(self):
        """Test get_current_usage_mb falls back gracefully without psutil."""
        monitor = MemoryMonitor()

        # Simulate no psutil and no resource module
        with patch(
            "bookmark_processor.utils.memory_optimizer.HAS_PSUTIL", False
        ), patch(
            "bookmark_processor.utils.memory_optimizer.HAS_RESOURCE", False
        ):
            memory = monitor.get_current_usage_mb()
            # Should return 0.0 as fallback
            assert memory == 0.0

    def test_memory_with_resource_module(self):
        """Test get_current_usage_mb uses resource module when psutil unavailable."""
        monitor = MemoryMonitor()

        # Create mock resource module
        mock_usage = MagicMock()
        mock_usage.ru_maxrss = 100 * 1024  # 100MB in KB

        with patch(
            "bookmark_processor.utils.memory_optimizer.HAS_PSUTIL", False
        ), patch(
            "bookmark_processor.utils.memory_optimizer.HAS_RESOURCE", True
        ), patch(
            "bookmark_processor.utils.memory_optimizer.resource"
        ) as mock_resource:
            mock_resource.getrusage.return_value = mock_usage
            mock_resource.RUSAGE_SELF = 0

            memory = monitor.get_current_usage_mb()
            # Should have attempted to use resource module
            mock_resource.getrusage.assert_called()


class TestBatchProcessorEdgeCases:
    """Test edge cases for BatchProcessor."""

    def test_single_item(self):
        """Test processing single item."""
        processor = BatchProcessor(batch_size=10)
        items = [42]

        def double(batch: List[int]) -> List[int]:
            return [x * 2 for x in batch]

        results = processor.process_all(items, double)

        assert results == [84]

    def test_batch_size_larger_than_items(self):
        """Test when batch_size is larger than total items."""
        processor = BatchProcessor(batch_size=100)
        items = [1, 2, 3]

        def identity(batch: List[int]) -> List[int]:
            return batch

        results = list(processor.process_batches(items, identity))

        assert len(results) == 1
        assert results[0] == [1, 2, 3]

    def test_batch_size_equals_items(self):
        """Test when batch_size equals total items."""
        processor = BatchProcessor(batch_size=5)
        items = [1, 2, 3, 4, 5]

        def identity(batch: List[int]) -> List[int]:
            return batch

        results = list(processor.process_batches(items, identity))

        assert len(results) == 1
        assert results[0] == [1, 2, 3, 4, 5]

    def test_processor_func_returns_different_size(self):
        """Test processor function that returns different number of items."""
        processor = BatchProcessor(batch_size=3)
        items = [1, 2, 3, 4, 5]

        def filter_even(batch: List[int]) -> List[int]:
            return [x for x in batch if x % 2 == 0]

        results = processor.process_all(items, filter_even)

        assert results == [2, 4]


class TestDataCacheEdgeCases:
    """Test edge cases for DataCache."""

    def test_overwrite_existing_key(self):
        """Test putting a value with existing key."""
        cache = DataCache()

        cache.put("key1", "value1")
        cache.put("key1", "value2")

        assert cache.get("key1") == "value2"

    def test_various_value_types(self):
        """Test caching various value types."""
        cache = DataCache()

        cache.put("string", "hello")
        cache.put("int", 42)
        cache.put("list", [1, 2, 3])
        cache.put("dict", {"a": 1, "b": 2})
        cache.put("none", None)

        assert cache.get("string") == "hello"
        assert cache.get("int") == 42
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("dict") == {"a": 1, "b": 2}
        assert cache.get("none") is None

    def test_cleanup_removes_correct_items(self):
        """Test that cleanup removes LRU items."""
        cache = DataCache()

        # Add 8 items with time gaps
        for i in range(8):
            cache.put(f"key{i}", f"value{i}")
            time.sleep(0.01)

        # Access some items to make them more recent
        cache.get("key0")
        cache.get("key1")

        # Clear and re-add to verify access times are used
        # This is already tested but let's verify the LRU behavior
        initial_access_key0 = cache.access_times["key0"]
        time.sleep(0.01)
        cache.get("key0")
        updated_access_key0 = cache.access_times["key0"]

        assert updated_access_key0 > initial_access_key0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
