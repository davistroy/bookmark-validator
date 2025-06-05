"""
Memory Optimization Module

Provides memory-efficient processing techniques for large bookmark datasets.
Includes batch processing, memory monitoring, and garbage collection strategies.
"""

import gc
import sys
import resource
import threading
import time
from typing import List, Dict, Any, Optional, Iterator, Callable, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
import weakref

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    current_mb: float
    peak_mb: float
    available_mb: float
    gc_collections: Dict[str, int]
    timestamp: datetime


class MemoryMonitor:
    """Monitor and track memory usage"""
    
    def __init__(self, warning_threshold_mb: float = 3000, critical_threshold_mb: float = 3500):
        """
        Initialize memory monitor
        
        Args:
            warning_threshold_mb: Memory usage warning threshold in MB
            critical_threshold_mb: Memory usage critical threshold in MB
        """
        self.warning_threshold = warning_threshold_mb
        self.critical_threshold = critical_threshold_mb
        self.peak_memory = 0.0
        self.history: List[MemoryStats] = []
        self.lock = threading.RLock()
        
    def get_current_memory(self) -> float:
        """Get current memory usage in MB"""
        try:
            # Try to get RSS memory usage
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # On Linux, ru_maxrss is in KB
            memory_mb = usage.ru_maxrss / 1024
            
            # On some systems, it might be in bytes
            if memory_mb > 100000:  # Likely in bytes
                memory_mb = memory_mb / 1024 / 1024
                
            return memory_mb
        except:
            # Fallback: estimate from sys.getsizeof for major objects
            return 0.0
    
    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory statistics"""
        with self.lock:
            current_memory = self.get_current_memory()
            self.peak_memory = max(self.peak_memory, current_memory)
            
            # GC statistics
            gc_stats = {f"gen_{i}": gc.get_count()[i] for i in range(3)}
            
            # Available memory estimate (assume 4GB limit)
            available_memory = max(0, 4000 - current_memory)
            
            stats = MemoryStats(
                current_mb=current_memory,
                peak_mb=self.peak_memory,
                available_mb=available_memory,
                gc_collections=gc_stats,
                timestamp=datetime.now()
            )
            
            self.history.append(stats)
            
            # Keep only recent history (last 100 entries)
            if len(self.history) > 100:
                self.history = self.history[-100:]
            
            return stats
    
    def check_memory_pressure(self) -> str:
        """Check current memory pressure level"""
        current_memory = self.get_current_memory()
        
        if current_memory >= self.critical_threshold:
            return "critical"
        elif current_memory >= self.warning_threshold:
            return "warning"
        else:
            return "normal"
    
    def force_cleanup(self):
        """Force garbage collection and cleanup"""
        # Collect all generations
        collected = [gc.collect(i) for i in range(3)]
        
        # Force cleanup of weak references
        gc.collect()
        
        return sum(collected)


class BatchProcessor(Generic[T, R]):
    """Memory-efficient batch processor for large datasets"""
    
    def __init__(self, 
                 batch_size: int = 100,
                 memory_monitor: Optional[MemoryMonitor] = None,
                 enable_gc: bool = True):
        """
        Initialize batch processor
        
        Args:
            batch_size: Number of items to process in each batch
            memory_monitor: Optional memory monitor instance
            enable_gc: Whether to enable automatic garbage collection
        """
        self.batch_size = batch_size
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.enable_gc = enable_gc
        self.processed_count = 0
        self.total_items = 0
        
    def process_batches(self,
                       items: List[T],
                       processor_func: Callable[[List[T]], List[R]],
                       progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Iterator[List[R]]:
        """
        Process items in memory-efficient batches
        
        Args:
            items: List of items to process
            processor_func: Function to process each batch
            progress_callback: Optional progress callback (processed, total, status)
            
        Yields:
            Lists of processed results for each batch
        """
        self.total_items = len(items)
        self.processed_count = 0
        
        # Process in batches
        for batch_start in range(0, self.total_items, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.total_items)
            batch = items[batch_start:batch_end]
            
            # Check memory before processing
            memory_pressure = self.memory_monitor.check_memory_pressure()
            if memory_pressure == "critical":
                # Force cleanup before proceeding
                cleaned = self.memory_monitor.force_cleanup()
                if progress_callback:
                    progress_callback(self.processed_count, self.total_items, 
                                    f"Memory cleanup: freed {cleaned} objects")
            
            # Process batch
            try:
                if progress_callback:
                    progress_callback(self.processed_count, self.total_items,
                                    f"Processing batch {batch_start//self.batch_size + 1}")
                
                batch_results = processor_func(batch)
                yield batch_results
                
                self.processed_count += len(batch)
                
                # Cleanup after batch if enabled
                if self.enable_gc:
                    if memory_pressure in ["warning", "critical"]:
                        gc.collect()
                
                # Progress update
                if progress_callback:
                    memory_stats = self.memory_monitor.get_memory_stats()
                    progress_callback(self.processed_count, self.total_items,
                                    f"Completed batch: {memory_stats.current_mb:.1f}MB used")
                
            except Exception as e:
                if progress_callback:
                    progress_callback(self.processed_count, self.total_items, f"Batch error: {e}")
                raise
            
            # Brief pause to allow other operations
            time.sleep(0.001)
    
    def process_all(self,
                   items: List[T],
                   processor_func: Callable[[List[T]], List[R]],
                   progress_callback: Optional[Callable[[int, int, str], None]] = None) -> List[R]:
        """
        Process all items and return combined results
        
        Args:
            items: List of items to process
            processor_func: Function to process each batch
            progress_callback: Optional progress callback
            
        Returns:
            Combined list of all processed results
        """
        all_results = []
        
        for batch_results in self.process_batches(items, processor_func, progress_callback):
            all_results.extend(batch_results)
        
        return all_results


class StreamingProcessor:
    """Memory-efficient streaming processor for very large datasets"""
    
    def __init__(self, memory_monitor: Optional[MemoryMonitor] = None):
        """Initialize streaming processor"""
        self.memory_monitor = memory_monitor or MemoryMonitor()
    
    @contextmanager
    def stream_items(self, items: List[Any], chunk_size: int = 50):
        """
        Stream items in chunks to minimize memory usage
        
        Args:
            items: Items to stream
            chunk_size: Size of each chunk
            
        Yields:
            Chunks of items
        """
        try:
            for i in range(0, len(items), chunk_size):
                chunk = items[i:i + chunk_size]
                yield chunk
                
                # Memory pressure check
                if self.memory_monitor.check_memory_pressure() == "critical":
                    gc.collect()
                    
        finally:
            # Final cleanup
            gc.collect()


class DataCache:
    """Memory-aware data cache with automatic cleanup"""
    
    def __init__(self, max_size_mb: float = 500):
        """
        Initialize data cache
        
        Args:
            max_size_mb: Maximum cache size in MB
        """
        self.max_size_mb = max_size_mb
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = datetime.now()
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache, return True if successful"""
        with self.lock:
            # Estimate memory usage
            estimated_size = sys.getsizeof(value) / 1024 / 1024  # MB
            
            # Check if we need to cleanup
            if self._get_cache_size() + estimated_size > self.max_size_mb:
                if not self._cleanup_cache():
                    return False  # Cache full, couldn't cleanup enough
            
            self.cache[key] = value
            self.access_times[key] = datetime.now()
            return True
    
    def _get_cache_size(self) -> float:
        """Estimate current cache size in MB"""
        return sum(sys.getsizeof(v) for v in self.cache.values()) / 1024 / 1024
    
    def _cleanup_cache(self) -> bool:
        """Remove least recently used items"""
        if not self.cache:
            return True
        
        # Sort by access time
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        
        # Remove oldest 25% of items
        items_to_remove = len(sorted_items) // 4
        if items_to_remove == 0:
            items_to_remove = 1
        
        for key, _ in sorted_items[:items_to_remove]:
            del self.cache[key]
            del self.access_times[key]
        
        return True
    
    def clear(self):
        """Clear entire cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()


# Global instances
memory_monitor = MemoryMonitor()
data_cache = DataCache()


@contextmanager
def memory_context(operation_name: str = "operation"):
    """Context manager for memory monitoring"""
    start_stats = memory_monitor.get_memory_stats()
    
    try:
        yield memory_monitor
    finally:
        end_stats = memory_monitor.get_memory_stats()
        memory_delta = end_stats.current_mb - start_stats.current_mb
        
        print(f"Memory usage for {operation_name}: "
              f"{memory_delta:+.1f}MB (now: {end_stats.current_mb:.1f}MB)")


def optimize_for_large_dataset(func):
    """Decorator to add memory optimization to functions"""
    def wrapper(*args, **kwargs):
        with memory_context(func.__name__):
            # Force GC before operation
            gc.collect()
            
            result = func(*args, **kwargs)
            
            # Check memory pressure after operation
            if memory_monitor.check_memory_pressure() != "normal":
                memory_monitor.force_cleanup()
            
            return result
    
    return wrapper