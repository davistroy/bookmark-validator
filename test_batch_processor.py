#!/usr/bin/env python3
"""
Quick test script for the enhanced batch processor implementation.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bookmark_processor.core.url_validator import URLValidator, BatchConfig, EnhancedBatchProcessor

def test_batch_processor():
    """Test the enhanced batch processor functionality"""
    
    # Enable test mode to avoid actual network requests
    os.environ["BOOKMARK_PROCESSOR_TEST_MODE"] = "true"
    
    try:
        # Create URL validator
        validator = URLValidator(
            timeout=30.0,
            max_concurrent=5,
            user_agent_rotation=True,
            verify_ssl=True
        )
        
        print("✅ URLValidator created successfully")
        
        # Test interface methods
        optimal_size = validator.get_optimal_batch_size()
        print(f"✅ Optimal batch size: {optimal_size}")
        
        # Test time estimation
        estimated_time = validator.estimate_processing_time(10)
        print(f"✅ Estimated time for 10 URLs: {estimated_time:.2f}s")
        
        # Test batch processing with sample URLs
        test_urls = [
            "https://example.com",
            "https://github.com", 
            "https://stackoverflow.com",
            "https://docs.python.org",
            "https://invalid-test-url.example"
        ]
        
        # Test direct batch processing
        batch_result = validator.process_batch(test_urls, "test_batch_001")
        print(f"✅ Direct batch processing: {batch_result.items_successful}/{batch_result.items_processed} successful")
        
        # Test enhanced batch processor
        config = BatchConfig(
            min_batch_size=2,
            max_batch_size=10,
            optimal_batch_size=5,
            auto_tune_batch_size=True,
            max_concurrent_batches=2
        )
        
        enhanced_processor = validator.create_enhanced_batch_processor(
            config=config,
            progress_callback=lambda msg: print(f"📊 Progress: {msg}")
        )
        
        print("✅ Enhanced batch processor created successfully")
        
        # Add items and process
        enhanced_processor.add_items(test_urls)
        results = enhanced_processor.process_all()
        
        print(f"✅ Enhanced batch processing: {len(results)} results returned")
        
        # Get statistics
        stats = enhanced_processor.get_processing_statistics()
        print(f"✅ Processing statistics: {stats}")
        
        # Test with larger dataset
        large_test_urls = [f"https://test-site-{i}.example.com" for i in range(25)]
        enhanced_processor.reset()
        enhanced_processor.add_items(large_test_urls)
        large_results = enhanced_processor.process_all()
        
        print(f"✅ Large batch processing: {len(large_results)} results from {len(large_test_urls)} URLs")
        
        final_stats = enhanced_processor.get_processing_statistics()
        print(f"✅ Final statistics: {final_stats}")
        
        print("\n🎉 All batch processor tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test mode
        if "BOOKMARK_PROCESSOR_TEST_MODE" in os.environ:
            del os.environ["BOOKMARK_PROCESSOR_TEST_MODE"]

if __name__ == "__main__":
    success = test_batch_processor()
    sys.exit(0 if success else 1)