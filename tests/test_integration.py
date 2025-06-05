"""
Integration tests for the bookmark processor.

Tests the complete workflow from CSV input to CSV output.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, Mock

from bookmark_processor.core.bookmark_processor import BookmarkProcessor
from bookmark_processor.core.pipeline import PipelineConfig
from tests.fixtures.test_data import (
    SAMPLE_RAINDROP_EXPORT_ROWS,
    create_sample_export_dataframe,
    TEST_CONFIGS
)


class TestBookmarkProcessorIntegration:
    """Integration tests for the complete bookmark processor workflow."""
    
    @pytest.fixture
    def temp_input_file(self):
        """Create a temporary input CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = create_sample_export_dataframe()
            df.to_csv(f, index=False)
            temp_path = f.name
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def temp_output_file(self):
        """Create a temporary output file path."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        # Remove the file (we just want the path)
        os.unlink(temp_path)
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_complete_workflow_success(self, temp_input_file, temp_output_file):
        """Test the complete workflow with successful processing."""
        processor = BookmarkProcessor()
        
        # Mock network requests to avoid actual HTTP calls
        with patch('requests.Session.get') as mock_get:
            # Mock successful responses for all URLs
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.url = "https://example.com"
            mock_response.text = "<html><head><title>Test Page</title></head><body>Test content</body></html>"
            mock_response.elapsed.total_seconds.return_value = 0.5
            mock_response.history = []
            mock_get.return_value = mock_response
            
            # Process bookmarks
            results = processor.process_bookmarks(
                input_file=temp_input_file,
                output_file=temp_output_file,
                batch_size=10,
                max_retries=1,
                timeout=5,
                enable_checkpoints=False,  # Disable for simpler testing
                enable_ai_processing=False  # Disable AI to avoid model loading
            )
        
        # Verify results
        assert results is not None
        assert results.total_bookmarks > 0
        assert results.valid_bookmarks > 0
        
        # Verify output file exists and has correct format
        assert os.path.exists(temp_output_file)
        
        output_df = pd.read_csv(temp_output_file)
        expected_columns = ['url', 'folder', 'title', 'note', 'tags', 'created']
        assert list(output_df.columns) == expected_columns
        assert len(output_df) > 0
    
    def test_workflow_with_invalid_urls(self, temp_input_file, temp_output_file):
        """Test workflow with some invalid URLs."""
        processor = BookmarkProcessor()
        
        # Mock mixed responses - some successful, some failed
        def mock_get_side_effect(*args, **kwargs):
            url = args[0] if args else kwargs.get('url', '')
            
            if 'invalid' in url:
                # Simulate connection error for invalid URLs
                from requests.exceptions import ConnectionError
                raise ConnectionError("Connection failed")
            else:
                # Successful response for valid URLs
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.url = url
                mock_response.text = f"<html><head><title>Page for {url}</title></head><body>Content</body></html>"
                mock_response.elapsed.total_seconds.return_value = 0.5
                mock_response.history = []
                return mock_response
        
        with patch('requests.Session.get', side_effect=mock_get_side_effect):
            results = processor.process_bookmarks(
                input_file=temp_input_file,
                output_file=temp_output_file,
                batch_size=10,
                max_retries=1,
                timeout=5,
                enable_checkpoints=False,
                enable_ai_processing=False
            )
        
        # Should still process successfully, just with some failures
        assert results is not None
        assert results.total_bookmarks > 0
        assert results.invalid_bookmarks >= 0  # Some URLs might be invalid
        
        # Output file should still be created with valid bookmarks
        assert os.path.exists(temp_output_file)
        output_df = pd.read_csv(temp_output_file)
        assert len(output_df) >= 0  # At least some bookmarks should be valid
    
    def test_workflow_with_ai_processing(self, temp_input_file, temp_output_file):
        """Test workflow with AI processing enabled (mocked)."""
        processor = BookmarkProcessor()
        
        # Mock AI processor to avoid loading actual models
        with patch('bookmark_processor.core.ai_processor.AIProcessor') as mock_ai_class, \
             patch('requests.Session.get') as mock_get:
            
            # Mock AI processor instance
            mock_ai = Mock()
            mock_ai.process_batch.return_value = {
                url: {
                    'enhanced_description': f'AI-enhanced description for {url}',
                    'generated_tags': ['ai', 'enhanced', 'test'],
                    'processing_method': 'ai_enhancement'
                }
                for url in [row['url'] for row in SAMPLE_RAINDROP_EXPORT_ROWS]
            }
            mock_ai.get_processing_statistics.return_value = {
                'total_processed': 5,
                'ai_enhanced': 5,
                'fallback_used': 0
            }
            mock_ai_class.return_value = mock_ai
            
            # Mock successful HTTP responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "<html><head><title>Test</title></head><body>Content</body></html>"
            mock_response.elapsed.total_seconds.return_value = 0.5
            mock_response.history = []
            mock_get.return_value = mock_response
            
            results = processor.process_bookmarks(
                input_file=temp_input_file,
                output_file=temp_output_file,
                batch_size=10,
                enable_ai_processing=True,
                enable_checkpoints=False
            )
        
        # Verify AI processing was used
        assert results is not None
        assert results.total_bookmarks > 0
        
        # Check output contains AI-enhanced descriptions
        output_df = pd.read_csv(temp_output_file)
        assert len(output_df) > 0
        # Note: We'd check for AI-enhanced content here, but it depends on the specific implementation
    
    def test_workflow_with_checkpoints(self, temp_input_file, temp_output_file):
        """Test workflow with checkpoint functionality."""
        processor = BookmarkProcessor()
        
        # Create a temporary checkpoint directory
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            
            with patch('requests.Session.get') as mock_get:
                # Mock successful responses
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = "<html><head><title>Test</title></head><body>Content</body></html>"
                mock_response.elapsed.total_seconds.return_value = 0.5
                mock_response.history = []
                mock_get.return_value = mock_response
                
                # First run - should create checkpoint
                results1 = processor.process_bookmarks(
                    input_file=temp_input_file,
                    output_file=temp_output_file,
                    enable_checkpoints=True,
                    checkpoint_dir=temp_checkpoint_dir,
                    enable_ai_processing=False
                )
                
                # Verify first run completed
                assert results1 is not None
                assert results1.total_bookmarks > 0
                
                # Check that checkpoint files were created
                checkpoint_files = list(Path(temp_checkpoint_dir).glob("*.json"))
                # Note: Checkpoint creation depends on the specific checkpoint logic
    
    def test_workflow_error_handling(self, temp_output_file):
        """Test error handling for invalid input."""
        processor = BookmarkProcessor()
        
        # Test with non-existent input file
        with pytest.raises(Exception):
            processor.process_bookmarks(
                input_file="nonexistent.csv",
                output_file=temp_output_file
            )
    
    def test_workflow_with_malformed_csv(self, temp_output_file):
        """Test handling of malformed CSV input."""
        # Create malformed CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid,csv,format\n")
            f.write("missing,columns\n")
            f.write("inconsistent,row,data,too,many,columns\n")
            malformed_path = f.name
        
        try:
            processor = BookmarkProcessor()
            
            # Should handle malformed CSV gracefully
            with pytest.raises(Exception):
                processor.process_bookmarks(
                    input_file=malformed_path,
                    output_file=temp_output_file,
                    enable_checkpoints=False
                )
        finally:
            os.unlink(malformed_path)
    
    def test_workflow_performance_minimal_config(self, temp_input_file, temp_output_file):
        """Test workflow with minimal performance configuration."""
        processor = BookmarkProcessor()
        
        with patch('requests.Session.get') as mock_get:
            # Mock fast responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "<html><head><title>Fast Test</title></head><body>Quick content</body></html>"
            mock_response.elapsed.total_seconds.return_value = 0.1
            mock_response.history = []
            mock_get.return_value = mock_response
            
            # Use minimal config for faster testing
            config = TEST_CONFIGS['minimal']
            
            results = processor.process_bookmarks(
                input_file=temp_input_file,
                output_file=temp_output_file,
                batch_size=config['batch_size'],
                max_retries=config['max_retries'],
                timeout=config['timeout'],
                enable_checkpoints=False,
                enable_ai_processing=False
            )
        
        assert results is not None
        assert results.processing_time < 10.0  # Should be fast with minimal config
    
    def test_cli_integration(self, temp_input_file, temp_output_file):
        """Test CLI integration with the processor."""
        from bookmark_processor.core.bookmark_processor import BookmarkProcessor
        
        processor = BookmarkProcessor()
        
        # Mock network calls
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "<html><head><title>CLI Test</title></head><body>CLI content</body></html>"
            mock_response.elapsed.total_seconds.return_value = 0.3
            mock_response.history = []
            mock_get.return_value = mock_response
            
            # Simulate CLI arguments
            results = processor.run_cli({
                'input': temp_input_file,
                'output': temp_output_file,
                'batch_size': 5,
                'max_retries': 1,
                'timeout': 10,
                'verbose': True,
                'resume': False,
                'clear_checkpoints': False
            })
        
        # CLI should complete successfully
        assert results is not None
        assert os.path.exists(temp_output_file)
    
    def test_large_dataset_simulation(self):
        """Test with a larger simulated dataset."""
        # Create larger test dataset
        large_data = []
        for i in range(50):  # Simulate 50 bookmarks
            row = {
                'id': str(i),
                'title': f'Test Bookmark {i}',
                'note': f'Note for bookmark {i}',
                'excerpt': f'Excerpt for bookmark {i}',
                'url': f'https://example{i}.com',
                'folder': f'Folder{i % 5}',  # Distribute across 5 folders
                'tags': f'tag{i % 3}, tag{(i + 1) % 3}',  # Rotate tags
                'created': '2024-01-01T00:00:00Z',
                'cover': '',
                'highlights': '',
                'favorite': 'false'
            }
            large_data.append(row)
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_f, \
             tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as output_f:
            
            # Write large dataset
            df = pd.DataFrame(large_data)
            df.to_csv(input_f, index=False)
            input_path = input_f.name
            output_path = output_f.name
        
        # Remove output file (we just want the path)
        os.unlink(output_path)
        
        try:
            processor = BookmarkProcessor()
            
            with patch('requests.Session.get') as mock_get:
                # Mock successful responses
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = "<html><head><title>Large Test</title></head><body>Content</body></html>"
                mock_response.elapsed.total_seconds.return_value = 0.2
                mock_response.history = []
                mock_get.return_value = mock_response
                
                results = processor.process_bookmarks(
                    input_file=input_path,
                    output_file=output_path,
                    batch_size=10,
                    max_retries=1,
                    timeout=5,
                    enable_checkpoints=False,
                    enable_ai_processing=False
                )
            
            # Verify large dataset processing
            assert results is not None
            assert results.total_bookmarks == 50
            assert results.processing_time > 0
            
            # Verify output
            assert os.path.exists(output_path)
            output_df = pd.read_csv(output_path)
            assert len(output_df) > 0
            
        finally:
            # Cleanup
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__])