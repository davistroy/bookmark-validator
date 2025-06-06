"""
Test Fixtures for Integration Testing

Provides data generation, mock service management, and comprehensive
test fixtures for integration testing scenarios.
"""

import json
import random
import string
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator, Callable
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import logging

from bookmark_processor.core.data_models import Bookmark, ProcessingResults, BookmarkMetadata


class DataGenerator:
    """Generates test data for various scenarios."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.logger = logging.getLogger("data_generator")
    
    def generate_bookmark_data(
        self,
        count: int,
        include_invalid: bool = False,
        invalid_ratio: float = 0.1
    ) -> List[Dict[str, str]]:
        """Generate bookmark test data."""
        
        bookmarks = []
        invalid_count = int(count * invalid_ratio) if include_invalid else 0
        valid_count = count - invalid_count
        
        # Generate valid bookmarks
        for i in range(valid_count):
            bookmark = self._generate_valid_bookmark(i + 1)
            bookmarks.append(bookmark)
        
        # Generate invalid bookmarks
        for i in range(invalid_count):
            bookmark = self._generate_invalid_bookmark(valid_count + i + 1)
            bookmarks.append(bookmark)
        
        # Shuffle to mix valid and invalid
        random.shuffle(bookmarks)
        
        return bookmarks
    
    def _generate_valid_bookmark(self, id_num: int) -> Dict[str, str]:
        """Generate a valid bookmark."""
        domains = [
            "example.com", "httpbin.org", "github.com", "stackoverflow.com",
            "docs.python.org", "mozilla.org", "w3.org", "ietf.org"
        ]
        
        folders = [
            "Technology", "Programming", "Research", "Documentation",
            "Tech/AI", "Tech/Web", "Resources/Tools", "Learning"
        ]
        
        tags_pool = [
            "programming", "python", "web", "api", "documentation",
            "tutorial", "reference", "tool", "example", "test"
        ]
        
        domain = random.choice(domains)
        folder = random.choice(folders)
        tags = random.sample(tags_pool, random.randint(1, 3))
        
        # Generate realistic date
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        random_days = random.randint(0, 365)
        created_date = base_date + timedelta(days=random_days)
        
        return {
            'id': str(id_num),
            'title': f'Test Resource {id_num}',
            'note': f'User note for resource {id_num}',
            'excerpt': f'Auto-extracted excerpt for resource {id_num}',
            'url': f'https://{domain}/resource{id_num}',
            'folder': folder,
            'tags': ', '.join(tags),
            'created': created_date.isoformat(),
            'cover': '',
            'highlights': '',
            'favorite': str(random.choice([True, False])).lower()
        }
    
    def _generate_invalid_bookmark(self, id_num: int) -> Dict[str, str]:
        """Generate an invalid bookmark."""
        invalid_patterns = [
            # Invalid URL
            {
                'id': str(id_num),
                'title': f'Invalid URL {id_num}',
                'note': 'Has invalid URL',
                'excerpt': 'Invalid URL test',
                'url': 'not-a-valid-url',
                'folder': 'Invalid',
                'tags': 'invalid, test',
                'created': '2024-01-01T00:00:00Z',
                'cover': '',
                'highlights': '',
                'favorite': 'false'
            },
            # Missing URL
            {
                'id': str(id_num),
                'title': f'Missing URL {id_num}',
                'note': 'Has no URL',
                'excerpt': 'Missing URL test',
                'url': '',
                'folder': 'Invalid',
                'tags': 'invalid, missing',
                'created': '2024-01-01T00:00:00Z',
                'cover': '',
                'highlights': '',
                'favorite': 'false'
            },
            # Non-existent domain
            {
                'id': str(id_num),
                'title': f'Non-existent Domain {id_num}',
                'note': 'Domain does not exist',
                'excerpt': 'Non-existent domain test',
                'url': f'https://definitely-does-not-exist-{random.randint(10000, 99999)}.com',
                'folder': 'Invalid',
                'tags': 'invalid, nonexistent',
                'created': '2024-01-01T00:00:00Z',
                'cover': '',
                'highlights': '',
                'favorite': 'false'
            }
        ]
        
        return random.choice(invalid_patterns)
    
    def generate_large_dataset(
        self,
        size: int,
        variation_seed: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Generate a large dataset with realistic variation."""
        
        if variation_seed is not None:
            random.seed(variation_seed)
        
        # Create categories for realistic distribution
        categories = {
            'Technology': 0.3,
            'Programming': 0.25,
            'Research': 0.15,
            'Documentation': 0.15,
            'Resources': 0.1,
            'Other': 0.05
        }
        
        bookmarks = []
        
        for i in range(size):
            # Select category based on distribution
            category = self._select_by_distribution(categories)
            bookmark = self._generate_bookmark_for_category(i + 1, category)
            bookmarks.append(bookmark)
        
        return bookmarks
    
    def _select_by_distribution(self, distribution: Dict[str, float]) -> str:
        """Select an item based on probability distribution."""
        rand_val = random.random()
        cumulative = 0.0
        
        for item, probability in distribution.items():
            cumulative += probability
            if rand_val <= cumulative:
                return item
        
        return list(distribution.keys())[-1]  # Fallback
    
    def _generate_bookmark_for_category(self, id_num: int, category: str) -> Dict[str, str]:
        """Generate a bookmark tailored to a specific category."""
        
        category_configs = {
            'Technology': {
                'domains': ['techcrunch.com', 'wired.com', 'arstechnica.com'],
                'tags': ['technology', 'innovation', 'gadgets', 'software'],
                'title_prefix': 'Tech News:'
            },
            'Programming': {
                'domains': ['github.com', 'stackoverflow.com', 'docs.python.org'],
                'tags': ['programming', 'code', 'development', 'tutorial'],
                'title_prefix': 'Programming:'
            },
            'Research': {
                'domains': ['arxiv.org', 'scholar.google.com', 'researchgate.net'],
                'tags': ['research', 'academic', 'paper', 'study'],
                'title_prefix': 'Research:'
            },
            'Documentation': {
                'domains': ['docs.microsoft.com', 'developer.mozilla.org', 'w3.org'],
                'tags': ['documentation', 'reference', 'manual', 'guide'],
                'title_prefix': 'Docs:'
            },
            'Resources': {
                'domains': ['awesome-lists.com', 'resources.com', 'tools.com'],
                'tags': ['resources', 'tools', 'collection', 'useful'],
                'title_prefix': 'Resource:'
            },
            'Other': {
                'domains': ['example.com', 'test.com', 'sample.org'],
                'tags': ['misc', 'other', 'general'],
                'title_prefix': 'General:'
            }
        }
        
        config = category_configs.get(category, category_configs['Other'])
        domain = random.choice(config['domains'])
        tags = random.sample(config['tags'], random.randint(1, 3))
        
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        random_days = random.randint(0, 365)
        created_date = base_date + timedelta(days=random_days)
        
        return {
            'id': str(id_num),
            'title': f"{config['title_prefix']} Item {id_num}",
            'note': f'Note for {category.lower()} item {id_num}',
            'excerpt': f'Excerpt for {category.lower()} resource {id_num}',
            'url': f'https://{domain}/item{id_num}',
            'folder': category,
            'tags': ', '.join(tags),
            'created': created_date.isoformat(),
            'cover': '',
            'highlights': '',
            'favorite': str(random.choice([True, False])).lower()
        }
    
    def generate_malformed_csv_data(self) -> str:
        """Generate malformed CSV data for error testing."""
        malformed_examples = [
            # Missing quotes
            'id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite\n'
            '1,Title with, comma,Note,Excerpt,https://example.com,Folder,tag1, tag2,2024-01-01T00:00:00Z,,false\n',
            
            # Inconsistent column count
            'id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite\n'
            '1,Title,Note,Excerpt,https://example.com,Folder\n'  # Missing columns
            '2,Title,Note,Excerpt,https://example.com,Folder,Tags,Date,Cover,Highlights,Favorite,Extra\n',  # Extra column
            
            # Invalid characters
            'id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite\n'
            '1,"Title\nwith\nnewlines","Note","Excerpt","https://example.com","Folder","Tags","2024-01-01T00:00:00Z","","","false"\n',
            
            # Empty file
            '',
            
            # Headers only
            'id,title,note,excerpt,url,folder,tags,created,cover,highlights,favorite\n'
        ]
        
        return random.choice(malformed_examples)


class MockServiceManager:
    """Manages mock services for integration testing."""
    
    def __init__(self):
        self.active_mocks = {}
        self.mock_configs = {}
        self.logger = logging.getLogger("mock_service_manager")
    
    def setup_http_mock(
        self,
        success_rate: float = 0.9,
        response_delay: float = 0.1,
        error_types: List[str] = None
    ) -> Mock:
        """Set up HTTP request mocking."""
        
        if error_types is None:
            error_types = ['connection_error', 'timeout', 'http_error']
        
        def mock_request(*args, **kwargs):
            import time
            import requests
            
            # Simulate network delay
            if response_delay > 0:
                time.sleep(response_delay)
            
            # Determine if this request should succeed
            if random.random() < success_rate:
                # Success response
                response = Mock()
                response.status_code = 200
                response.url = args[0] if args else kwargs.get('url', 'https://example.com')
                response.headers = {'content-type': 'text/html'}
                response.text = self._generate_mock_html_content(response.url)
                response.history = []
                response.elapsed.total_seconds.return_value = response_delay
                return response
            else:
                # Error response
                error_type = random.choice(error_types)
                if error_type == 'connection_error':
                    raise requests.exceptions.ConnectionError("Mock connection error")
                elif error_type == 'timeout':
                    raise requests.exceptions.Timeout("Mock timeout error")
                elif error_type == 'http_error':
                    response = Mock()
                    response.status_code = random.choice([404, 500, 503])
                    response.url = args[0] if args else kwargs.get('url', 'https://example.com')
                    response.raise_for_status.side_effect = requests.exceptions.HTTPError("Mock HTTP error")
                    return response
        
        # Set up the mock
        mock_session = MagicMock()
        mock_session.get.side_effect = mock_request
        mock_session.head.side_effect = mock_request
        
        mock_patcher = patch('requests.Session', return_value=mock_session)
        mock_instance = mock_patcher.start()
        
        self.active_mocks['http'] = mock_patcher
        return mock_instance
    
    def setup_ai_mock(
        self,
        response_quality: str = 'good',
        processing_delay: float = 0.1
    ) -> Mock:
        """Set up AI processing mocking."""
        
        def mock_process_batch(bookmarks):
            import time
            
            # Simulate processing delay
            if processing_delay > 0:
                time.sleep(processing_delay * len(bookmarks))
            
            for bookmark in bookmarks:
                # Generate mock AI results based on quality setting
                if response_quality == 'good':
                    bookmark.enhanced_description = f"High-quality AI description for {bookmark.title}"
                    bookmark.optimized_tags = ["ai", "enhanced", "quality"]
                elif response_quality == 'poor':
                    bookmark.enhanced_description = f"Poor AI description"
                    bookmark.optimized_tags = ["poor", "ai"]
                else:  # random quality
                    quality = random.choice(['good', 'poor'])
                    if quality == 'good':
                        bookmark.enhanced_description = f"Good AI description for {bookmark.title}"
                        bookmark.optimized_tags = ["ai", "good", "random"]
                    else:
                        bookmark.enhanced_description = "Poor AI description"
                        bookmark.optimized_tags = ["poor", "random"]
            
            return bookmarks
        
        mock_ai = Mock()
        mock_ai.process_batch = mock_process_batch
        mock_ai.is_available = True
        mock_ai.model_name = "mock-ai-model"
        
        mock_patcher = patch('bookmark_processor.core.ai_processor.EnhancedAIProcessor', return_value=mock_ai)
        mock_instance = mock_patcher.start()
        
        self.active_mocks['ai'] = mock_patcher
        return mock_instance
    
    def setup_checkpoint_mock(self) -> Mock:
        """Set up checkpoint functionality mocking."""
        
        mock_checkpoint_manager = Mock()
        
        # Mock checkpoint data storage
        checkpoint_data = {}
        
        def mock_save_checkpoint(checkpoint_id, data):
            checkpoint_data[checkpoint_id] = data
            return True
        
        def mock_load_checkpoint(checkpoint_id):
            return checkpoint_data.get(checkpoint_id)
        
        def mock_list_checkpoints():
            return list(checkpoint_data.keys())
        
        mock_checkpoint_manager.save_checkpoint = mock_save_checkpoint
        mock_checkpoint_manager.load_checkpoint = mock_load_checkpoint
        mock_checkpoint_manager.list_checkpoints = mock_list_checkpoints
        mock_checkpoint_manager.has_checkpoint.return_value = len(checkpoint_data) > 0
        
        mock_patcher = patch(
            'bookmark_processor.core.checkpoint_manager.CheckpointManager',
            return_value=mock_checkpoint_manager
        )
        mock_instance = mock_patcher.start()
        
        self.active_mocks['checkpoint'] = mock_patcher
        return mock_instance
    
    def _generate_mock_html_content(self, url: str) -> str:
        """Generate realistic mock HTML content."""
        title = f"Mock Page for {url}"
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta name="description" content="Mock description for {url}">
            <meta name="keywords" content="mock, test, content">
        </head>
        <body>
            <h1>{title}</h1>
            <p>This is mock content for testing purposes.</p>
            <p>URL: {url}</p>
            <p>Generated at: {datetime.now().isoformat()}</p>
        </body>
        </html>
        """
    
    def cleanup_mocks(self) -> None:
        """Clean up all active mocks."""
        for name, mock_patcher in self.active_mocks.items():
            try:
                mock_patcher.stop()
                self.logger.debug(f"Stopped mock: {name}")
            except Exception as e:
                self.logger.warning(f"Failed to stop mock {name}: {e}")
        
        self.active_mocks.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup_mocks()


class IntegrationTestFixtures:
    """Main fixtures class for integration testing."""
    
    def __init__(self, environment):
        self.environment = environment
        self.data_generator = DataGenerator()
        self.mock_manager = MockServiceManager()
        self.logger = logging.getLogger("integration_test_fixtures")
    
    def create_test_dataset(
        self,
        name: str,
        size: int,
        include_invalid: bool = False,
        save_to_file: bool = True
    ) -> Path:
        """Create a test dataset."""
        
        data = self.data_generator.generate_bookmark_data(
            count=size,
            include_invalid=include_invalid
        )
        
        if save_to_file:
            file_path = self.environment.get_directory('input') / f"{name}.csv"
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            return file_path
        
        return data
    
    def create_performance_dataset(
        self,
        name: str,
        size: int
    ) -> Path:
        """Create a performance test dataset."""
        
        data = self.data_generator.generate_large_dataset(size)
        
        file_path = self.environment.get_directory('input') / f"{name}_performance.csv"
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        
        self.logger.info(f"Created performance dataset: {file_path} ({size} items)")
        return file_path
    
    def create_error_test_dataset(self, name: str) -> Path:
        """Create a dataset designed to test error handling."""
        
        # Mix of valid and invalid data
        valid_data = self.data_generator.generate_bookmark_data(count=5, include_invalid=False)
        invalid_data = self.data_generator.generate_bookmark_data(count=3, include_invalid=True, invalid_ratio=1.0)
        
        all_data = valid_data + invalid_data
        random.shuffle(all_data)
        
        file_path = self.environment.get_directory('input') / f"{name}_error_test.csv"
        df = pd.DataFrame(all_data)
        df.to_csv(file_path, index=False)
        
        return file_path
    
    def create_malformed_csv(self, name: str) -> Path:
        """Create a malformed CSV file for testing."""
        
        malformed_content = self.data_generator.generate_malformed_csv_data()
        
        file_path = self.environment.get_directory('input') / f"{name}_malformed.csv"
        file_path.write_text(malformed_content)
        
        return file_path
    
    def setup_standard_mocks(
        self,
        network_success_rate: float = 0.9,
        ai_quality: str = 'good'
    ) -> Dict[str, Mock]:
        """Set up standard mocks for integration testing."""
        
        mocks = {}
        
        # HTTP mocking
        mocks['http'] = self.mock_manager.setup_http_mock(
            success_rate=network_success_rate,
            response_delay=0.1
        )
        
        # AI mocking
        mocks['ai'] = self.mock_manager.setup_ai_mock(
            response_quality=ai_quality,
            processing_delay=0.05
        )
        
        # Checkpoint mocking
        mocks['checkpoint'] = self.mock_manager.setup_checkpoint_mock()
        
        self.logger.info("Standard mocks set up for integration testing")
        return mocks
    
    def create_checkpoint_scenario(
        self,
        checkpoint_id: str,
        processed_count: int,
        total_count: int
    ) -> Path:
        """Create a checkpoint scenario for resume testing."""
        
        checkpoint_data = {
            'checkpoint_id': checkpoint_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'progress': {
                'processed_count': processed_count,
                'total_count': total_count,
                'last_processed_id': str(processed_count),
                'batch_size': 10,
                'current_batch': processed_count // 10
            },
            'metadata': {
                'test_scenario': True,
                'created_by': 'integration_test_fixtures'
            }
        }
        
        checkpoint_file = self.environment.get_directory('checkpoints') / f"{checkpoint_id}.json"
        checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))
        
        return checkpoint_file
    
    def verify_test_output(
        self,
        output_file: Path,
        expected_structure: bool = True,
        min_rows: int = 0
    ) -> Dict[str, Any]:
        """Verify test output and return validation results."""
        
        results = {
            'file_exists': output_file.exists(),
            'structure_valid': False,
            'row_count': 0,
            'columns': [],
            'errors': []
        }
        
        if not results['file_exists']:
            results['errors'].append("Output file does not exist")
            return results
        
        try:
            df = pd.read_csv(output_file)
            results['row_count'] = len(df)
            results['columns'] = list(df.columns)
            
            if expected_structure:
                expected_columns = ['url', 'folder', 'title', 'note', 'tags', 'created']
                if results['columns'] == expected_columns:
                    results['structure_valid'] = True
                else:
                    results['errors'].append(f"Column structure incorrect: {results['columns']}")
            
            if results['row_count'] < min_rows:
                results['errors'].append(f"Row count {results['row_count']} below minimum {min_rows}")
        
        except Exception as e:
            results['errors'].append(f"Failed to read output file: {str(e)}")
        
        return results
    
    def cleanup(self) -> None:
        """Clean up fixtures."""
        self.mock_manager.cleanup_mocks()
        self.logger.info("Integration test fixtures cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()