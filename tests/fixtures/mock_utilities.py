"""
Mock utilities and builders for bookmark processor tests.

This module provides specialized mock objects and utilities for testing
various components of the bookmark processor application.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pandas as pd
import requests

from bookmark_processor.core.data_models import (
    Bookmark,
    BookmarkMetadata,
    ProcessingResults,
    ProcessingStatus,
)


class MockResponse:
    """Mock HTTP response for testing URL validation and content extraction."""

    def __init__(
        self,
        status_code: int = 200,
        url: str = "https://example.com",
        final_url: Optional[str] = None,
        content_type: str = "text/html",
        text: str = "",
        headers: Optional[Dict[str, str]] = None,
        history: Optional[List] = None,
        raise_for_status: bool = False,
    ):
        self.status_code = status_code
        self.url = url
        self.final_url = final_url or url
        self.content_type = content_type
        self.text = text
        self.headers = headers or {"content-type": content_type}
        self.history = history or []
        self._raise_for_status = raise_for_status

        # Common HTML content if none provided
        if not text and content_type == "text/html":
            domain = url.replace("https://", "").replace("http://", "").split("/")[0]
            self.text = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Test Page - {domain}</title>
                <meta name="description" content="Test page for {domain}">
                <meta name="keywords" content="test, mock, example">
            </head>
            <body>
                <h1>Test Page</h1>
                <p>This is a mock page for testing purposes.</p>
                <p>URL: {url}</p>
            </body>
            </html>
            """

    def raise_for_status(self):
        """Raise an exception if status indicates error."""
        if self._raise_for_status and self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code} Error")

    def json(self):
        """Return JSON data (for API responses)."""
        return {"url": self.url, "status": "success", "data": "mock response"}


class MockRequestsSession:
    """Mock requests.Session with configurable responses."""

    def __init__(self):
        self.responses: Dict[str, MockResponse] = {}
        self.request_count = 0
        self.request_history: List[Dict[str, Any]] = []
        self.delay_per_request = 0.0

        # Default responses for common test URLs
        self._setup_default_responses()

    def _setup_default_responses(self):
        """Set up default responses for common test URLs."""
        self.responses.update(
            {
                "https://docs.python.org/3/": MockResponse(
                    url="https://docs.python.org/3/",
                    text="""
                <html>
                <head>
                    <title>Python 3 Documentation</title>
                    <meta name="description" content="Official Python 3 documentation">
                    <meta name="keywords" content="python, programming, documentation">
                </head>
                <body>
                    <h1>Python 3 Documentation</h1>
                    <p>Welcome to the official Python 3 documentation.</p>
                </body>
                </html>
                """,
                ),
                "https://github.com/microsoft/vscode": MockResponse(
                    url="https://github.com/microsoft/vscode",
                    text="""
                <html>
                <head>
                    <title>GitHub - microsoft/vscode: Visual Studio Code</title>
                    <meta name="description" content="Visual Studio Code">
                    <meta name="keywords" content="vscode, editor, microsoft, github">
                </head>
                <body>
                    <h1>Visual Studio Code</h1>
                    <p>Open source code editor by Microsoft.</p>
                </body>
                </html>
                """,
                ),
                "https://stackoverflow.com/": MockResponse(
                    url="https://stackoverflow.com/",
                    text="""
                <html>
                <head>
                    <title>Stack Overflow - Where Developers Learn</title>
                    <meta name="description" content="Stack Overflow Q&A community">
                    <meta name="keywords" content="programming, qa, community, developers">
                </head>
                <body>
                    <h1>Stack Overflow</h1>
                    <p>Where developers learn, share, and build careers.</p>
                </body>
                </html>
                """,
                ),
                "https://example.com": MockResponse(),
            }
        )

    def add_response(self, url: str, response: MockResponse):
        """Add a custom response for a specific URL."""
        self.responses[url] = response

    def add_error_response(
        self, url: str, error_class=requests.ConnectionError, error_message="Mock error"
    ):
        """Add an error response for a specific URL."""

        def raise_error(*args, **kwargs):
            raise error_class(error_message)

        self.responses[url] = raise_error

    def get(self, url: str, **kwargs) -> MockResponse:
        """Mock GET request."""
        return self._make_request("GET", url, **kwargs)

    def head(self, url: str, **kwargs) -> MockResponse:
        """Mock HEAD request."""
        return self._make_request("HEAD", url, **kwargs)

    def post(self, url: str, **kwargs) -> MockResponse:
        """Mock POST request."""
        return self._make_request("POST", url, **kwargs)

    def _make_request(self, method: str, url: str, **kwargs) -> MockResponse:
        """Internal method to handle request logic."""
        self.request_count += 1

        # Record request for testing
        self.request_history.append(
            {
                "method": method,
                "url": url,
                "kwargs": kwargs,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        # Simulate network delay
        if self.delay_per_request > 0:
            time.sleep(self.delay_per_request)

        # Check if we have a specific response for this URL
        if url in self.responses:
            response_or_error = self.responses[url]
            if callable(response_or_error):
                response_or_error()  # Raise error
            return response_or_error

        # Default response for unknown URLs
        return MockResponse(url=url, status_code=404, text="Not Found")

    def set_delay(self, delay: float):
        """Set delay per request for testing rate limiting."""
        self.delay_per_request = delay

    def get_request_count(self, url: Optional[str] = None) -> int:
        """Get total request count or count for specific URL."""
        if url is None:
            return self.request_count
        return len([r for r in self.request_history if r["url"] == url])

    def reset(self):
        """Reset request history and count."""
        self.request_count = 0
        self.request_history.clear()


class MockAIProcessor:
    """Mock AI processor for testing without actual AI models."""

    def __init__(self, success_rate: float = 1.0, processing_delay: float = 0.0):
        self.success_rate = success_rate
        self.processing_delay = processing_delay
        self.model_name = "mock-ai-model"
        self.is_available = True
        self.processed_count = 0
        self.failed_count = 0

        # Predefined AI results for common test URLs
        self.predefined_results = {
            "https://docs.python.org/3/": {
                "enhanced_description": "Comprehensive official documentation for Python 3 programming language",
                "generated_tags": [
                    "python3",
                    "documentation",
                    "programming",
                    "reference",
                ],
            },
            "https://github.com/microsoft/vscode": {
                "enhanced_description": "Open-source code editor by Microsoft with extensible features",
                "generated_tags": [
                    "vscode",
                    "editor",
                    "microsoft",
                    "opensource",
                    "development",
                ],
            },
            "https://stackoverflow.com/": {
                "enhanced_description": "Premier Q&A community for programmers and developers",
                "generated_tags": [
                    "stackoverflow",
                    "programming",
                    "qa",
                    "community",
                    "help",
                ],
            },
        }

    def process_bookmark(self, bookmark: Bookmark) -> Bookmark:
        """Process a single bookmark with AI enhancement."""
        self.processed_count += 1

        # Simulate processing delay
        if self.processing_delay > 0:
            time.sleep(self.processing_delay)

        # Simulate occasional failures
        import random

        if random.random() > self.success_rate:
            self.failed_count += 1
            bookmark.processing_status.ai_processing_error = "Mock AI processing failed"
            return bookmark

        # Use predefined results if available
        if bookmark.url in self.predefined_results:
            result = self.predefined_results[bookmark.url]
            bookmark.enhanced_description = result["enhanced_description"]
            bookmark.optimized_tags = result["generated_tags"]
        else:
            # Generate mock results
            bookmark.enhanced_description = (
                f"AI-enhanced description for {bookmark.get_effective_title()}"
            )
            bookmark.optimized_tags = self._generate_mock_tags(bookmark)

        bookmark.processing_status.ai_processed = True
        return bookmark

    def process_batch(self, bookmarks: List[Bookmark]) -> List[Bookmark]:
        """Process a batch of bookmarks."""
        return [self.process_bookmark(bookmark) for bookmark in bookmarks]

    def _generate_mock_tags(self, bookmark: Bookmark) -> List[str]:
        """Generate mock tags based on bookmark content."""
        mock_tags = ["ai-processed"]

        # Add tags based on URL domain
        if "github.com" in bookmark.url:
            mock_tags.extend(["github", "development", "code"])
        elif "stackoverflow.com" in bookmark.url:
            mock_tags.extend(["programming", "qa", "community"])
        elif "python.org" in bookmark.url:
            mock_tags.extend(["python", "documentation", "programming"])
        else:
            mock_tags.extend(["web", "bookmark", "test"])

        # Add tags based on existing tags
        if bookmark.tags:
            mock_tags.extend(bookmark.tags[:2])  # Include some original tags

        return list(set(mock_tags))  # Remove duplicates

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total = self.processed_count + self.failed_count
        return {
            "total_processed": self.processed_count,
            "total_failed": self.failed_count,
            "success_rate": self.processed_count / total if total > 0 else 0,
            "model_name": self.model_name,
            "is_available": self.is_available,
        }


class MockEnhancedAIProcessor(MockAIProcessor):
    """Mock EnhancedAIProcessor for testing with enhanced AI functionality."""
    
    def __init__(self, success_rate: float = 1.0, processing_delay: float = 0.0, engine: str = "local"):
        super().__init__(success_rate, processing_delay)
        self.engine = engine
        self.stats = {
            "total_processed": 0,
            "ai_generated": 0,
            "fallback_used": 0,
            "errors": 0,
            "processing_times": [],
        }
    
    def process_bookmark(self, bookmark: Bookmark) -> Bookmark:
        """Process bookmark and update stats like the real processor."""
        result = super().process_bookmark(bookmark)
        
        # Update stats to match real processor
        self.stats["total_processed"] += 1
        if result.processing_status.ai_processed:
            self.stats["ai_generated"] += 1
        else:
            if result.processing_status.ai_processing_error:
                self.stats["errors"] += 1
            else:
                self.stats["fallback_used"] += 1
        
        # Add fake processing time
        self.stats["processing_times"].append(0.1)
        
        return result


class MockContentAnalyzer:
    """Mock content analyzer for testing content extraction."""

    def __init__(self, extraction_success_rate: float = 1.0):
        self.extraction_success_rate = extraction_success_rate
        self.extracted_count = 0
        self.failed_count = 0

        # Predefined metadata for common test URLs
        self.predefined_metadata = {
            "https://docs.python.org/3/": BookmarkMetadata(
                title="Python 3 Documentation",
                description="Official documentation for Python 3 programming language",
                keywords=["python", "programming", "documentation", "reference"],
                canonical_url="https://docs.python.org/3/",
            ),
            "https://github.com/microsoft/vscode": BookmarkMetadata(
                title="GitHub - microsoft/vscode: Visual Studio Code",
                description="Visual Studio Code repository on GitHub",
                keywords=["vscode", "editor", "microsoft", "github", "development"],
                canonical_url="https://github.com/microsoft/vscode",
            ),
            "https://stackoverflow.com/": BookmarkMetadata(
                title="Stack Overflow - Where Developers Learn",
                description="Stack Overflow Q&A community for developers",
                keywords=["programming", "qa", "community", "developers"],
                canonical_url="https://stackoverflow.com/",
            ),
        }

    def extract_metadata(
        self, url: str, content: Optional[str] = None
    ) -> Optional[BookmarkMetadata]:
        """Extract metadata from URL content."""
        import random

        # Simulate occasional failures
        if random.random() > self.extraction_success_rate:
            self.failed_count += 1
            return None

        self.extracted_count += 1

        # Use predefined metadata if available
        if url in self.predefined_metadata:
            return self.predefined_metadata[url]

        # Generate mock metadata
        domain = url.replace("https://", "").replace("http://", "").split("/")[0]
        return BookmarkMetadata(
            title=f"Mock Page - {domain}",
            description=f"Mock description for {url}",
            keywords=["mock", "test", domain.split(".")[0]],
            canonical_url=url,
        )

    def analyze_content_categories(self, content: str) -> List[str]:
        """Analyze content and return categories."""
        categories = ["general"]

        content_lower = content.lower()
        if "python" in content_lower:
            categories.append("programming")
        if "github" in content_lower:
            categories.append("development")
        if "stack overflow" in content_lower:
            categories.append("qa")

        return categories

    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        total = self.extracted_count + self.failed_count
        return {
            "total_extracted": self.extracted_count,
            "total_failed": self.failed_count,
            "success_rate": self.extracted_count / total if total > 0 else 0,
        }


class MockCheckpointManager:
    """Mock checkpoint manager for testing checkpoint functionality."""

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.checkpoint_dir = checkpoint_dir or Path("/tmp/mock_checkpoints")
        self.saved_checkpoints: Dict[str, Any] = {}
        self.save_count = 0
        self.load_count = 0

    def save_checkpoint(self, checkpoint_id: str, data: Any) -> bool:
        """Save checkpoint data."""
        self.saved_checkpoints[checkpoint_id] = data
        self.save_count += 1
        return True

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Any]:
        """Load checkpoint data."""
        self.load_count += 1
        return self.saved_checkpoints.get(checkpoint_id)

    def has_checkpoint(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists."""
        return checkpoint_id in self.saved_checkpoints

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        return list(self.saved_checkpoints.keys())

    def clear_checkpoints(self) -> int:
        """Clear all checkpoints."""
        count = len(self.saved_checkpoints)
        self.saved_checkpoints.clear()
        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get checkpoint statistics."""
        return {
            "total_saves": self.save_count,
            "total_loads": self.load_count,
            "active_checkpoints": len(self.saved_checkpoints),
        }


class MockProgressTracker:
    """Mock progress tracker for testing progress reporting."""

    def __init__(self):
        self.progress_updates: List[Dict[str, Any]] = []
        self.current_progress = 0.0
        self.total_items = 0
        self.completed_items = 0

    def start(self, total_items: int, description: str = "Processing"):
        """Start progress tracking."""
        self.total_items = total_items
        self.completed_items = 0
        self.current_progress = 0.0
        self.progress_updates.append(
            {
                "action": "start",
                "total_items": total_items,
                "description": description,
                "timestamp": datetime.now(timezone.utc),
            }
        )

    def update(self, completed: int, message: Optional[str] = None):
        """Update progress."""
        self.completed_items = completed
        self.current_progress = (
            completed / self.total_items if self.total_items > 0 else 0.0
        )

        self.progress_updates.append(
            {
                "action": "update",
                "completed": completed,
                "total": self.total_items,
                "progress": self.current_progress,
                "message": message,
                "timestamp": datetime.now(timezone.utc),
            }
        )

    def finish(self, final_message: Optional[str] = None):
        """Finish progress tracking."""
        self.progress_updates.append(
            {
                "action": "finish",
                "final_message": final_message,
                "timestamp": datetime.now(timezone.utc),
            }
        )

    def get_progress(self) -> float:
        """Get current progress percentage."""
        return self.current_progress

    def get_update_count(self) -> int:
        """Get number of progress updates."""
        return len([u for u in self.progress_updates if u["action"] == "update"])


def create_mock_pipeline_context() -> Dict[str, Any]:
    """Create a complete mock context for pipeline testing."""
    return {
        "requests_session": MockRequestsSession(),
        "ai_processor": MockAIProcessor(),
        "content_analyzer": MockContentAnalyzer(),
        "checkpoint_manager": MockCheckpointManager(),
        "progress_tracker": MockProgressTracker(),
        "rate_limiter": Mock(),
        "config": {"batch_size": 10, "max_retries": 3, "timeout": 30, "verbose": True},
    }


def create_performance_test_data(size: int) -> pd.DataFrame:
    """Create test data for performance testing."""
    import random

    base_urls = [
        "https://docs.python.org/3/",
        "https://github.com/microsoft/vscode",
        "https://stackoverflow.com/",
        "https://example.com",
        "https://www.google.com",
        "https://github.com",
        "https://www.reddit.com",
        "https://news.ycombinator.com",
    ]

    folders = [
        "Programming/Python",
        "Development/Tools",
        "Programming/Resources",
        "Examples",
        "Search",
        "Development",
        "Social",
        "News",
    ]

    data = []
    for i in range(size):
        base_url = random.choice(base_urls)
        url = f"{base_url}?test={i}" if "?" not in base_url else f"{base_url}&test={i}"

        data.append(
            {
                "id": str(i + 1),
                "title": f"Test Bookmark {i + 1}",
                "note": f"Test note for bookmark {i + 1}",
                "excerpt": f"Test excerpt for bookmark {i + 1}",
                "url": url,
                "folder": random.choice(folders),
                "tags": f"test, bookmark{i % 10}, tag{i % 5}",
                "created": f"2024-01-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00Z",
                "cover": "",
                "highlights": "",
                "favorite": "true" if i % 10 == 0 else "false",
            }
        )

    return pd.DataFrame(data)


# Export key classes and functions
__all__ = [
    "MockResponse",
    "MockRequestsSession",
    "MockAIProcessor",
    "MockContentAnalyzer",
    "MockCheckpointManager",
    "MockProgressTracker",
    "create_mock_pipeline_context",
    "create_performance_test_data",
]
