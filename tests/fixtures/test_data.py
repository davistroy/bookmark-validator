"""
Test data fixtures for bookmark processor tests.

Contains sample data that mimics real raindrop.io export/import formats.
"""

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from bookmark_processor.core.data_models import (
    Bookmark,
    BookmarkMetadata,
    ProcessingStatus,
)

# Sample raindrop.io export data (11 columns)
SAMPLE_RAINDROP_EXPORT_ROWS = [
    {
        "id": "1",
        "title": "Python Documentation",
        "note": "Official Python documentation",
        "excerpt": "Welcome to Python.org, the official documentation for Python programming language.",
        "url": "https://docs.python.org/3/",
        "folder": "Programming/Python",
        "tags": "python, documentation, programming",
        "created": "2024-01-01T00:00:00Z",
        "cover": "",
        "highlights": "",
        "favorite": "false",
    },
    {
        "id": "2",
        "title": "GitHub - Microsoft/vscode",
        "note": "",
        "excerpt": "Visual Studio Code - Open Source ('Code - OSS')",
        "url": "https://github.com/microsoft/vscode",
        "folder": "Development/Tools",
        "tags": "vscode, editor, microsoft",
        "created": "2024-01-02T12:30:00Z",
        "cover": "",
        "highlights": "",
        "favorite": "true",
    },
    {
        "id": "3",
        "title": "Stack Overflow",
        "note": "Programming Q&A community",
        "excerpt": "Stack Overflow is the largest, most trusted online community for developers to learn.",
        "url": "https://stackoverflow.com/",
        "folder": "Programming/Resources",
        "tags": "programming, community, qa",
        "created": "2024-01-03T08:15:00Z",
        "cover": "",
        "highlights": "",
        "favorite": "false",
    },
    {
        "id": "4",
        "title": "Invalid URL Example",
        "note": "This bookmark has an invalid URL",
        "excerpt": "",
        "url": "not-a-valid-url",
        "folder": "Test",
        "tags": "test, invalid",
        "created": "2024-01-04T16:45:00Z",
        "cover": "",
        "highlights": "",
        "favorite": "false",
    },
    {
        "id": "5",
        "title": "Example with No Tags",
        "note": "Bookmark without any tags",
        "excerpt": "A simple example domain",
        "url": "https://example.com",
        "folder": "Examples",
        "tags": "",
        "created": "2024-01-05T20:00:00Z",
        "cover": "",
        "highlights": "",
        "favorite": "false",
    },
]

# Expected raindrop.io import data (6 columns)
EXPECTED_RAINDROP_IMPORT_ROWS = [
    {
        "url": "https://docs.python.org/3/",
        "folder": "Programming/Python",
        "title": "Python Documentation",
        "note": "Official Python documentation",
        "tags": "python, documentation, programming",
        "created": "2024-01-01T00:00:00+00:00",
    },
    {
        "url": "https://github.com/microsoft/vscode",
        "folder": "Development/Tools",
        "title": "GitHub - Microsoft/vscode",
        "note": "Visual Studio Code - Open Source ('Code - OSS')",
        "tags": "vscode, editor, microsoft",
        "created": "2024-01-02T12:30:00+00:00",
    },
    {
        "url": "https://stackoverflow.com/",
        "folder": "Programming/Resources",
        "title": "Stack Overflow",
        "note": "Programming Q&A community",
        "tags": "programming, community, qa",
        "created": "2024-01-03T08:15:00+00:00",
    },
    {
        "url": "https://example.com",
        "folder": "Examples",
        "title": "Example with No Tags",
        "note": "Bookmark without any tags",
        "tags": "",
        "created": "2024-01-05T20:00:00+00:00",
    },
]

# Mock content data for testing
MOCK_CONTENT_DATA = {
    "https://docs.python.org/3/": {
        "title": "Python Documentation",
        "description": "Official documentation for the Python programming language",
        "content_text": "Python is a programming language that lets you work quickly and integrate systems more effectively.",
        "meta_keywords": ["python", "programming", "documentation"],
        "content_categories": ["programming", "documentation", "tutorial"],
        "status_code": 200,
        "final_url": "https://docs.python.org/3/",
        "content_type": "text/html",
    },
    "https://github.com/microsoft/vscode": {
        "title": "GitHub - microsoft/vscode: Visual Studio Code",
        "description": "Visual Studio Code. Contribute to microsoft/vscode development by creating an account on GitHub.",
        "content_text": "Visual Studio Code is a lightweight but powerful source code editor",
        "meta_keywords": ["vscode", "editor", "microsoft", "github"],
        "content_categories": ["development", "tools", "editor"],
        "status_code": 200,
        "final_url": "https://github.com/microsoft/vscode",
        "content_type": "text/html",
    },
    "https://stackoverflow.com/": {
        "title": "Stack Overflow - Where Developers Learn, Share, & Build Careers",
        "description": "Stack Overflow is the largest, most trusted online community for developers",
        "content_text": "Stack Overflow is a question and answer website for professional and enthusiast programmers",
        "meta_keywords": ["programming", "qa", "community", "developers"],
        "content_categories": ["programming", "community", "questions"],
        "status_code": 200,
        "final_url": "https://stackoverflow.com/",
        "content_type": "text/html",
    },
    "https://example.com": {
        "title": "Example Domain",
        "description": "Example Domain",
        "content_text": "This domain is for use in illustrative examples in documents.",
        "meta_keywords": ["example", "domain"],
        "content_categories": ["example", "test"],
        "status_code": 200,
        "final_url": "https://example.com",
        "content_type": "text/html",
    },
}

# Mock AI processing results
MOCK_AI_RESULTS = {
    "https://docs.python.org/3/": {
        "enhanced_description": "Comprehensive official documentation for Python 3, covering all language features, standard library modules, and best practices for Python development.",
        "generated_tags": [
            "python3",
            "documentation",
            "programming",
            "reference",
            "tutorial",
        ],
        "processing_method": "ai_enhancement",
    },
    "https://github.com/microsoft/vscode": {
        "enhanced_description": "Open-source code editor by Microsoft with excellent Python support, debugging capabilities, and extensive extension ecosystem.",
        "generated_tags": [
            "vscode",
            "editor",
            "microsoft",
            "opensource",
            "development",
        ],
        "processing_method": "ai_enhancement",
    },
    "https://stackoverflow.com/": {
        "enhanced_description": "Premier Q&A community for programmers and developers to ask questions, share knowledge, and solve coding problems collaboratively.",
        "generated_tags": ["stackoverflow", "programming", "qa", "community", "help"],
        "processing_method": "ai_enhancement",
    },
    "https://example.com": {
        "enhanced_description": "Standard example domain used for documentation and testing purposes in web development and networking examples.",
        "generated_tags": ["example", "domain", "testing", "documentation"],
        "processing_method": "fallback",
    },
}


def create_sample_bookmark_objects() -> List[Bookmark]:
    """Create sample Bookmark objects for testing."""
    bookmarks = []

    for row in SAMPLE_RAINDROP_EXPORT_ROWS:
        bookmark = Bookmark.from_raindrop_export(row)
        bookmarks.append(bookmark)

    return bookmarks


def create_sample_export_dataframe() -> pd.DataFrame:
    """Create sample raindrop.io export DataFrame."""
    return pd.DataFrame(SAMPLE_RAINDROP_EXPORT_ROWS)


def create_expected_import_dataframe() -> pd.DataFrame:
    """Create expected raindrop.io import DataFrame."""
    return pd.DataFrame(EXPECTED_RAINDROP_IMPORT_ROWS)


def create_sample_processed_bookmark() -> Bookmark:
    """Create a fully processed bookmark for testing."""
    bookmark = Bookmark.from_raindrop_export(SAMPLE_RAINDROP_EXPORT_ROWS[0])

    # Add processed data
    bookmark.enhanced_description = MOCK_AI_RESULTS["https://docs.python.org/3/"][
        "enhanced_description"
    ]
    bookmark.optimized_tags = MOCK_AI_RESULTS["https://docs.python.org/3/"][
        "generated_tags"
    ]

    # Add metadata
    content_data = MOCK_CONTENT_DATA["https://docs.python.org/3/"]
    bookmark.extracted_metadata = BookmarkMetadata(
        title=content_data["title"],
        description=content_data["description"],
        keywords=content_data["meta_keywords"],
    )

    # Update processing status
    bookmark.processing_status = ProcessingStatus(
        url_validated=True,
        content_extracted=True,
        ai_processed=True,
        tags_optimized=True,
    )

    return bookmark


def create_invalid_csv_content() -> str:
    """Create invalid CSV content for testing error handling."""
    return """id,title,note,excerpt,url
1,"Missing columns",,,"https://example.com"
2,"Invalid structure"
"No quotes","Missing commas"
"""


def create_malformed_bookmark_data() -> List[Dict[str, Any]]:
    """Create malformed bookmark data for testing validation."""
    return [
        {
            "id": "",
            "title": "",
            "note": "",
            "excerpt": "",
            "url": "",
            "folder": "",
            "tags": "",
            "created": "",
            "cover": "",
            "highlights": "",
            "favorite": "",
        },
        {
            "id": "test",
            "title": "No URL",
            "note": "This bookmark has no URL",
            "excerpt": "",
            "url": "",
            "folder": "Test",
            "tags": "test",
            "created": "2024-01-01T00:00:00Z",
            "cover": "",
            "highlights": "",
            "favorite": "false",
        },
        {
            "id": "test2",
            "title": "",
            "note": "",
            "excerpt": "",
            "url": "https://example.com",
            "folder": "",
            "tags": "",
            "created": "invalid-date",
            "cover": "",
            "highlights": "",
            "favorite": "maybe",
        },
    ]


# Test configuration for different scenarios
TEST_CONFIGS = {
    "minimal": {"batch_size": 2, "max_retries": 1, "timeout": 5, "verbose": False},
    "standard": {"batch_size": 10, "max_retries": 3, "timeout": 30, "verbose": True},
    "performance": {
        "batch_size": 100,
        "max_retries": 5,
        "timeout": 60,
        "verbose": False,
    },
}
