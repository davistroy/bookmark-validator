"""
Data Sources Module for Bookmark Processing.

This module provides abstractions for different bookmark data sources,
enabling the bookmark processor to work with various storage backends
(CSV files, APIs, MCP servers, databases, etc.).

Main Components:
    - BookmarkDataSource: Protocol defining the data source interface
    - CSVDataSource: CSV file implementation using RaindropCSVHandler
    - ProcessingStateTracker: SQLite-based state tracking for incremental updates
    - BulkUpdateResult: Result container for bulk operations
    - MCPClient: Client for MCP server communication (Phase 5)
    - RaindropMCPDataSource: Raindrop.io data source via MCP (Phase 5)

Usage:
    >>> from bookmark_processor.core.data_sources import CSVDataSource
    >>> source = CSVDataSource(Path("export.csv"), Path("import.csv"))
    >>> bookmarks = source.fetch_bookmarks()
    >>> # Process bookmarks...
    >>> source.bulk_update(bookmarks)
    >>> source.save()

For incremental processing:
    >>> from bookmark_processor.core.data_sources import ProcessingStateTracker
    >>> tracker = ProcessingStateTracker()
    >>> unprocessed = tracker.get_unprocessed(bookmarks)

For MCP/Raindrop.io integration:
    >>> from bookmark_processor.core.data_sources import RaindropMCPDataSource
    >>> async with RaindropMCPDataSource(server_url, token) as source:
    ...     bookmarks = await source.fetch_bookmarks({"collection": "Tech"})
"""

from .protocol import (
    AbstractBookmarkDataSource,
    BookmarkDataSource,
    BulkUpdateResult,
    DataSourceConnectionError,
    DataSourceError,
    DataSourceReadError,
    DataSourceValidationError,
    DataSourceWriteError,
)

from .csv_source import CSVDataSource

from .state_tracker import ProcessingStateTracker

from .mcp_client import (
    MCPClient,
    MCPClientError,
    MCPConnectionError,
    MCPTimeoutError,
    MCPToolError,
    MCPAuthenticationError,
)

from .raindrop_mcp import RaindropMCPDataSource


__all__ = [
    # Protocol and base classes
    "BookmarkDataSource",
    "AbstractBookmarkDataSource",
    "BulkUpdateResult",
    # Exceptions - Data Source
    "DataSourceError",
    "DataSourceConnectionError",
    "DataSourceReadError",
    "DataSourceWriteError",
    "DataSourceValidationError",
    # Exceptions - MCP
    "MCPClientError",
    "MCPConnectionError",
    "MCPTimeoutError",
    "MCPToolError",
    "MCPAuthenticationError",
    # Implementations
    "CSVDataSource",
    "ProcessingStateTracker",
    # MCP Integration (Phase 5)
    "MCPClient",
    "RaindropMCPDataSource",
]
