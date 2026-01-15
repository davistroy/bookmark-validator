"""
Unit tests for the Raindrop.io MCP Data Source.

Tests the RaindropMCPDataSource class for bookmark operations via MCP.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.core.data_sources import (
    BulkUpdateResult,
    DataSourceConnectionError,
    DataSourceReadError,
    MCPClient,
    MCPToolError,
    RaindropMCPDataSource,
)


class MockMCPClient:
    """Mock MCP client for testing."""

    def __init__(self, responses: Dict[str, Any] = None):
        self.responses = responses or {}
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append((tool_name, arguments))
        if tool_name in self.responses:
            return self.responses[tool_name]
        return {}

    async def list_tools(self) -> List[Dict[str, Any]]:
        return [{"name": "test_tool"}]


class TestRaindropMCPDataSourceBasics:
    """Test basic RaindropMCPDataSource functionality."""

    def test_initialization(self):
        """Test RaindropMCPDataSource initialization."""
        source = RaindropMCPDataSource(
            server_url="http://localhost:3000",
            access_token="test-token"
        )

        assert source.server_url == "http://localhost:3000"
        assert source.access_token == "test-token"
        assert source.is_connected is False

    def test_source_name(self):
        """Test source_name property."""
        source = RaindropMCPDataSource(
            server_url="http://localhost:3000",
            access_token="test-token"
        )

        assert source.source_name == "Raindrop.io (MCP)"

    def test_supports_incremental(self):
        """Test supports_incremental property."""
        source = RaindropMCPDataSource(
            server_url="http://localhost:3000",
            access_token="test-token"
        )

        assert source.supports_incremental is True

    def test_repr(self):
        """Test string representation."""
        source = RaindropMCPDataSource(
            server_url="http://localhost:3000",
            access_token="test-token"
        )

        repr_str = repr(source)
        assert "RaindropMCPDataSource" in repr_str
        assert "localhost:3000" in repr_str


class TestRaindropMCPDataSourceContextManager:
    """Test RaindropMCPDataSource async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_enter(self):
        """Test entering context manager connects to MCP server."""
        with patch.object(
            RaindropMCPDataSource,
            "_load_collections",
            new_callable=AsyncMock
        ):
            source = RaindropMCPDataSource(
                server_url="http://localhost:3000",
                access_token="test-token"
            )

            # Create mock client with async methods
            mock_http_client = AsyncMock()
            mock_http_client.aclose = AsyncMock()

            with patch("bookmark_processor.core.data_sources.mcp_client.httpx.AsyncClient", return_value=mock_http_client):
                async with source:
                    assert source.is_connected is True

    @pytest.mark.asyncio
    async def test_context_manager_exit(self):
        """Test exiting context manager disconnects."""
        with patch.object(
            RaindropMCPDataSource,
            "_load_collections",
            new_callable=AsyncMock
        ):
            source = RaindropMCPDataSource(
                server_url="http://localhost:3000",
                access_token="test-token"
            )

            # Create mock client with async methods
            mock_http_client = AsyncMock()
            mock_http_client.aclose = AsyncMock()

            with patch("bookmark_processor.core.data_sources.mcp_client.httpx.AsyncClient", return_value=mock_http_client):
                async with source:
                    pass

                assert source.is_connected is False


class TestRaindropMCPDataSourceFetchBookmarks:
    """Test fetching bookmarks from Raindrop.io."""

    @pytest.mark.asyncio
    async def test_fetch_bookmarks_basic(self):
        """Test basic bookmark fetching."""
        mock_bookmarks = [
            {
                "_id": 12345,
                "title": "Test Bookmark",
                "link": "https://example.com",
                "tags": ["test", "example"],
                "created": "2024-01-15T10:30:00Z",
                "note": "A test note",
                "excerpt": "Test excerpt",
            }
        ]

        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        with patch.object(
            MCPClient,
            "call_tool",
            new_callable=AsyncMock,
            return_value={"raindrops": mock_bookmarks}
        ):
            with patch.object(
                RaindropMCPDataSource,
                "_load_collections",
                new_callable=AsyncMock
            ):
                source = RaindropMCPDataSource(
                    server_url="http://localhost:3000",
                    access_token="test-token"
                )

                with patch("bookmark_processor.core.data_sources.mcp_client.httpx.AsyncClient", return_value=mock_http_client):
                    async with source:
                        bookmarks = await source.fetch_bookmarks()

                assert len(bookmarks) == 1
                assert bookmarks[0].title == "Test Bookmark"
                assert bookmarks[0].url == "https://example.com"
                assert "test" in bookmarks[0].tags

    @pytest.mark.asyncio
    async def test_fetch_bookmarks_with_collection_filter(self):
        """Test fetching bookmarks with collection filter."""
        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        with patch.object(
            MCPClient,
            "call_tool",
            new_callable=AsyncMock,
            return_value={"raindrops": []}
        ) as mock_call:
            with patch.object(
                RaindropMCPDataSource,
                "_load_collections",
                new_callable=AsyncMock
            ):
                source = RaindropMCPDataSource(
                    server_url="http://localhost:3000",
                    access_token="test-token"
                )
                source._collection_cache = {"tech": 123}

                with patch("bookmark_processor.core.data_sources.mcp_client.httpx.AsyncClient", return_value=mock_http_client):
                    async with source:
                        await source.fetch_bookmarks({"collection": "Tech"})

                # Check collection_id was passed
                call_args = mock_call.call_args_list[-1]
                assert call_args[0][1].get("collection_id") == 123

    @pytest.mark.asyncio
    async def test_fetch_bookmarks_with_tag_filter(self):
        """Test fetching bookmarks with tag filter."""
        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        with patch.object(
            MCPClient,
            "call_tool",
            new_callable=AsyncMock,
            return_value={"raindrops": []}
        ) as mock_call:
            with patch.object(
                RaindropMCPDataSource,
                "_load_collections",
                new_callable=AsyncMock
            ):
                source = RaindropMCPDataSource(
                    server_url="http://localhost:3000",
                    access_token="test-token"
                )

                with patch("bookmark_processor.core.data_sources.mcp_client.httpx.AsyncClient", return_value=mock_http_client):
                    async with source:
                        await source.fetch_bookmarks({"tags": ["python", "ai"]})

                # Check tags were passed
                call_args = mock_call.call_args_list[-1]
                assert call_args[0][1].get("tags") == ["python", "ai"]


class TestRaindropMCPDataSourceUpdateBookmark:
    """Test updating bookmarks in Raindrop.io."""

    @pytest.mark.asyncio
    async def test_update_bookmark_success(self):
        """Test successful bookmark update."""
        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        with patch.object(
            MCPClient,
            "call_tool",
            new_callable=AsyncMock,
            return_value={"success": True}
        ):
            with patch.object(
                RaindropMCPDataSource,
                "_load_collections",
                new_callable=AsyncMock
            ):
                source = RaindropMCPDataSource(
                    server_url="http://localhost:3000",
                    access_token="test-token"
                )

                bookmark = Bookmark(
                    id="12345",
                    title="Updated Title",
                    url="https://example.com"
                )

                with patch("bookmark_processor.core.data_sources.mcp_client.httpx.AsyncClient", return_value=mock_http_client):
                    async with source:
                        result = await source.update_bookmark(bookmark)

                assert result is True

    @pytest.mark.asyncio
    async def test_update_bookmark_without_id(self):
        """Test update fails without bookmark ID."""
        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        with patch.object(
            RaindropMCPDataSource,
            "_load_collections",
            new_callable=AsyncMock
        ):
            source = RaindropMCPDataSource(
                server_url="http://localhost:3000",
                access_token="test-token"
            )

            bookmark = Bookmark(
                title="Test",
                url="https://example.com"
            )  # No ID

            with patch("bookmark_processor.core.data_sources.mcp_client.httpx.AsyncClient", return_value=mock_http_client):
                async with source:
                    result = await source.update_bookmark(bookmark)

            assert result is False


class TestRaindropMCPDataSourceBulkUpdate:
    """Test bulk updating bookmarks."""

    @pytest.mark.asyncio
    async def test_bulk_update_success(self):
        """Test successful bulk update."""
        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        with patch.object(
            MCPClient,
            "call_tool",
            new_callable=AsyncMock,
            return_value={"modified": 2, "errors": []}
        ):
            with patch.object(
                RaindropMCPDataSource,
                "_load_collections",
                new_callable=AsyncMock
            ):
                source = RaindropMCPDataSource(
                    server_url="http://localhost:3000",
                    access_token="test-token"
                )

                bookmarks = [
                    Bookmark(id="1", title="Title 1", url="https://example1.com"),
                    Bookmark(id="2", title="Title 2", url="https://example2.com"),
                ]

                with patch("bookmark_processor.core.data_sources.mcp_client.httpx.AsyncClient", return_value=mock_http_client):
                    async with source:
                        result = await source.bulk_update(bookmarks)

                assert isinstance(result, BulkUpdateResult)
                assert result.total == 2
                assert result.succeeded == 2

    @pytest.mark.asyncio
    async def test_bulk_update_empty_list(self):
        """Test bulk update with empty list."""
        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        with patch.object(
            RaindropMCPDataSource,
            "_load_collections",
            new_callable=AsyncMock
        ):
            source = RaindropMCPDataSource(
                server_url="http://localhost:3000",
                access_token="test-token"
            )

            with patch("bookmark_processor.core.data_sources.mcp_client.httpx.AsyncClient", return_value=mock_http_client):
                async with source:
                    result = await source.bulk_update([])

            assert result.total == 0
            assert result.succeeded == 0


class TestRaindropMCPDataSourceConversion:
    """Test API/Bookmark conversion methods."""

    def test_api_to_bookmark_conversion(self):
        """Test converting API response to Bookmark."""
        source = RaindropMCPDataSource(
            server_url="http://localhost:3000",
            access_token="test-token"
        )

        api_data = {
            "_id": 12345,
            "title": "Test Bookmark",
            "link": "https://example.com",
            "tags": ["test", "example"],
            "created": "2024-01-15T10:30:00Z",
            "note": "A test note",
            "excerpt": "Test excerpt",
            "favorite": True,
        }

        bookmark = source._api_to_bookmark(api_data)

        assert bookmark.id == "12345"
        assert bookmark.title == "Test Bookmark"
        assert bookmark.url == "https://example.com"
        assert "test" in bookmark.tags
        assert bookmark.note == "A test note"
        assert bookmark.excerpt == "Test excerpt"
        assert bookmark.favorite is True

    def test_bookmark_to_api_update_conversion(self):
        """Test converting Bookmark to API update format."""
        source = RaindropMCPDataSource(
            server_url="http://localhost:3000",
            access_token="test-token"
        )

        bookmark = Bookmark(
            id="12345",
            title="Updated Title",
            url="https://example.com",
            tags=["updated", "tags"],
        )
        bookmark.enhanced_description = "Enhanced description"
        bookmark.optimized_tags = ["optimized", "tags"]

        updates = source._bookmark_to_api_update(bookmark)

        assert updates["title"] == "Updated Title"
        assert updates["note"] == "Enhanced description"
        assert updates["tags"] == ["optimized", "tags"]


class TestRaindropMCPDataSourceBackupRestore:
    """Test backup and restore functionality."""

    @pytest.mark.asyncio
    async def test_create_backup(self):
        """Test creating a backup."""
        source = RaindropMCPDataSource(
            server_url="http://localhost:3000",
            access_token="test-token"
        )

        bookmarks = [
            Bookmark(id="1", title="Bookmark 1", url="https://example1.com", tags=["tag1"]),
            Bookmark(id="2", title="Bookmark 2", url="https://example2.com", tags=["tag2"]),
        ]

        backup = await source.create_backup(bookmarks)

        assert backup["source"] == "Raindrop.io (MCP)"
        assert backup["bookmark_count"] == 2
        assert len(backup["bookmarks"]) == 2
        assert backup["bookmarks"][0]["id"] == "1"
        assert "timestamp" in backup

    @pytest.mark.asyncio
    async def test_restore_from_backup(self):
        """Test restoring from a backup."""
        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        with patch.object(
            MCPClient,
            "call_tool",
            new_callable=AsyncMock,
            return_value={"modified": 2, "errors": []}
        ):
            with patch.object(
                RaindropMCPDataSource,
                "_load_collections",
                new_callable=AsyncMock
            ):
                source = RaindropMCPDataSource(
                    server_url="http://localhost:3000",
                    access_token="test-token"
                )

                backup_data = {
                    "timestamp": "2024-01-15T10:30:00",
                    "source": "Raindrop.io (MCP)",
                    "bookmark_count": 2,
                    "bookmarks": [
                        {"id": "1", "url": "https://example1.com", "title": "Title 1", "tags": []},
                        {"id": "2", "url": "https://example2.com", "title": "Title 2", "tags": []},
                    ]
                }

                with patch("bookmark_processor.core.data_sources.mcp_client.httpx.AsyncClient", return_value=mock_http_client):
                    async with source:
                        result = await source.restore_from_backup(backup_data)

                assert result.total == 2
                assert result.succeeded == 2


class TestRaindropMCPDataSourceCollections:
    """Test collection management."""

    @pytest.mark.asyncio
    async def test_get_collections(self):
        """Test getting collections list."""
        mock_http_client = AsyncMock()
        mock_http_client.aclose = AsyncMock()

        mock_collections = [
            {"_id": 1, "title": "Tech"},
            {"_id": 2, "title": "Research"},
        ]

        with patch.object(
            MCPClient,
            "call_tool",
            new_callable=AsyncMock,
            return_value={"collections": mock_collections}
        ):
            with patch.object(
                RaindropMCPDataSource,
                "_load_collections",
                new_callable=AsyncMock
            ):
                source = RaindropMCPDataSource(
                    server_url="http://localhost:3000",
                    access_token="test-token"
                )

                with patch("bookmark_processor.core.data_sources.mcp_client.httpx.AsyncClient", return_value=mock_http_client):
                    async with source:
                        collections = await source.get_collections()

                assert len(collections) == 2
                assert collections[0]["title"] == "Tech"

    def test_get_collection_id(self):
        """Test getting collection ID from name."""
        source = RaindropMCPDataSource(
            server_url="http://localhost:3000",
            access_token="test-token"
        )
        source._collection_cache = {
            "tech": 123,
            "research": 456,
        }

        assert source._get_collection_id("Tech") == 123
        assert source._get_collection_id("TECH") == 123
        assert source._get_collection_id("NotFound") is None

    def test_get_collection_name(self):
        """Test getting collection name from ID."""
        source = RaindropMCPDataSource(
            server_url="http://localhost:3000",
            access_token="test-token"
        )
        source._collection_name_cache = {
            123: "Tech",
            456: "Research",
        }

        assert source._get_collection_name(123) == "Tech"
        assert source._get_collection_name(999) == ""


class TestRaindropMCPDataSourceErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_fetch_not_connected_raises_error(self):
        """Test fetch raises error when not connected."""
        source = RaindropMCPDataSource(
            server_url="http://localhost:3000",
            access_token="test-token"
        )

        with pytest.raises(DataSourceConnectionError):
            await source.fetch_bookmarks()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
