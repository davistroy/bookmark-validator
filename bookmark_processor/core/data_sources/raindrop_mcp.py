"""
Raindrop.io MCP Data Source Implementation.

This module provides a Raindrop.io data source that communicates through
an MCP (Model Context Protocol) server, enabling direct API integration
without manual CSV export/import.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from ..data_models import Bookmark, BookmarkMetadata
from .mcp_client import (
    MCPClient,
    MCPClientError,
    MCPConnectionError,
    MCPTimeoutError,
    MCPToolError,
)
from .protocol import (
    AbstractBookmarkDataSource,
    BulkUpdateResult,
    DataSourceConnectionError,
    DataSourceError,
    DataSourceReadError,
    DataSourceWriteError,
)
from .state_tracker import ProcessingStateTracker


class RaindropMCPDataSource(AbstractBookmarkDataSource):
    """
    Raindrop.io data source via MCP (Model Context Protocol) server.

    This data source communicates with Raindrop.io through an MCP server,
    enabling direct API access for fetching and updating bookmarks without
    requiring manual CSV export/import.

    The MCP server acts as an intermediary, translating MCP tool calls
    into Raindrop.io API requests. This approach provides:
    - Real-time bookmark access
    - Incremental updates
    - Direct API modifications
    - Collection and tag filtering

    Attributes:
        server_url: URL of the MCP server
        access_token: Raindrop.io API access token
        state_tracker: Optional state tracker for incremental processing
        collection_cache: Cache of collection ID to name mappings

    Example:
        >>> source = RaindropMCPDataSource(
        ...     server_url="http://localhost:3000",
        ...     access_token="your-token"
        ... )
        >>> async with source:
        ...     bookmarks = await source.fetch_bookmarks(
        ...         filters={"collection": "Tech"}
        ...     )
    """

    # Tool names for Raindrop.io MCP operations
    TOOL_BOOKMARK_SEARCH = "bookmark_search"
    TOOL_BOOKMARK_MANAGE = "bookmark_manage"
    TOOL_BULK_EDIT = "bulk_edit_raindrops"
    TOOL_LIST_COLLECTIONS = "list_collections"
    TOOL_GET_BOOKMARK = "get_raindrop"

    def __init__(
        self,
        server_url: str,
        access_token: str,
        state_tracker: Optional[ProcessingStateTracker] = None,
        timeout: float = 30.0,
        batch_size: int = 50
    ):
        """
        Initialize the Raindrop.io MCP data source.

        Args:
            server_url: URL of the MCP server
            access_token: Raindrop.io API access token
            state_tracker: Optional state tracker for incremental processing
            timeout: Request timeout in seconds
            batch_size: Number of bookmarks to fetch per request
        """
        self.server_url = server_url
        self.access_token = access_token
        self.state_tracker = state_tracker
        self.timeout = timeout
        self.batch_size = batch_size
        self._client: Optional[MCPClient] = None
        self._connected = False
        self._collection_cache: Dict[str, int] = {}
        self._collection_name_cache: Dict[int, str] = {}
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self) -> "RaindropMCPDataSource":
        """
        Enter async context and connect to MCP server.

        Returns:
            Self for use in async with block
        """
        self._client = MCPClient(
            server_url=self.server_url,
            timeout=self.timeout,
            access_token=self.access_token
        )
        await self._client.__aenter__()
        self._connected = True
        self.logger.info(f"Connected to Raindrop.io MCP server at {self.server_url}")

        # Pre-load collection cache
        try:
            await self._load_collections()
        except Exception as e:
            self.logger.warning(f"Failed to pre-load collections: {e}")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit async context and disconnect from MCP server.

        Args:
            exc_type: Exception type if any
            exc_val: Exception value if any
            exc_tb: Exception traceback if any
        """
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            self._client = None
        self._connected = False
        self.logger.info("Disconnected from Raindrop.io MCP server")

    def _ensure_connected(self) -> None:
        """
        Ensure data source is connected.

        Raises:
            DataSourceConnectionError: If not connected
        """
        if not self._connected or self._client is None:
            raise DataSourceConnectionError(
                "Not connected to MCP server. Use 'async with' context manager.",
                source_name=self.source_name
            )

    async def _load_collections(self) -> None:
        """
        Load and cache collection mappings from Raindrop.io.
        """
        try:
            result = await self._client.call_tool(
                self.TOOL_LIST_COLLECTIONS,
                {"access_token": self.access_token}
            )

            collections = result.get("collections", [])
            for collection in collections:
                collection_id = collection.get("_id")
                collection_title = collection.get("title", "")
                if collection_id is not None:
                    self._collection_cache[collection_title.lower()] = collection_id
                    self._collection_name_cache[collection_id] = collection_title

            self.logger.debug(f"Loaded {len(self._collection_cache)} collections")

        except MCPToolError as e:
            self.logger.warning(f"Failed to load collections: {e}")

    def _get_collection_id(self, collection_name: str) -> Optional[int]:
        """
        Get collection ID from name (case-insensitive).

        Args:
            collection_name: Collection name

        Returns:
            Collection ID or None if not found
        """
        return self._collection_cache.get(collection_name.lower())

    def _get_collection_name(self, collection_id: int) -> str:
        """
        Get collection name from ID.

        Args:
            collection_id: Collection ID

        Returns:
            Collection name or empty string if not found
        """
        return self._collection_name_cache.get(collection_id, "")

    async def fetch_bookmarks(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Bookmark]:
        """
        Fetch bookmarks from Raindrop.io via MCP.

        Args:
            filters: Optional filter criteria:
                - collection: Collection name to filter by
                - tags: List of tags to filter by
                - query: Search query string
                - since: Filter bookmarks created after this datetime
                - since_last_run: Only fetch bookmarks not yet processed
                - limit: Maximum number of bookmarks to fetch

        Returns:
            List of Bookmark objects

        Raises:
            DataSourceReadError: If fetching fails
            DataSourceConnectionError: If not connected
        """
        self._ensure_connected()
        self.logger.info("Fetching bookmarks from Raindrop.io")

        try:
            # Build search parameters
            params = {"access_token": self.access_token}

            if filters:
                # Collection filter
                if "collection" in filters:
                    collection_name = filters["collection"]
                    collection_id = self._get_collection_id(collection_name)
                    if collection_id is not None:
                        params["collection_id"] = collection_id
                    else:
                        # Try as numeric ID
                        try:
                            params["collection_id"] = int(collection_name)
                        except ValueError:
                            self.logger.warning(
                                f"Collection '{collection_name}' not found"
                            )

                # Tag filter
                if "tags" in filters:
                    tags = filters["tags"]
                    if isinstance(tags, list):
                        params["tags"] = tags
                    else:
                        params["tags"] = [tags]

                # Search query
                if "query" in filters:
                    params["query"] = filters["query"]

                # Date filter
                if "since" in filters:
                    since = filters["since"]
                    if isinstance(since, datetime):
                        params["created_after"] = since.isoformat()
                    elif isinstance(since, timedelta):
                        since_date = datetime.now() - since
                        params["created_after"] = since_date.isoformat()

                # Limit
                if "limit" in filters:
                    params["perpage"] = min(filters["limit"], 50)
                else:
                    params["perpage"] = self.batch_size

            # Fetch bookmarks from MCP server
            all_bookmarks = []
            page = 0
            max_bookmarks = filters.get("limit", 10000) if filters else 10000

            while len(all_bookmarks) < max_bookmarks:
                params["page"] = page
                result = await self._client.call_tool(
                    self.TOOL_BOOKMARK_SEARCH,
                    params
                )

                items = result.get("raindrops", result.get("items", []))
                if not items:
                    break

                for item in items:
                    bookmark = self._api_to_bookmark(item)
                    all_bookmarks.append(bookmark)

                # Check if there are more pages
                if len(items) < params.get("perpage", self.batch_size):
                    break

                page += 1

                # Safety limit
                if page > 200:
                    self.logger.warning("Reached page limit (200), stopping fetch")
                    break

            self.logger.info(f"Fetched {len(all_bookmarks)} bookmarks from Raindrop.io")

            # Apply incremental filter if requested
            if filters and filters.get("since_last_run") and self.state_tracker:
                all_bookmarks = self.state_tracker.get_unprocessed(all_bookmarks)
                self.logger.info(
                    f"After incremental filter: {len(all_bookmarks)} bookmarks to process"
                )

            return all_bookmarks

        except MCPConnectionError as e:
            raise DataSourceConnectionError(
                f"Failed to connect to Raindrop.io: {e.message}",
                source_name=self.source_name,
                original_error=e.original_error
            )
        except MCPTimeoutError as e:
            raise DataSourceReadError(
                f"Timeout fetching bookmarks: {e.message}",
                source_name=self.source_name,
                original_error=e.original_error
            )
        except MCPToolError as e:
            raise DataSourceReadError(
                f"Failed to fetch bookmarks: {e.message}",
                source_name=self.source_name,
                original_error=e.original_error
            )
        except MCPClientError as e:
            raise DataSourceReadError(
                f"Error fetching bookmarks: {e.message}",
                source_name=self.source_name,
                original_error=e.original_error
            )

    async def update_bookmark(self, bookmark: Bookmark) -> bool:
        """
        Update a single bookmark in Raindrop.io.

        Args:
            bookmark: The bookmark to update (must have valid ID)

        Returns:
            True if update succeeded, False otherwise

        Raises:
            DataSourceWriteError: If update fails due to API error
            DataSourceConnectionError: If not connected
        """
        self._ensure_connected()

        if not bookmark.id:
            self.logger.warning(f"Cannot update bookmark without ID: {bookmark.url}")
            return False

        try:
            updates = self._bookmark_to_api_update(bookmark)

            await self._client.call_tool(
                self.TOOL_BOOKMARK_MANAGE,
                {
                    "access_token": self.access_token,
                    "action": "update",
                    "id": int(bookmark.id),
                    "updates": updates
                }
            )

            # Mark as processed in state tracker
            if self.state_tracker:
                self.state_tracker.mark_processed(
                    bookmark,
                    ai_engine="mcp-raindrop"
                )

            self.logger.debug(f"Updated bookmark: {bookmark.url}")
            return True

        except MCPToolError as e:
            self.logger.error(f"Failed to update bookmark {bookmark.url}: {e}")
            return False
        except MCPClientError as e:
            self.logger.error(f"API error updating bookmark {bookmark.url}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error updating bookmark: {e}")
            return False

    async def bulk_update(
        self,
        bookmarks: List[Bookmark]
    ) -> BulkUpdateResult:
        """
        Bulk update multiple bookmarks in Raindrop.io.

        This method attempts to use the bulk edit API if available,
        falling back to individual updates if needed.

        Args:
            bookmarks: List of bookmarks to update

        Returns:
            BulkUpdateResult with statistics and any errors

        Raises:
            DataSourceWriteError: If bulk update fails completely
        """
        self._ensure_connected()

        if not bookmarks:
            return BulkUpdateResult(total=0, succeeded=0, failed=0, errors=[])

        self.logger.info(f"Bulk updating {len(bookmarks)} bookmarks")

        # Filter to only bookmarks with IDs
        valid_bookmarks = [b for b in bookmarks if b.id]
        invalid_count = len(bookmarks) - len(valid_bookmarks)

        if invalid_count > 0:
            self.logger.warning(
                f"{invalid_count} bookmarks skipped (no ID)"
            )

        # Try bulk API first
        try:
            result = await self._bulk_update_api(valid_bookmarks)
            return result
        except MCPToolError:
            self.logger.info("Bulk API not available, falling back to individual updates")
            return await self._bulk_update_individual(valid_bookmarks)

    async def _bulk_update_api(
        self,
        bookmarks: List[Bookmark]
    ) -> BulkUpdateResult:
        """
        Attempt bulk update using MCP bulk edit tool.

        Args:
            bookmarks: Bookmarks to update

        Returns:
            BulkUpdateResult
        """
        ids = [int(b.id) for b in bookmarks if b.id]
        updates = [self._bookmark_to_api_update(b) for b in bookmarks if b.id]

        result = await self._client.call_tool(
            self.TOOL_BULK_EDIT,
            {
                "access_token": self.access_token,
                "ids": ids,
                "updates": updates
            }
        )

        modified = result.get("modified", 0)
        errors = result.get("errors", [])

        # Mark successful ones as processed
        if self.state_tracker:
            for bookmark in bookmarks[:modified]:
                self.state_tracker.mark_processed(bookmark, ai_engine="mcp-raindrop")

        return BulkUpdateResult(
            total=len(bookmarks),
            succeeded=modified,
            failed=len(bookmarks) - modified,
            errors=errors
        )

    async def _bulk_update_individual(
        self,
        bookmarks: List[Bookmark]
    ) -> BulkUpdateResult:
        """
        Fallback bulk update using individual update calls.

        Args:
            bookmarks: Bookmarks to update

        Returns:
            BulkUpdateResult
        """
        succeeded = 0
        failed = 0
        errors = []

        for bookmark in bookmarks:
            try:
                if await self.update_bookmark(bookmark):
                    succeeded += 1
                else:
                    failed += 1
                    errors.append({
                        "url": bookmark.url,
                        "id": bookmark.id,
                        "error": "Update returned False"
                    })
            except Exception as e:
                failed += 1
                errors.append({
                    "url": bookmark.url,
                    "id": bookmark.id,
                    "error": str(e)
                })

        return BulkUpdateResult(
            total=len(bookmarks),
            succeeded=succeeded,
            failed=failed,
            errors=errors
        )

    def _api_to_bookmark(self, data: Dict[str, Any]) -> Bookmark:
        """
        Convert Raindrop.io API response to Bookmark object.

        Args:
            data: API response dictionary

        Returns:
            Bookmark object
        """
        # Parse created date
        created = None
        created_str = data.get("created", "")
        if created_str:
            try:
                # Handle ISO format with timezone
                created = datetime.fromisoformat(
                    created_str.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        # Extract collection/folder info
        collection = data.get("collection", {})
        if isinstance(collection, dict):
            collection_id = collection.get("$id") or collection.get("_id")
            folder = self._get_collection_name(collection_id) if collection_id else ""
        else:
            folder = ""

        # Extract tags
        tags = data.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        # Create bookmark
        bookmark = Bookmark(
            id=str(data.get("_id", "")),
            title=data.get("title", ""),
            note=data.get("note", ""),
            excerpt=data.get("excerpt", ""),
            url=data.get("link", ""),
            folder=folder,
            tags=tags,
            created=created,
            cover=data.get("cover", ""),
            highlights=data.get("highlights", ""),
            favorite=data.get("favorite", False)
        )

        # Extract metadata if available
        if data.get("type") or data.get("domain"):
            bookmark.extracted_metadata = BookmarkMetadata(
                title=data.get("title"),
                description=data.get("excerpt"),
                author=data.get("author"),
                canonical_url=data.get("link")
            )

        return bookmark

    def _bookmark_to_api_update(self, bookmark: Bookmark) -> Dict[str, Any]:
        """
        Convert Bookmark to Raindrop.io API update format.

        Args:
            bookmark: Bookmark object

        Returns:
            Dictionary suitable for API update
        """
        updates = {}

        # Title
        effective_title = bookmark.get_effective_title()
        if effective_title:
            updates["title"] = effective_title

        # Note/description (Raindrop uses 'note' field for notes)
        effective_desc = bookmark.get_effective_description()
        if effective_desc:
            updates["note"] = effective_desc

        # Excerpt (for display in Raindrop)
        if bookmark.excerpt:
            updates["excerpt"] = bookmark.excerpt

        # Tags - use optimized if available, otherwise original
        tags = bookmark.optimized_tags if bookmark.optimized_tags else bookmark.tags
        if tags:
            updates["tags"] = tags

        # Folder/collection
        if bookmark.folder:
            collection_id = self._get_collection_id(bookmark.folder)
            if collection_id is not None:
                updates["collection"] = {"$id": collection_id}

        # Favorite
        if bookmark.favorite:
            updates["important"] = True

        return updates

    def _folder_to_collection_id(self, folder: str) -> Optional[int]:
        """
        Convert folder name to collection ID.

        Args:
            folder: Folder name

        Returns:
            Collection ID or None if not found
        """
        return self._get_collection_id(folder)

    async def get_collections(self) -> List[Dict[str, Any]]:
        """
        Get list of Raindrop.io collections.

        Returns:
            List of collection dictionaries with id, title, count, etc.
        """
        self._ensure_connected()

        try:
            result = await self._client.call_tool(
                self.TOOL_LIST_COLLECTIONS,
                {"access_token": self.access_token}
            )
            return result.get("collections", [])
        except MCPClientError as e:
            self.logger.error(f"Failed to get collections: {e}")
            return []

    async def get_bookmark_by_id(self, bookmark_id: str) -> Optional[Bookmark]:
        """
        Get a single bookmark by ID.

        Args:
            bookmark_id: Raindrop.io bookmark ID

        Returns:
            Bookmark if found, None otherwise
        """
        self._ensure_connected()

        try:
            result = await self._client.call_tool(
                self.TOOL_GET_BOOKMARK,
                {
                    "access_token": self.access_token,
                    "id": int(bookmark_id)
                }
            )
            if result and "item" in result:
                return self._api_to_bookmark(result["item"])
            return None
        except MCPClientError as e:
            self.logger.error(f"Failed to get bookmark {bookmark_id}: {e}")
            return None

    async def create_backup(
        self,
        bookmarks: List[Bookmark]
    ) -> Dict[str, Any]:
        """
        Create a backup record of bookmarks before modification.

        This is used for rollback functionality.

        Args:
            bookmarks: Bookmarks to back up

        Returns:
            Backup record with timestamp and bookmark data
        """
        backup = {
            "timestamp": datetime.now().isoformat(),
            "source": self.source_name,
            "bookmark_count": len(bookmarks),
            "bookmarks": [
                {
                    "id": b.id,
                    "url": b.url,
                    "title": b.title,
                    "note": b.note,
                    "tags": b.tags,
                    "folder": b.folder
                }
                for b in bookmarks
            ]
        }
        return backup

    async def restore_from_backup(
        self,
        backup: Dict[str, Any]
    ) -> BulkUpdateResult:
        """
        Restore bookmarks from a backup record.

        Args:
            backup: Backup record created by create_backup()

        Returns:
            BulkUpdateResult indicating restoration success
        """
        self._ensure_connected()

        bookmark_data = backup.get("bookmarks", [])
        if not bookmark_data:
            return BulkUpdateResult(total=0, succeeded=0, failed=0, errors=[])

        # Convert backup data to bookmarks
        bookmarks = []
        for data in bookmark_data:
            bookmark = Bookmark(
                id=data.get("id"),
                url=data.get("url", ""),
                title=data.get("title", ""),
                note=data.get("note", ""),
                tags=data.get("tags", []),
                folder=data.get("folder", "")
            )
            bookmarks.append(bookmark)

        # Restore using bulk update
        return await self.bulk_update(bookmarks)

    # Implement abstract methods from protocol

    def fetch_bookmarks_sync(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Bookmark]:
        """
        Synchronous wrapper for fetch_bookmarks.

        Note: This is for protocol compliance. Prefer async version.
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.fetch_bookmarks(filters)
        )

    def update_bookmark_sync(self, bookmark: Bookmark) -> bool:
        """
        Synchronous wrapper for update_bookmark.

        Note: This is for protocol compliance. Prefer async version.
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.update_bookmark(bookmark)
        )

    @property
    def supports_incremental(self) -> bool:
        """
        Whether this data source supports incremental updates.

        Returns:
            True - Raindrop.io MCP supports incremental updates
        """
        return True

    @property
    def source_name(self) -> str:
        """
        Human-readable name for this data source.

        Returns:
            "Raindrop.io (MCP)"
        """
        return "Raindrop.io (MCP)"

    @property
    def is_connected(self) -> bool:
        """Check if data source is currently connected."""
        return self._connected and self._client is not None

    def __repr__(self) -> str:
        return (
            f"RaindropMCPDataSource(server_url={self.server_url!r}, "
            f"connected={self.is_connected}, "
            f"collections={len(self._collection_cache)})"
        )
