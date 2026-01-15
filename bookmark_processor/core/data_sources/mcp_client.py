"""
MCP Client for communicating with Model Context Protocol servers.

This module provides an async HTTP client for interacting with MCP servers,
enabling the bookmark processor to communicate with external services like
Raindrop.io through MCP-compatible interfaces.
"""

import logging
from typing import Any, Dict, List, Optional

import httpx


class MCPClientError(Exception):
    """
    Base exception for MCP client errors.

    Attributes:
        message: Error description
        status_code: HTTP status code if applicable
        original_error: The underlying exception if any
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.status_code = status_code
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [self.message]
        if self.status_code:
            parts.append(f"(HTTP {self.status_code})")
        if self.original_error:
            parts.append(f"Caused by: {type(self.original_error).__name__}: {self.original_error}")
        return " ".join(parts)


class MCPConnectionError(MCPClientError):
    """Exception raised when connection to MCP server fails."""
    pass


class MCPTimeoutError(MCPClientError):
    """Exception raised when MCP request times out."""
    pass


class MCPToolError(MCPClientError):
    """Exception raised when MCP tool execution fails."""
    pass


class MCPAuthenticationError(MCPClientError):
    """Exception raised when MCP authentication fails."""
    pass


class MCPClient:
    """
    Client for communicating with MCP (Model Context Protocol) servers.

    This client provides an async interface for interacting with MCP servers,
    supporting tool discovery, invocation, and resource management.

    The client is designed to be used as an async context manager:

        async with MCPClient("http://localhost:3000") as client:
            tools = await client.list_tools()
            result = await client.call_tool("bookmark_search", {"query": "test"})

    Attributes:
        server_url: Base URL of the MCP server
        timeout: Request timeout in seconds
        access_token: Optional authentication token

    Example:
        >>> async with MCPClient("http://localhost:3000", timeout=30.0) as client:
        ...     tools = await client.list_tools()
        ...     for tool in tools:
        ...         print(f"Tool: {tool['name']}")
    """

    # Default headers for MCP protocol
    DEFAULT_HEADERS = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    def __init__(
        self,
        server_url: str,
        timeout: float = 30.0,
        access_token: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the MCP client.

        Args:
            server_url: Base URL of the MCP server (trailing slash removed)
            timeout: Request timeout in seconds (default: 30.0)
            access_token: Optional authentication token for the MCP server
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Base delay between retries in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.access_token = access_token
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._client: Optional[httpx.AsyncClient] = None
        self._connected = False
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self) -> "MCPClient":
        """
        Enter async context and create HTTP client.

        Returns:
            Self for use in async with block
        """
        headers = self.DEFAULT_HEADERS.copy()
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers=headers,
            follow_redirects=True
        )
        self._connected = True
        self.logger.debug(f"MCP client connected to {self.server_url}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit async context and close HTTP client.

        Args:
            exc_type: Exception type if any
            exc_val: Exception value if any
            exc_tb: Exception traceback if any
        """
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        self.logger.debug("MCP client disconnected")

    def _ensure_connected(self) -> None:
        """
        Ensure client is connected.

        Raises:
            MCPConnectionError: If client is not connected
        """
        if not self._connected or self._client is None:
            raise MCPConnectionError(
                "MCP client not connected. Use 'async with' context manager."
            )

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the MCP server with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (will be appended to server_url)
            json_data: Optional JSON body data
            params: Optional query parameters

        Returns:
            Parsed JSON response

        Raises:
            MCPConnectionError: If connection fails
            MCPTimeoutError: If request times out
            MCPClientError: For other HTTP errors
        """
        self._ensure_connected()

        url = f"{self.server_url}/{endpoint.lstrip('/')}"
        last_error: Optional[Exception] = None

        for attempt in range(self.retry_attempts):
            try:
                if method.upper() == "GET":
                    response = await self._client.get(url, params=params)
                elif method.upper() == "POST":
                    response = await self._client.post(url, json=json_data, params=params)
                elif method.upper() == "PUT":
                    response = await self._client.put(url, json=json_data, params=params)
                elif method.upper() == "DELETE":
                    response = await self._client.delete(url, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Check for HTTP errors
                if response.status_code == 401:
                    raise MCPAuthenticationError(
                        "Authentication failed",
                        status_code=401
                    )
                elif response.status_code == 403:
                    raise MCPAuthenticationError(
                        "Access forbidden",
                        status_code=403
                    )
                elif response.status_code >= 400:
                    error_body = response.text
                    raise MCPClientError(
                        f"HTTP error: {error_body}",
                        status_code=response.status_code
                    )

                # Parse and return JSON response
                return response.json()

            except httpx.ConnectError as e:
                last_error = e
                self.logger.warning(
                    f"Connection error (attempt {attempt + 1}/{self.retry_attempts}): {e}"
                )
                if attempt < self.retry_attempts - 1:
                    import asyncio
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                continue

            except httpx.TimeoutException as e:
                last_error = e
                self.logger.warning(
                    f"Timeout error (attempt {attempt + 1}/{self.retry_attempts}): {e}"
                )
                if attempt < self.retry_attempts - 1:
                    import asyncio
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                continue

            except (MCPClientError, MCPAuthenticationError):
                # Don't retry auth errors
                raise

            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Request error (attempt {attempt + 1}/{self.retry_attempts}): {e}"
                )
                if attempt < self.retry_attempts - 1:
                    import asyncio
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                continue

        # All retries exhausted
        if isinstance(last_error, httpx.ConnectError):
            raise MCPConnectionError(
                f"Failed to connect to {self.server_url}",
                original_error=last_error
            )
        elif isinstance(last_error, httpx.TimeoutException):
            raise MCPTimeoutError(
                f"Request to {url} timed out after {self.timeout}s",
                original_error=last_error
            )
        else:
            raise MCPClientError(
                f"Request failed after {self.retry_attempts} attempts",
                original_error=last_error
            )

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call an MCP tool with arguments.

        This method invokes a tool on the MCP server and returns the result.
        The tool name and arguments are sent to the server, which executes
        the tool and returns the result.

        Args:
            tool_name: Name of the tool to invoke
            arguments: Dictionary of arguments to pass to the tool

        Returns:
            Dictionary containing the tool's response

        Raises:
            MCPToolError: If tool execution fails
            MCPConnectionError: If connection fails
            MCPTimeoutError: If request times out

        Example:
            >>> result = await client.call_tool("bookmark_search", {
            ...     "access_token": "token",
            ...     "query": "python"
            ... })
        """
        self.logger.debug(f"Calling MCP tool: {tool_name}")

        try:
            response = await self._make_request(
                "POST",
                f"tools/{tool_name}",
                json_data={"arguments": arguments}
            )

            # Check for tool-level errors in response
            if "error" in response:
                raise MCPToolError(
                    f"Tool '{tool_name}' error: {response['error']}"
                )

            self.logger.debug(f"Tool {tool_name} completed successfully")
            return response

        except (MCPConnectionError, MCPTimeoutError, MCPAuthenticationError):
            raise
        except MCPClientError as e:
            raise MCPToolError(
                f"Failed to execute tool '{tool_name}': {e.message}",
                status_code=e.status_code,
                original_error=e.original_error
            )

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available MCP tools on the server.

        Returns a list of tool definitions including their names,
        descriptions, and parameter schemas.

        Returns:
            List of tool definition dictionaries, each containing:
            - name: Tool name
            - description: Tool description
            - inputSchema: JSON Schema for tool parameters

        Raises:
            MCPConnectionError: If connection fails
            MCPTimeoutError: If request times out

        Example:
            >>> tools = await client.list_tools()
            >>> for tool in tools:
            ...     print(f"Tool: {tool['name']} - {tool.get('description', '')}")
        """
        self.logger.debug("Listing MCP tools")

        try:
            response = await self._make_request("GET", "tools")

            # Extract tools list from response
            tools = response.get("tools", [])
            self.logger.debug(f"Found {len(tools)} MCP tools")
            return tools

        except (MCPConnectionError, MCPTimeoutError, MCPAuthenticationError):
            raise
        except MCPClientError as e:
            self.logger.error(f"Failed to list tools: {e}")
            raise

    async def list_resources(self) -> List[Dict[str, Any]]:
        """
        List available MCP resources on the server.

        Returns a list of resource definitions that the server provides.

        Returns:
            List of resource definition dictionaries

        Raises:
            MCPConnectionError: If connection fails
            MCPTimeoutError: If request times out
        """
        self.logger.debug("Listing MCP resources")

        try:
            response = await self._make_request("GET", "resources")
            resources = response.get("resources", [])
            self.logger.debug(f"Found {len(resources)} MCP resources")
            return resources

        except (MCPConnectionError, MCPTimeoutError, MCPAuthenticationError):
            raise
        except MCPClientError as e:
            self.logger.error(f"Failed to list resources: {e}")
            raise

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read an MCP resource by URI.

        Args:
            uri: Resource URI to read

        Returns:
            Resource content as dictionary

        Raises:
            MCPConnectionError: If connection fails
            MCPTimeoutError: If request times out
            MCPClientError: If resource read fails
        """
        self.logger.debug(f"Reading MCP resource: {uri}")

        try:
            response = await self._make_request(
                "POST",
                "resources/read",
                json_data={"uri": uri}
            )
            return response

        except (MCPConnectionError, MCPTimeoutError, MCPAuthenticationError):
            raise
        except MCPClientError as e:
            self.logger.error(f"Failed to read resource {uri}: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if the MCP server is healthy and responding.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            # Try to list tools as a health check
            await self.list_tools()
            return True
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        """Check if client is currently connected."""
        return self._connected and self._client is not None

    def __repr__(self) -> str:
        return (
            f"MCPClient(server_url={self.server_url!r}, "
            f"timeout={self.timeout}, "
            f"connected={self.is_connected})"
        )
