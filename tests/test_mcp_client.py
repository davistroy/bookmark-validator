"""
Unit tests for the MCP Client.

Tests the MCPClient class for communicating with MCP servers.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bookmark_processor.core.data_sources import (
    MCPClient,
    MCPClientError,
    MCPConnectionError,
    MCPTimeoutError,
    MCPToolError,
    MCPAuthenticationError,
)


class TestMCPClientBasics:
    """Test basic MCPClient functionality."""

    def test_initialization(self):
        """Test MCPClient initialization."""
        client = MCPClient(
            server_url="http://localhost:3000",
            timeout=30.0
        )

        assert client.server_url == "http://localhost:3000"
        assert client.timeout == 30.0
        assert client.access_token is None
        assert client.is_connected is False

    def test_initialization_with_token(self):
        """Test MCPClient initialization with access token."""
        client = MCPClient(
            server_url="http://localhost:3000",
            access_token="test-token"
        )

        assert client.access_token == "test-token"

    def test_initialization_strips_trailing_slash(self):
        """Test that server URL trailing slash is stripped."""
        client = MCPClient(server_url="http://localhost:3000/")
        assert client.server_url == "http://localhost:3000"

    def test_repr(self):
        """Test string representation."""
        client = MCPClient("http://localhost:3000", timeout=30.0)
        repr_str = repr(client)

        assert "MCPClient" in repr_str
        assert "localhost:3000" in repr_str
        assert "timeout=30.0" in repr_str


class TestMCPClientContextManager:
    """Test MCPClient async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_enter(self):
        """Test entering context manager."""
        client = MCPClient("http://localhost:3000")

        async with client:
            assert client.is_connected is True
            assert client._client is not None

    @pytest.mark.asyncio
    async def test_context_manager_exit(self):
        """Test exiting context manager."""
        client = MCPClient("http://localhost:3000")

        async with client:
            pass

        assert client.is_connected is False
        assert client._client is None

    @pytest.mark.asyncio
    async def test_ensure_connected_raises_when_not_connected(self):
        """Test that _ensure_connected raises when not connected."""
        client = MCPClient("http://localhost:3000")

        with pytest.raises(MCPConnectionError) as exc_info:
            client._ensure_connected()

        assert "not connected" in str(exc_info.value).lower()


class TestMCPClientToolCalls:
    """Test MCP tool calling functionality."""

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test successful tool call."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": "success"}
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = MCPClient("http://localhost:3000")

            async with client:
                result = await client.call_tool(
                    "test_tool",
                    {"param1": "value1"}
                )

            assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_call_tool_with_error_response(self):
        """Test tool call with error in response."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"error": "Tool execution failed"}
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = MCPClient("http://localhost:3000")

            async with client:
                with pytest.raises(MCPToolError) as exc_info:
                    await client.call_tool("test_tool", {})

            assert "Tool execution failed" in str(exc_info.value)


class TestMCPClientListTools:
    """Test MCP list_tools functionality."""

    @pytest.mark.asyncio
    async def test_list_tools_success(self):
        """Test successful tool listing."""
        mock_tools = [
            {"name": "tool1", "description": "First tool"},
            {"name": "tool2", "description": "Second tool"},
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"tools": mock_tools}
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = MCPClient("http://localhost:3000")

            async with client:
                tools = await client.list_tools()

            assert len(tools) == 2
            assert tools[0]["name"] == "tool1"
            assert tools[1]["name"] == "tool2"

    @pytest.mark.asyncio
    async def test_list_tools_empty(self):
        """Test list_tools with no tools."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"tools": []}
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = MCPClient("http://localhost:3000")

            async with client:
                tools = await client.list_tools()

            assert tools == []


class TestMCPClientErrorHandling:
    """Test MCP client error handling."""

    @pytest.mark.asyncio
    async def test_authentication_error(self):
        """Test handling of 401 authentication error."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = MCPClient("http://localhost:3000")

            async with client:
                with pytest.raises(MCPAuthenticationError):
                    await client.list_tools()

    @pytest.mark.asyncio
    async def test_forbidden_error(self):
        """Test handling of 403 forbidden error."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = MCPClient("http://localhost:3000")

            async with client:
                with pytest.raises(MCPAuthenticationError):
                    await client.list_tools()

    @pytest.mark.asyncio
    async def test_http_error(self):
        """Test handling of HTTP errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = MCPClient("http://localhost:3000")

            async with client:
                with pytest.raises(MCPClientError) as exc_info:
                    await client.list_tools()

            assert exc_info.value.status_code == 500


class TestMCPClientHealthCheck:
    """Test MCP client health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check returns True when healthy."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"tools": [{"name": "test"}]}
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = MCPClient("http://localhost:3000")

            async with client:
                healthy = await client.health_check()

            assert healthy is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Test health check returns False when unhealthy."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client_class.return_value = mock_client

            client = MCPClient("http://localhost:3000", retry_attempts=1)

            async with client:
                healthy = await client.health_check()

            assert healthy is False


class TestMCPClientResources:
    """Test MCP resource operations."""

    @pytest.mark.asyncio
    async def test_list_resources(self):
        """Test listing resources."""
        mock_resources = [
            {"uri": "resource://test1"},
            {"uri": "resource://test2"},
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"resources": mock_resources}
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = MCPClient("http://localhost:3000")

            async with client:
                resources = await client.list_resources()

            assert len(resources) == 2

    @pytest.mark.asyncio
    async def test_read_resource(self):
        """Test reading a resource."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"content": "resource data"}
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = MCPClient("http://localhost:3000")

            async with client:
                result = await client.read_resource("resource://test")

            assert result == {"content": "resource data"}


class TestMCPClientExceptions:
    """Test MCP client exception classes."""

    def test_mcp_client_error_basic(self):
        """Test basic MCPClientError."""
        error = MCPClientError("Test error")
        assert "Test error" in str(error)
        assert error.message == "Test error"

    def test_mcp_client_error_with_status_code(self):
        """Test MCPClientError with status code."""
        error = MCPClientError("Test error", status_code=404)
        assert "404" in str(error)
        assert error.status_code == 404

    def test_mcp_client_error_with_original_error(self):
        """Test MCPClientError with original error."""
        original = ValueError("Original")
        error = MCPClientError("Test error", original_error=original)
        assert "ValueError" in str(error)
        assert error.original_error is original

    def test_mcp_connection_error(self):
        """Test MCPConnectionError."""
        error = MCPConnectionError("Connection failed")
        assert isinstance(error, MCPClientError)
        assert "Connection failed" in str(error)

    def test_mcp_timeout_error(self):
        """Test MCPTimeoutError."""
        error = MCPTimeoutError("Request timed out")
        assert isinstance(error, MCPClientError)
        assert "timed out" in str(error)

    def test_mcp_tool_error(self):
        """Test MCPToolError."""
        error = MCPToolError("Tool failed")
        assert isinstance(error, MCPClientError)
        assert "Tool failed" in str(error)

    def test_mcp_authentication_error(self):
        """Test MCPAuthenticationError."""
        error = MCPAuthenticationError("Auth failed", status_code=401)
        assert isinstance(error, MCPClientError)
        assert error.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
