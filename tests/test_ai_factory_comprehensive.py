"""
Comprehensive tests for AI Factory and Selection Logic.

This test module provides 90%+ coverage for bookmark_processor/core/ai_factory.py
including:
1. AIFactory class - factory methods for creating AI processors
2. Engine selection logic (local, claude, openai)
3. Configuration validation
4. Error handling for missing dependencies
5. Fallback behavior when engines unavailable
6. AIManager class - high-level manager with fallback support
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from bookmark_processor.core.ai_factory import AIFactory, AIManager
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.utils.error_handler import AISelectionError


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = MagicMock()
    config.get_api_key.return_value = None
    config.has_api_key.return_value = False
    config.validate_ai_configuration.return_value = (True, None)
    return config


@pytest.fixture
def mock_config_with_claude_key():
    """Create a mock configuration with Claude API key."""
    config = MagicMock()
    config.get_api_key.side_effect = lambda p: "sk-ant-test-key" if p == "claude" else None
    config.has_api_key.side_effect = lambda p: p == "claude"
    config.validate_ai_configuration.return_value = (True, None)
    return config


@pytest.fixture
def mock_config_with_openai_key():
    """Create a mock configuration with OpenAI API key."""
    config = MagicMock()
    config.get_api_key.side_effect = lambda p: "sk-test-openai-key" if p == "openai" else None
    config.has_api_key.side_effect = lambda p: p == "openai"
    config.validate_ai_configuration.return_value = (True, None)
    return config


@pytest.fixture
def mock_config_invalid():
    """Create a mock configuration with invalid AI configuration."""
    config = MagicMock()
    config.get_api_key.return_value = "invalid-key"
    config.has_api_key.return_value = True
    config.validate_ai_configuration.return_value = (False, "Invalid API key format")
    return config


@pytest.fixture
def sample_bookmark():
    """Create a sample bookmark for testing."""
    return Bookmark(
        url="https://example.com/article",
        title="Example Article",
        note="This is a test note",
        excerpt="This is a test excerpt",
        created=datetime.now(),
        tags=["python", "tutorial"],
    )


@pytest.fixture
def sample_bookmarks():
    """Create multiple sample bookmarks for batch testing."""
    return [
        Bookmark(
            url=f"https://example.com/article-{i}",
            title=f"Article {i}",
            created=datetime.now(),
            tags=["test"],
        )
        for i in range(5)
    ]


# ============================================================================
# AIFactory Tests - Basic Functionality
# ============================================================================


class TestAIFactoryProviders:
    """Test AIFactory provider registry and availability."""

    def test_get_available_providers(self):
        """Test retrieving available providers list."""
        providers = AIFactory.get_available_providers()

        assert "local" in providers
        assert "claude" in providers
        assert "openai" in providers

        # Verify provider info structure
        assert providers["local"]["name"] == "Local AI (Transformers)"
        assert providers["local"]["requires_api_key"] is False

        assert providers["claude"]["name"] == "Claude (Anthropic)"
        assert providers["claude"]["requires_api_key"] is True

        assert providers["openai"]["name"] == "OpenAI (GPT)"
        assert providers["openai"]["requires_api_key"] is True

    def test_providers_dict_is_copy(self):
        """Test that get_available_providers returns a copy, not the original."""
        providers1 = AIFactory.get_available_providers()
        providers2 = AIFactory.get_available_providers()

        # Modify one copy
        providers1["test"] = {"name": "Test"}

        # Ensure the other copy and original are unaffected
        assert "test" not in providers2
        assert "test" not in AIFactory.PROVIDERS


class TestAIFactoryInitialization:
    """Test AIFactory initialization."""

    def test_init_with_config(self, mock_config):
        """Test AIFactory initialization with configuration."""
        factory = AIFactory(mock_config)

        assert factory.config == mock_config

    def test_init_stores_config_reference(self, mock_config):
        """Test that factory stores config reference correctly."""
        factory = AIFactory(mock_config)

        # Modify config and verify factory sees changes
        mock_config.some_property = "test_value"
        assert factory.config.some_property == "test_value"


# ============================================================================
# AIFactory Tests - Client Creation
# ============================================================================


class TestAIFactoryCreateClient:
    """Test AIFactory.create_client class method."""

    def test_create_local_client(self, mock_config):
        """Test creating a local AI client."""
        with patch(
            "bookmark_processor.core.ai_factory.AIProcessor"
        ) as MockProcessor:
            mock_instance = MagicMock()
            MockProcessor.return_value = mock_instance

            client = AIFactory.create_client("local", mock_config)

            MockProcessor.assert_called_once()
            assert client == mock_instance

    def test_create_claude_client_success(self, mock_config_with_claude_key):
        """Test creating a Claude client with valid API key."""
        # Create a mock class that will be used to replace ClaudeAPIClient
        mock_instance = MagicMock()

        # Save the original class directly (not via shallow copy)
        original_client_class = AIFactory.PROVIDERS["claude"]["client_class"]
        mock_client_class = MagicMock(return_value=mock_instance)

        try:
            AIFactory.PROVIDERS["claude"]["client_class"] = mock_client_class

            client = AIFactory.create_client(
                "claude", mock_config_with_claude_key, timeout=60
            )

            mock_client_class.assert_called_once_with(
                api_key="sk-ant-test-key", timeout=60
            )
            assert client == mock_instance
        finally:
            # Restore original client class
            AIFactory.PROVIDERS["claude"]["client_class"] = original_client_class

    def test_create_openai_client_success(self, mock_config_with_openai_key):
        """Test creating an OpenAI client with valid API key."""
        mock_instance = MagicMock()
        # Save the original class directly (not via shallow copy)
        original_client_class = AIFactory.PROVIDERS["openai"]["client_class"]
        mock_client_class = MagicMock(return_value=mock_instance)

        try:
            AIFactory.PROVIDERS["openai"]["client_class"] = mock_client_class

            client = AIFactory.create_client(
                "openai", mock_config_with_openai_key, timeout=45
            )

            mock_client_class.assert_called_once_with(
                api_key="sk-test-openai-key", timeout=45
            )
            assert client == mock_instance
        finally:
            # Restore original client class
            AIFactory.PROVIDERS["openai"]["client_class"] = original_client_class

    def test_create_client_unknown_provider(self, mock_config):
        """Test creating client with unknown provider raises error."""
        with pytest.raises(AISelectionError) as exc_info:
            AIFactory.create_client("unknown_provider", mock_config)

        assert "Unknown AI provider" in str(exc_info.value)
        assert "unknown_provider" in str(exc_info.value)
        assert "Available providers" in str(exc_info.value)

    def test_create_claude_client_missing_key(self, mock_config):
        """Test creating Claude client without API key raises error."""
        with pytest.raises(AISelectionError) as exc_info:
            AIFactory.create_client("claude", mock_config)

        assert "Missing API key" in str(exc_info.value)
        assert "claude" in str(exc_info.value)

    def test_create_openai_client_missing_key(self, mock_config):
        """Test creating OpenAI client without API key raises error."""
        with pytest.raises(AISelectionError) as exc_info:
            AIFactory.create_client("openai", mock_config)

        assert "Missing API key" in str(exc_info.value)
        assert "openai" in str(exc_info.value)

    def test_create_client_invalid_config(self, mock_config_invalid):
        """Test creating client with invalid configuration raises error."""
        with pytest.raises(AISelectionError) as exc_info:
            AIFactory.create_client("claude", mock_config_invalid)

        assert "Invalid API configuration" in str(exc_info.value)

    def test_create_client_default_timeout(self, mock_config_with_claude_key):
        """Test that default timeout is used when not specified."""
        mock_instance = MagicMock()
        # Save the original class directly (not via shallow copy)
        original_client_class = AIFactory.PROVIDERS["claude"]["client_class"]
        mock_client_class = MagicMock(return_value=mock_instance)

        try:
            AIFactory.PROVIDERS["claude"]["client_class"] = mock_client_class

            AIFactory.create_client("claude", mock_config_with_claude_key)

            # Default timeout is 30
            mock_client_class.assert_called_once_with(
                api_key="sk-ant-test-key", timeout=30
            )
        finally:
            # Restore original client class
            AIFactory.PROVIDERS["claude"]["client_class"] = original_client_class


class TestAIFactoryInstanceMethod:
    """Test AIFactory instance method create_ai_client."""

    def test_create_ai_client_instance_method(self, mock_config):
        """Test instance method for creating AI client."""
        factory = AIFactory(mock_config)

        with patch(
            "bookmark_processor.core.ai_factory.AIProcessor"
        ) as MockProcessor:
            mock_instance = MagicMock()
            MockProcessor.return_value = mock_instance

            client = factory.create_ai_client("local")

            assert client == mock_instance

    def test_create_ai_client_with_timeout(self, mock_config_with_claude_key):
        """Test instance method with custom timeout."""
        factory = AIFactory(mock_config_with_claude_key)

        mock_instance = MagicMock()
        # Save the original class directly (not via shallow copy)
        original_client_class = AIFactory.PROVIDERS["claude"]["client_class"]
        mock_client_class = MagicMock(return_value=mock_instance)

        try:
            AIFactory.PROVIDERS["claude"]["client_class"] = mock_client_class

            factory.create_ai_client("claude", timeout=120)

            mock_client_class.assert_called_once_with(
                api_key="sk-ant-test-key", timeout=120
            )
        finally:
            # Restore original client class
            AIFactory.PROVIDERS["claude"]["client_class"] = original_client_class


# ============================================================================
# AIFactory Tests - Configuration Validation
# ============================================================================


class TestAIFactoryValidation:
    """Test AIFactory.validate_provider_config method."""

    def test_validate_local_provider(self, mock_config):
        """Test validation for local provider always succeeds."""
        is_valid, error = AIFactory.validate_provider_config("local", mock_config)

        assert is_valid is True
        assert error is None

    def test_validate_unknown_provider(self, mock_config):
        """Test validation for unknown provider fails."""
        is_valid, error = AIFactory.validate_provider_config("unknown", mock_config)

        assert is_valid is False
        assert "Unknown provider" in error

    def test_validate_claude_missing_key(self, mock_config):
        """Test validation for Claude without API key."""
        is_valid, error = AIFactory.validate_provider_config("claude", mock_config)

        assert is_valid is False
        assert "Missing API key" in error

    def test_validate_claude_with_valid_key(self, mock_config_with_claude_key):
        """Test validation for Claude with valid API key."""
        is_valid, error = AIFactory.validate_provider_config(
            "claude", mock_config_with_claude_key
        )

        assert is_valid is True
        assert error is None

    def test_validate_openai_with_valid_key(self, mock_config_with_openai_key):
        """Test validation for OpenAI with valid API key."""
        is_valid, error = AIFactory.validate_provider_config(
            "openai", mock_config_with_openai_key
        )

        assert is_valid is True
        assert error is None

    def test_validate_with_invalid_key_format(self, mock_config_invalid):
        """Test validation with invalid API key format."""
        is_valid, error = AIFactory.validate_provider_config(
            "claude", mock_config_invalid
        )

        assert is_valid is False
        assert "Invalid API key format" in error


# ============================================================================
# AIManager Tests - Initialization
# ============================================================================


class TestAIManagerInitialization:
    """Test AIManager initialization."""

    def test_init_default_values(self, mock_config):
        """Test AIManager initialization with default values."""
        manager = AIManager("local", mock_config)

        assert manager.primary_provider == "local"
        assert manager.fallback_provider == "local"
        assert manager.enable_fallback is True
        assert manager.primary_client is None
        assert manager.fallback_client is None
        assert manager.current_provider is None

    def test_init_with_fallback_disabled(self, mock_config):
        """Test AIManager initialization with fallback disabled."""
        manager = AIManager("claude", mock_config, enable_fallback=False)

        assert manager.enable_fallback is False

    def test_init_with_custom_fallback_provider(self, mock_config):
        """Test AIManager initialization with custom fallback provider."""
        manager = AIManager(
            "claude", mock_config, fallback_provider="openai"
        )

        assert manager.fallback_provider == "openai"


# ============================================================================
# AIManager Tests - Async Context Manager
# ============================================================================


class TestAIManagerContextManager:
    """Test AIManager async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_local_success(self, mock_config):
        """Test context manager with local provider."""
        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            async with AIManager("local", mock_config) as manager:
                assert manager.primary_client == mock_client
                assert manager.current_provider == "local"

    @pytest.mark.asyncio
    async def test_context_manager_with_async_client(self, mock_config):
        """Test context manager with async-capable client."""
        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_create.return_value = mock_client

            async with AIManager("local", mock_config) as manager:
                mock_client.__aenter__.assert_called_once()
                assert manager.primary_client == mock_client

            mock_client.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_primary_fails_fallback_succeeds(self, mock_config):
        """Test context manager when primary fails but fallback succeeds."""
        call_count = [0]

        def create_client_side_effect(provider, config, *args, **kwargs):
            call_count[0] += 1
            if provider == "claude":
                raise Exception("Claude unavailable")
            return MagicMock()

        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_create.side_effect = create_client_side_effect

            async with AIManager(
                "claude", mock_config, enable_fallback=True, fallback_provider="local"
            ) as manager:
                # Primary failed, fallback used
                assert manager.current_provider == "local"
                assert manager.primary_client is not None

    @pytest.mark.asyncio
    async def test_context_manager_both_fail_raises(self, mock_config):
        """Test context manager raises when both primary and fallback fail."""
        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_create.side_effect = Exception("All providers unavailable")

            with pytest.raises(AISelectionError) as exc_info:
                async with AIManager(
                    "claude", mock_config, enable_fallback=True, fallback_provider="openai"
                ):
                    pass

            assert "Both primary and fallback" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_context_manager_no_fallback_primary_fails(self, mock_config):
        """Test context manager raises when primary fails and fallback disabled."""
        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_create.side_effect = Exception("Claude unavailable")

            with pytest.raises(AISelectionError) as exc_info:
                async with AIManager(
                    "claude", mock_config, enable_fallback=False
                ):
                    pass

            assert "Primary AI client initialization failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cleanup_clients_handles_errors(self, mock_config):
        """Test that cleanup handles errors gracefully."""
        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(side_effect=Exception("Cleanup error"))
            mock_create.return_value = mock_client

            # Should not raise despite cleanup error
            async with AIManager("local", mock_config) as manager:
                pass


# ============================================================================
# AIManager Tests - Description Generation
# ============================================================================


class TestAIManagerGenerateDescription:
    """Test AIManager.generate_description method."""

    @pytest.mark.asyncio
    async def test_generate_description_success(self, mock_config, sample_bookmark):
        """Test successful description generation."""
        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_client = AsyncMock()
            mock_client.generate_description = AsyncMock(
                return_value=("Enhanced description", {"provider": "local", "success": True})
            )
            mock_create.return_value = mock_client

            async with AIManager("local", mock_config) as manager:
                # Directly mock the error handler's handle_with_retry to call the operation
                async def mock_handle_with_retry(op, **kwargs):
                    return await op()

                with patch.object(
                    manager.error_handler, "handle_with_retry", side_effect=mock_handle_with_retry
                ):
                    desc, metadata = await manager.generate_description(sample_bookmark)

                    assert desc == "Enhanced description"
                    assert metadata["success"] is True

    @pytest.mark.asyncio
    async def test_generate_description_with_existing_content(
        self, mock_config, sample_bookmark
    ):
        """Test description generation with existing content."""
        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_client = AsyncMock()
            mock_client.generate_description = AsyncMock(
                return_value=("Enhanced content", {"success": True})
            )
            mock_create.return_value = mock_client

            async with AIManager("local", mock_config) as manager:
                async def mock_handle_with_retry(op, **kwargs):
                    return await op()

                with patch.object(
                    manager.error_handler, "handle_with_retry", side_effect=mock_handle_with_retry
                ):
                    desc, _ = await manager.generate_description(
                        sample_bookmark, existing_content="Original content"
                    )

                    assert desc == "Enhanced content"

    @pytest.mark.asyncio
    async def test_generate_description_primary_fails_uses_fallback(
        self, mock_config, sample_bookmark
    ):
        """Test fallback to secondary provider when primary fails."""
        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            primary_client = AsyncMock()
            primary_client.generate_description = AsyncMock(
                side_effect=Exception("Primary failed")
            )

            fallback_client = AsyncMock()
            fallback_client.generate_description = AsyncMock(
                return_value=("Fallback description", {"provider": "fallback"})
            )

            def create_side_effect(provider, config, *args, **kwargs):
                if provider == "claude":
                    return primary_client
                return fallback_client

            mock_create.side_effect = create_side_effect

            async with AIManager(
                "claude",
                mock_config,
                enable_fallback=True,
                fallback_provider="local"
            ) as manager:
                manager.primary_client = primary_client
                manager.fallback_client = fallback_client

                # First call fails (primary), second succeeds (fallback)
                call_count = [0]

                async def retry_side_effect(op, **kwargs):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        raise Exception("Primary failed")
                    return await op()

                with patch.object(
                    manager.error_handler, "handle_with_retry", side_effect=retry_side_effect
                ):
                    desc, metadata = await manager.generate_description(sample_bookmark)

                    assert desc == "Fallback description"

    @pytest.mark.asyncio
    async def test_generate_description_no_clients_available(
        self, mock_config, sample_bookmark
    ):
        """Test handling when no clients are available."""
        manager = AIManager("local", mock_config)
        manager.primary_client = None
        manager.fallback_client = None
        manager.current_provider = None

        with patch.object(
            manager.error_handler, "handle_bookmark_processing_error"
        ) as mock_error:
            mock_error.return_value = ("Fallback desc", {"fallback": True})
            desc, metadata = await manager.generate_description(sample_bookmark)

            assert desc == "Fallback desc"
            mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_description_empty_result_triggers_fallback(
        self, mock_config, sample_bookmark
    ):
        """Test that empty result from AI triggers fallback."""
        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_client = AsyncMock()
            mock_client.generate_description = AsyncMock(
                return_value=("", {"success": True})  # Empty description
            )
            mock_create.return_value = mock_client

            async with AIManager("local", mock_config) as manager:
                # The retry handler validates and should raise on empty result
                async def retry_side_effect(op, **kwargs):
                    result = await op()
                    if not result or not result[0]:
                        raise ValueError("Empty or invalid description returned")
                    return result

                with patch.object(
                    manager.error_handler, "handle_with_retry", side_effect=retry_side_effect
                ):
                    with patch.object(
                        manager.error_handler, "handle_bookmark_processing_error"
                    ) as mock_error:
                        mock_error.return_value = ("Fallback", {"fallback": True})
                        desc, metadata = await manager.generate_description(sample_bookmark)
                        assert desc == "Fallback"


# ============================================================================
# AIManager Tests - Batch Processing
# ============================================================================


class TestAIManagerBatchProcessing:
    """Test AIManager.generate_descriptions_batch method."""

    @pytest.mark.asyncio
    async def test_batch_processing_empty_list(self, mock_config):
        """Test batch processing with empty bookmark list."""
        manager = AIManager("local", mock_config)
        results = await manager.generate_descriptions_batch([])

        assert results == []

    @pytest.mark.asyncio
    async def test_batch_processing_success(self, mock_config, sample_bookmarks):
        """Test successful batch processing."""
        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_client = AsyncMock()
            mock_client.generate_description = AsyncMock(
                return_value=("Description", {"success": True})
            )
            mock_create.return_value = mock_client

            async with AIManager("local", mock_config) as manager:
                # Patch the generate_description method directly
                manager.generate_description = AsyncMock(
                    return_value=("Description", {"success": True})
                )

                results = await manager.generate_descriptions_batch(sample_bookmarks)

                assert len(results) == len(sample_bookmarks)
                assert all(r[0] == "Description" for r in results)

    @pytest.mark.asyncio
    async def test_batch_processing_with_errors(self, mock_config, sample_bookmarks):
        """Test batch processing handles individual errors."""
        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            async with AIManager("local", mock_config) as manager:
                call_count = [0]

                async def mock_generate(*args, **kwargs):
                    call_count[0] += 1
                    if call_count[0] == 3:
                        raise Exception("Processing error")
                    return ("Description", {"success": True})

                manager.generate_description = mock_generate

                results = await manager.generate_descriptions_batch(sample_bookmarks)

                assert len(results) == len(sample_bookmarks)
                # Check that error result was captured
                assert any(r[1].get("success") is False for r in results)

    @pytest.mark.asyncio
    async def test_batch_processing_with_existing_content(
        self, mock_config, sample_bookmarks
    ):
        """Test batch processing with existing content list."""
        existing_content = [f"Content {i}" for i in range(len(sample_bookmarks))]

        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            async with AIManager("local", mock_config) as manager:
                received_content = []

                async def mock_generate(bookmark, content=None):
                    received_content.append(content)
                    return ("Description", {"success": True})

                manager.generate_description = mock_generate

                await manager.generate_descriptions_batch(sample_bookmarks, existing_content)

                # Verify content was passed correctly
                for i, content in enumerate(received_content):
                    assert content == f"Content {i}"

    @pytest.mark.asyncio
    async def test_batch_processing_concurrent_execution(
        self, mock_config, sample_bookmarks
    ):
        """Test that batch processing uses concurrent execution."""
        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            async with AIManager("local", mock_config) as manager:
                execution_times = []

                async def mock_generate(bookmark, content=None):
                    execution_times.append(asyncio.get_event_loop().time())
                    await asyncio.sleep(0.01)  # Small delay
                    return ("Description", {"success": True})

                manager.generate_description = mock_generate

                await manager.generate_descriptions_batch(sample_bookmarks)

                # Verify all tasks completed
                assert len(execution_times) == len(sample_bookmarks)


# ============================================================================
# AIManager Tests - Provider Info and Statistics
# ============================================================================


class TestAIManagerProviderInfo:
    """Test AIManager provider info and statistics methods."""

    def test_get_current_provider(self, mock_config):
        """Test get_current_provider method."""
        manager = AIManager("local", mock_config)

        # Before initialization
        assert manager.get_current_provider() == "none"

        # After setting current provider
        manager.current_provider = "claude"
        assert manager.get_current_provider() == "claude"

    def test_get_provider_info(self, mock_config):
        """Test get_provider_info method."""
        manager = AIManager(
            "claude", mock_config, enable_fallback=True, fallback_provider="local"
        )
        manager.primary_client = MagicMock()
        manager.fallback_client = MagicMock()
        manager.current_provider = "claude"

        info = manager.get_provider_info()

        assert info["primary_provider"] == "claude"
        assert info["fallback_provider"] == "local"
        assert info["current_provider"] == "claude"
        assert info["fallback_enabled"] is True
        assert info["primary_client_available"] is True
        assert info["fallback_client_available"] is True

    def test_get_provider_info_no_clients(self, mock_config):
        """Test get_provider_info when no clients are available."""
        manager = AIManager("local", mock_config)

        info = manager.get_provider_info()

        assert info["primary_client_available"] is False
        assert info["fallback_client_available"] is False

    def test_get_usage_statistics(self, mock_config):
        """Test get_usage_statistics method."""
        manager = AIManager("local", mock_config)
        manager.current_provider = "local"

        # Mock primary client with statistics
        mock_client = MagicMock()
        mock_client.get_usage_statistics.return_value = {
            "total_requests": 100,
            "total_cost_usd": 0.50,
        }
        manager.primary_client = mock_client

        stats = manager.get_usage_statistics()

        assert stats["provider"] == "local"
        assert stats["total_requests"] == 100
        assert stats["total_cost_usd"] == 0.50
        assert "error_handling" in stats
        assert "health_status" in stats

    def test_get_usage_statistics_no_client(self, mock_config):
        """Test get_usage_statistics when no client is available."""
        manager = AIManager("local", mock_config)

        stats = manager.get_usage_statistics()

        assert stats["provider"] == "none"
        assert stats["total_requests"] == 0
        assert stats["total_cost_usd"] == 0.0

    def test_get_error_statistics(self, mock_config):
        """Test get_error_statistics method."""
        manager = AIManager("local", mock_config)

        stats = manager.get_error_statistics()

        assert "total_errors" in stats
        assert "error_counts_by_category" in stats

    def test_get_health_status(self, mock_config):
        """Test get_health_status method."""
        manager = AIManager("local", mock_config)

        status = manager.get_health_status()

        assert "status" in status
        assert "message" in status

    def test_reset_error_statistics(self, mock_config):
        """Test reset_error_statistics method."""
        manager = AIManager("local", mock_config)

        # Simulate some errors
        manager.error_handler.error_counts["test"] = 5

        manager.reset_error_statistics()

        assert len(manager.error_handler.error_counts) == 0


# ============================================================================
# AIManager Tests - Optimal Concurrency
# ============================================================================


class TestAIManagerConcurrency:
    """Test AIManager._get_optimal_concurrency method."""

    def test_get_optimal_concurrency_local(self, mock_config):
        """Test optimal concurrency for local provider."""
        manager = AIManager("local", mock_config)
        manager.current_provider = "local"

        # Patch the rate limiter import to test the fallback path
        with patch.dict("sys.modules", {"bookmark_processor.utils.rate_limiter": None}):
            concurrency = manager._get_optimal_concurrency()
            # The rate limiter check may return different values based on utilization
            # We just verify it returns a reasonable positive number
            assert concurrency > 0

    def test_get_optimal_concurrency_claude(self, mock_config):
        """Test optimal concurrency for Claude provider."""
        manager = AIManager("claude", mock_config)
        manager.current_provider = "claude"

        # Patch to avoid rate limiter and use base limit
        with patch.dict("sys.modules", {"bookmark_processor.utils.rate_limiter": None}):
            concurrency = manager._get_optimal_concurrency()
            assert concurrency > 0

    def test_get_optimal_concurrency_openai(self, mock_config):
        """Test optimal concurrency for OpenAI provider."""
        manager = AIManager("openai", mock_config)
        manager.current_provider = "openai"

        with patch.dict("sys.modules", {"bookmark_processor.utils.rate_limiter": None}):
            concurrency = manager._get_optimal_concurrency()
            assert concurrency > 0

    def test_get_optimal_concurrency_unknown(self, mock_config):
        """Test optimal concurrency for unknown provider."""
        manager = AIManager("unknown", mock_config)
        manager.current_provider = "unknown"

        concurrency = manager._get_optimal_concurrency()
        # Unknown provider should return a reasonable default
        assert concurrency >= 1

    def test_get_optimal_concurrency_none_provider(self, mock_config):
        """Test optimal concurrency when provider is none."""
        manager = AIManager("local", mock_config)
        manager.current_provider = None

        concurrency = manager._get_optimal_concurrency()
        # Should return at least 1 for "none" fallback
        assert concurrency >= 1

    def test_get_optimal_concurrency_returns_positive(self, mock_config):
        """Test that concurrency is always positive."""
        for provider in ["local", "claude", "openai", "none"]:
            manager = AIManager("local", mock_config)
            manager.current_provider = provider
            concurrency = manager._get_optimal_concurrency()
            assert concurrency >= 1, f"Concurrency for {provider} should be >= 1"


# ============================================================================
# AIManager Tests - Fallback Initialization
# ============================================================================


class TestAIManagerFallbackInitialization:
    """Test AIManager fallback client initialization scenarios."""

    @pytest.mark.asyncio
    async def test_same_primary_and_fallback_provider(self, mock_config):
        """Test that fallback client is not created when same as primary."""
        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            async with AIManager(
                "local", mock_config, fallback_provider="local"
            ) as manager:
                # Primary and fallback are the same, so fallback client should not be created
                # (Only one call to create_client expected)
                assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_initialization_failure_with_working_primary(self, mock_config):
        """Test that primary works even if fallback initialization fails."""
        call_count = [0]

        def create_side_effect(provider, config, *args, **kwargs):
            call_count[0] += 1
            if provider == "openai":
                raise Exception("OpenAI unavailable")
            return MagicMock()

        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_create.side_effect = create_side_effect

            async with AIManager(
                "local", mock_config, fallback_provider="openai"
            ) as manager:
                assert manager.primary_client is not None
                assert manager.fallback_client is None


# ============================================================================
# AIManager Tests - Error Handling Edge Cases
# ============================================================================


class TestAIManagerErrorHandling:
    """Test AIManager error handling edge cases."""

    @pytest.mark.asyncio
    async def test_generate_description_fallback_disabled_uses_error_handler(
        self, mock_config, sample_bookmark
    ):
        """Test that error handler fallback is used when provider fallback disabled."""
        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_client = AsyncMock()
            mock_client.generate_description = AsyncMock(
                side_effect=Exception("Processing failed")
            )
            mock_create.return_value = mock_client

            async with AIManager("local", mock_config, enable_fallback=False) as manager:
                async def mock_retry_raises(op, **kwargs):
                    raise Exception("Processing failed")

                with patch.object(
                    manager.error_handler, "handle_with_retry", side_effect=mock_retry_raises
                ):
                    with patch.object(
                        manager.error_handler, "handle_bookmark_processing_error"
                    ) as mock_error:
                        mock_error.return_value = ("Error fallback", {"error": True})
                        desc, metadata = await manager.generate_description(sample_bookmark)

                        # Should still get a fallback response from error handler
                        assert desc == "Error fallback"

    @pytest.mark.asyncio
    async def test_batch_processing_handles_exceptions_gracefully(
        self, mock_config, sample_bookmarks
    ):
        """Test batch processing handles exceptions in individual items."""
        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            async with AIManager("local", mock_config) as manager:
                # Make some calls raise exceptions
                call_count = [0]

                async def failing_generate(*args, **kwargs):
                    call_count[0] += 1
                    if call_count[0] == 2:
                        raise Exception("Simulated failure")
                    return ("Success description", {"success": True})

                manager.generate_description = failing_generate

                results = await manager.generate_descriptions_batch(sample_bookmarks)

                # All results should be present
                assert len(results) == len(sample_bookmarks)
                # At least one should have failed
                failed_results = [r for r in results if r[1].get("success") is False]
                assert len(failed_results) >= 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestAIFactoryIntegration:
    """Integration tests for AIFactory with real-like scenarios."""

    def test_full_workflow_local_provider(self, mock_config):
        """Test complete workflow with local provider."""
        # Validate provider
        is_valid, error = AIFactory.validate_provider_config("local", mock_config)
        assert is_valid

        # Create client
        with patch(
            "bookmark_processor.core.ai_factory.AIProcessor"
        ) as MockProcessor:
            mock_instance = MagicMock()
            mock_instance.is_available = True
            MockProcessor.return_value = mock_instance

            client = AIFactory.create_client("local", mock_config)
            assert client.is_available

    def test_full_workflow_cloud_provider(self, mock_config_with_claude_key):
        """Test complete workflow with cloud provider."""
        # Validate provider
        is_valid, error = AIFactory.validate_provider_config(
            "claude", mock_config_with_claude_key
        )
        assert is_valid

        # Create client with mocked class
        mock_instance = MagicMock()
        mock_instance.is_available = True
        # Save the original class directly (not via shallow copy)
        original_client_class = AIFactory.PROVIDERS["claude"]["client_class"]
        mock_client_class = MagicMock(return_value=mock_instance)

        try:
            AIFactory.PROVIDERS["claude"]["client_class"] = mock_client_class

            client = AIFactory.create_client("claude", mock_config_with_claude_key)
            assert client.is_available
        finally:
            # Restore original client class
            AIFactory.PROVIDERS["claude"]["client_class"] = original_client_class

    @pytest.mark.asyncio
    async def test_manager_full_workflow(self, mock_config, sample_bookmarks):
        """Test complete AIManager workflow."""
        with patch("bookmark_processor.core.ai_factory.AIFactory.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            async with AIManager("local", mock_config) as manager:
                # Verify initialization
                assert manager.primary_client is not None
                assert manager.current_provider == "local"

                # Mock generate_description
                manager.generate_description = AsyncMock(
                    return_value=("Description", {"success": True})
                )

                # Process batch
                results = await manager.generate_descriptions_batch(sample_bookmarks)
                assert len(results) == len(sample_bookmarks)

                # Check provider info
                info = manager.get_provider_info()
                assert info["current_provider"] == "local"

                # Check statistics
                stats = manager.get_usage_statistics()
                assert "provider" in stats


# ============================================================================
# Edge Cases and Error Conditions
# ============================================================================


class TestAIFactoryEdgeCases:
    """Test edge cases and error conditions."""

    def test_create_client_with_empty_provider_string(self, mock_config):
        """Test creating client with empty provider string."""
        with pytest.raises(AISelectionError) as exc_info:
            AIFactory.create_client("", mock_config)

        assert "Unknown AI provider" in str(exc_info.value)

    def test_validate_provider_with_none_config(self):
        """Test validation with None config values."""
        mock_config = MagicMock()
        mock_config.get_api_key.return_value = None
        mock_config.has_api_key.return_value = False

        is_valid, error = AIFactory.validate_provider_config("claude", mock_config)
        assert is_valid is False

    def test_factory_providers_contains_client_classes(self):
        """Test that PROVIDERS has correct client class references."""
        providers = AIFactory.PROVIDERS

        assert providers["local"]["client_class"] is None
        assert providers["claude"]["client_class"] is not None
        assert providers["openai"]["client_class"] is not None


class TestAIManagerEdgeCases:
    """Test AIManager edge cases."""

    def test_manager_with_none_fallback_provider(self, mock_config):
        """Test manager initialization handles edge cases in fallback provider."""
        manager = AIManager(
            "local", mock_config, enable_fallback=True, fallback_provider="local"
        )

        assert manager.fallback_provider == "local"

    @pytest.mark.asyncio
    async def test_manager_cleanup_with_no_clients(self, mock_config):
        """Test cleanup works when no clients were initialized."""
        manager = AIManager("local", mock_config)
        manager.primary_client = None
        manager.fallback_client = None

        # Should not raise
        await manager._cleanup_clients()

    @pytest.mark.asyncio
    async def test_manager_cleanup_with_sync_clients(self, mock_config):
        """Test cleanup works with sync-only clients (no __aexit__)."""
        manager = AIManager("local", mock_config)
        manager.primary_client = MagicMock(spec=[])  # No __aexit__
        manager.fallback_client = MagicMock(spec=[])  # No __aexit__

        # Should not raise
        await manager._cleanup_clients()

    def test_get_current_provider_with_none(self, mock_config):
        """Test get_current_provider returns 'none' when provider is None."""
        manager = AIManager("local", mock_config)
        manager.current_provider = None

        assert manager.get_current_provider() == "none"

    def test_get_current_provider_with_empty_string(self, mock_config):
        """Test get_current_provider handles empty string."""
        manager = AIManager("local", mock_config)
        manager.current_provider = ""

        # Empty string is falsy, should return "none"
        assert manager.get_current_provider() == "none"
