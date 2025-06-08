"""
AI Factory and Selection Logic

This module provides a factory pattern for creating appropriate AI clients
based on configuration and handles fallback logic between different engines.
"""

import logging
from typing import Any, Dict, Optional, Union

from bookmark_processor.config.configuration import Configuration
from bookmark_processor.core.ai_processor import (
    EnhancedAIProcessor as AIProcessor,  # Local AI
)
from bookmark_processor.core.claude_api_client import ClaudeAPIClient
from bookmark_processor.core.openai_api_client import OpenAIAPIClient
from bookmark_processor.utils.error_handler import get_error_handler


class AISelectionError(Exception):
    """Raised when AI engine selection fails."""

    pass


class AIFactory:
    """
    Factory class for creating appropriate AI clients based on configuration.
    """

    # Registry of available AI providers
    PROVIDERS = {
        "local": {
            "name": "Local AI (Transformers)",
            "requires_api_key": False,
            "client_class": None,  # Will be set during initialization
        },
        "claude": {
            "name": "Claude (Anthropic)",
            "requires_api_key": True,
            "client_class": ClaudeAPIClient,
        },
        "openai": {
            "name": "OpenAI (GPT)",
            "requires_api_key": True,
            "client_class": OpenAIAPIClient,
        },
    }

    def __init__(self, config: Configuration):
        """
        Initialize AIFactory with configuration.

        Args:
            config: Configuration object
        """
        self.config = config

    @classmethod
    def get_available_providers(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get list of available AI providers.

        Returns:
            Dictionary of provider information
        """
        return cls.PROVIDERS.copy()

    @classmethod
    def create_client(
        cls,
        provider: str,
        config: Configuration,
        timeout: int = 30,
    ) -> Union[ClaudeAPIClient, OpenAIAPIClient, AIProcessor]:
        """
        Create an AI client for the specified provider.

        Args:
            provider: AI provider name (local, claude, openai)
            config: Configuration object
            timeout: Request timeout for API clients

        Returns:
            Appropriate AI client instance

        Raises:
            AISelectionError: If provider is invalid or configuration is missing
        """
        if provider not in cls.PROVIDERS:
            raise AISelectionError(
                f"Unknown AI provider: {provider}. "
                f"Available providers: {list(cls.PROVIDERS.keys())}"
            )

        provider_info = cls.PROVIDERS[provider]

        # Handle local AI (no API key required)
        if provider == "local":
            return AIProcessor()

        # Handle cloud AI providers (require API keys)
        if provider_info["requires_api_key"]:
            api_key = config.get_api_key(provider)
            if not api_key:
                raise AISelectionError(
                    f"Missing API key for {provider}. "
                    f"Please add '{provider}_api_key' to your user_config.ini file."
                )

            # Validate API key format
            is_valid, error_msg = config.validate_ai_configuration()
            if not is_valid:
                raise AISelectionError(f"Invalid API configuration: {error_msg}")

            # Create the client
            client_class = provider_info["client_class"]
            return client_class(api_key=api_key, timeout=timeout)

        raise AISelectionError(f"Unknown provider type: {provider}")

    @classmethod
    def validate_provider_config(
        cls, provider: str, config: Configuration
    ) -> tuple[bool, Optional[str]]:
        """
        Validate configuration for a specific provider.

        Args:
            provider: AI provider name
            config: Configuration object

        Returns:
            Tuple of (is_valid, error_message)
        """
        if provider not in cls.PROVIDERS:
            return False, f"Unknown provider: {provider}"

        provider_info = cls.PROVIDERS[provider]

        # Local AI doesn't need validation
        if provider == "local":
            return True, None

        # Validate API key requirements
        if provider_info["requires_api_key"]:
            if not config.has_api_key(provider):
                return False, f"Missing API key for {provider}"

            # Validate API key format
            return config.validate_ai_configuration()

        return True, None

    def create_ai_client(
        self, provider: str, timeout: int = 30
    ) -> Union[ClaudeAPIClient, OpenAIAPIClient, AIProcessor]:
        """
        Instance method to create an AI client for the specified provider.
        This method provides backward compatibility with existing tests.

        Args:
            provider: AI provider name (local, claude, openai)
            timeout: Request timeout for API clients

        Returns:
            Appropriate AI client instance

        Raises:
            AISelectionError: If provider is invalid or configuration is missing
        """
        return self.create_client(provider, self.config, timeout)


class AIManager:
    """
    High-level manager for AI operations with fallback support.
    """

    def __init__(
        self,
        primary_provider: str,
        config: Configuration,
        enable_fallback: bool = True,
        fallback_provider: str = "local",
    ):
        """
        Initialize AI manager.

        Args:
            primary_provider: Primary AI provider to use
            config: Configuration object
            enable_fallback: Whether to enable fallback to other providers
            fallback_provider: Provider to fallback to (default: local)
        """
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.config = config
        self.enable_fallback = enable_fallback

        self.primary_client = None
        self.fallback_client = None
        self.current_provider = None

        # Initialize error handler
        self.error_handler = get_error_handler(enable_fallback)

        self.logger = logging.getLogger(__name__)

    async def __aenter__(self) -> "AIManager":
        """Async context manager entry."""
        await self._initialize_clients()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self._cleanup_clients()

    async def _initialize_clients(self) -> None:
        """Initialize AI clients."""
        # Initialize primary client
        try:
            self.primary_client = AIFactory.create_client(
                self.primary_provider, self.config
            )
            self.current_provider = self.primary_provider

            # Initialize client if it has async initialization
            if hasattr(self.primary_client, "__aenter__"):
                await self.primary_client.__aenter__()

            self.logger.info(f"Initialized primary AI client: {self.primary_provider}")

        except Exception as e:
            self.logger.error(
                f"Failed to initialize primary AI client ({self.primary_provider}): {e}"
            )

            if not self.enable_fallback:
                raise AISelectionError(f"Primary AI client initialization failed: {e}")

            # Primary failed, we'll fall back during operation
            self.primary_client = None

        # Initialize fallback client if enabled and different from primary
        if (
            self.enable_fallback
            and self.fallback_provider != self.primary_provider
            and self.fallback_provider in AIFactory.PROVIDERS
        ):
            try:
                self.fallback_client = AIFactory.create_client(
                    self.fallback_provider, self.config
                )

                # Initialize client if it has async initialization
                if hasattr(self.fallback_client, "__aenter__"):
                    await self.fallback_client.__aenter__()

                self.logger.info(
                    f"Initialized fallback AI client: {self.fallback_provider}"
                )

                # If primary failed, use fallback as primary
                if self.primary_client is None:
                    self.primary_client = self.fallback_client
                    self.current_provider = self.fallback_provider
                    self.fallback_client = None

            except Exception as e:
                self.logger.error(
                    f"Failed to initialize fallback AI client ({self.fallback_provider}): {e}"
                )
                self.fallback_client = None

                # If both primary and fallback failed
                if self.primary_client is None:
                    raise AISelectionError(
                        "Both primary and fallback AI client initialization failed"
                    )

    async def _cleanup_clients(self) -> None:
        """Cleanup AI clients."""
        if self.primary_client and hasattr(self.primary_client, "__aexit__"):
            try:
                await self.primary_client.__aexit__(None, None, None)
            except Exception as e:
                self.logger.error(f"Error cleaning up primary client: {e}")

        if self.fallback_client and hasattr(self.fallback_client, "__aexit__"):
            try:
                await self.fallback_client.__aexit__(None, None, None)
            except Exception as e:
                self.logger.error(f"Error cleaning up fallback client: {e}")

    async def generate_description(
        self, bookmark, existing_content: Optional[str] = None
    ) -> tuple[str, Dict[str, Any]]:
        """
        Generate description using the available AI client with comprehensive error handling.

        Args:
            bookmark: Bookmark object to process
            existing_content: Existing content to enhance

        Returns:
            Tuple of (enhanced_description, metadata)
        """
        context = {
            "bookmark_url": getattr(bookmark, "url", "Unknown"),
            "bookmark_title": getattr(bookmark, "title", "Unknown"),
            "primary_provider": self.primary_provider,
            "fallback_provider": self.fallback_provider,
        }

        # Try primary client with retry logic
        if self.primary_client:
            try:

                async def primary_operation():
                    result = await self.primary_client.generate_description(
                        bookmark, existing_content
                    )
                    if not result or not result[0]:  # Validate result
                        raise ValueError("Empty or invalid description returned")
                    return result

                result = await self.error_handler.handle_with_retry(
                    primary_operation, context=context
                )
                return result

            except Exception as e:
                self.logger.warning(
                    f"Primary AI client ({self.current_provider}) failed after retries: {e}"
                )

                # If fallback is enabled, continue to fallback logic
                if not self.enable_fallback:
                    # No fallback, use error handler's fallback strategy
                    return await self.error_handler.handle_bookmark_processing_error(
                        e, bookmark, existing_content, context
                    )

        # Try fallback client with retry logic
        if self.fallback_client and self.enable_fallback:
            try:
                self.logger.info(f"Falling back to {self.fallback_provider}")
                context["current_provider"] = self.fallback_provider

                async def fallback_operation():
                    result = await self.fallback_client.generate_description(
                        bookmark, existing_content
                    )
                    if not result or not result[0]:  # Validate result
                        raise ValueError("Empty or invalid description returned")
                    return result

                result = await self.error_handler.handle_with_retry(
                    fallback_operation, context=context
                )
                return result

            except Exception as e:
                self.logger.warning(
                    f"Fallback AI client ({self.fallback_provider}) failed after retries: {e}"
                )

                # Use error handler's fallback strategy as last resort
                return await self.error_handler.handle_bookmark_processing_error(
                    e, bookmark, existing_content, context
                )

        # No clients available, use error handler's fallback
        no_client_error = Exception("No AI clients available")
        return await self.error_handler.handle_bookmark_processing_error(
            no_client_error, bookmark, existing_content, context
        )

    async def generate_descriptions_batch(self, bookmarks, existing_content=None):
        """
        Generate descriptions for a batch of bookmarks with concurrent processing.

        Args:
            bookmarks: List of bookmark objects
            existing_content: List of existing content to enhance

        Returns:
            List of tuples (enhanced_description, metadata)
        """
        if not bookmarks:
            return []

        # Get optimal concurrency based on current provider and rate limits
        optimal_concurrency = self._get_optimal_concurrency()
        semaphore = asyncio.Semaphore(optimal_concurrency)

        async def process_single_bookmark(i: int, bookmark) -> tuple:
            """Process a single bookmark with rate limiting."""
            async with semaphore:
                content = (
                    existing_content[i]
                    if existing_content and i < len(existing_content)
                    else None
                )
                try:
                    result = await self.generate_description(bookmark, content)
                    return result
                except Exception as e:
                    # Return error result instead of raising
                    from datetime import datetime

                    error_metadata = {
                        "provider": self.get_current_provider(),
                        "error": str(e),
                        "success": False,
                        "timestamp": datetime.now().isoformat(),
                    }
                    return ("", error_metadata)

        # Create tasks for concurrent processing
        tasks = [
            process_single_bookmark(i, bookmark) for i, bookmark in enumerate(bookmarks)
        ]

        # Process concurrently with proper error handling
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions that occurred during processing
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Convert exception to error result
                    from datetime import datetime

                    error_metadata = {
                        "provider": self.get_current_provider(),
                        "error": str(result),
                        "success": False,
                        "timestamp": datetime.now().isoformat(),
                        "bookmark_index": i,
                    }
                    processed_results.append(("", error_metadata))
                else:
                    processed_results.append(result)

            return processed_results

        except Exception as e:
            # Fallback to sequential processing if concurrent processing fails
            logging.warning(
                f"Concurrent batch processing failed, falling back to sequential: {e}"
            )
            results = []
            for i, bookmark in enumerate(bookmarks):
                content = (
                    existing_content[i]
                    if existing_content and i < len(existing_content)
                    else None
                )
                try:
                    result = await self.generate_description(bookmark, content)
                    results.append(result)
                except Exception as bookmark_error:
                    from datetime import datetime

                    error_metadata = {
                        "provider": self.get_current_provider(),
                        "error": str(bookmark_error),
                        "success": False,
                        "timestamp": datetime.now().isoformat(),
                    }
                    results.append(("", error_metadata))
            return results

    def get_current_provider(self) -> str:
        """Get the name of the currently active provider."""
        return self.current_provider or "none"

    def _get_optimal_concurrency(self) -> int:
        """Calculate optimal concurrency based on current provider and rate limits."""
        provider = self.get_current_provider()

        # Base concurrency limits per provider
        base_limits = {
            "local": 20,  # Higher for local processing
            "claude": 5,  # Conservative for Claude API
            "openai": 8,  # Moderate for OpenAI API
            "none": 1,  # Fallback
        }

        base_limit = base_limits.get(provider, 3)

        try:
            # Try to get rate limiter status for dynamic adjustment
            from ..utils.rate_limiter import get_rate_limiter

            rate_limiter = get_rate_limiter(provider)
            status = rate_limiter.get_status()

            # Adjust based on current utilization
            utilization = status.get("utilization_percent", 0)
            if utilization > 80:
                # High utilization, reduce concurrency
                return max(1, base_limit // 2)
            elif utilization < 30:
                # Low utilization, can increase concurrency
                return min(base_limit * 2, 50)
            else:
                # Normal utilization
                return base_limit

        except Exception as e:
            # Fallback to base limits if rate limiter unavailable
            logging.debug(f"Could not get rate limiter status for {provider}: {e}")
            return base_limit

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the current provider setup.

        Returns:
            Dictionary with provider information
        """
        return {
            "primary_provider": self.primary_provider,
            "fallback_provider": self.fallback_provider,
            "current_provider": self.current_provider,
            "fallback_enabled": self.enable_fallback,
            "primary_client_available": self.primary_client is not None,
            "fallback_client_available": self.fallback_client is not None,
        }

    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive usage statistics including error handling.

        Returns:
            Dictionary with usage statistics
        """
        stats = {
            "provider": self.current_provider or "none",
            "total_requests": 0,
            "total_cost_usd": 0.0,
        }

        # Get AI client statistics
        if self.primary_client and hasattr(self.primary_client, "get_usage_statistics"):
            client_stats = self.primary_client.get_usage_statistics()
            stats.update(client_stats)

        # Add error handling statistics
        error_stats = self.error_handler.get_error_statistics()
        stats["error_handling"] = error_stats

        # Add health status
        health_status = self.error_handler.get_health_status()
        stats["health_status"] = health_status

        return stats

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get detailed error statistics.

        Returns:
            Dictionary with error statistics
        """
        return self.error_handler.get_error_statistics()

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status based on recent errors.

        Returns:
            Dictionary with health status
        """
        return self.error_handler.get_health_status()

    def reset_error_statistics(self) -> None:
        """Reset error tracking statistics."""
        self.error_handler.reset_statistics()
