"""
AI Factory and Selection Logic

This module provides a factory pattern for creating appropriate AI clients
based on configuration and handles fallback logic between different engines.
"""

import logging
from typing import Any, Dict, Optional, Union

from bookmark_processor.config.configuration import Configuration
from bookmark_processor.core.ai_processor import AIProcessor  # Local AI
from bookmark_processor.core.claude_api_client import ClaudeAPIClient
from bookmark_processor.core.openai_api_client import OpenAIAPIClient


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
    def validate_provider_config(cls, provider: str, config: Configuration) -> tuple[bool, Optional[str]]:
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
            if hasattr(self.primary_client, '__aenter__'):
                await self.primary_client.__aenter__()
            
            self.logger.info(f"Initialized primary AI client: {self.primary_provider}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize primary AI client ({self.primary_provider}): {e}")
            
            if not self.enable_fallback:
                raise AISelectionError(f"Primary AI client initialization failed: {e}")
            
            # Primary failed, we'll fall back during operation
            self.primary_client = None
        
        # Initialize fallback client if enabled and different from primary
        if (self.enable_fallback and 
            self.fallback_provider != self.primary_provider and
            self.fallback_provider in AIFactory.PROVIDERS):
            try:
                self.fallback_client = AIFactory.create_client(
                    self.fallback_provider, self.config
                )
                
                # Initialize client if it has async initialization
                if hasattr(self.fallback_client, '__aenter__'):
                    await self.fallback_client.__aenter__()
                
                self.logger.info(f"Initialized fallback AI client: {self.fallback_provider}")
                
                # If primary failed, use fallback as primary
                if self.primary_client is None:
                    self.primary_client = self.fallback_client
                    self.current_provider = self.fallback_provider
                    self.fallback_client = None
                
            except Exception as e:
                self.logger.error(f"Failed to initialize fallback AI client ({self.fallback_provider}): {e}")
                self.fallback_client = None
                
                # If both primary and fallback failed
                if self.primary_client is None:
                    raise AISelectionError("Both primary and fallback AI client initialization failed")
    
    async def _cleanup_clients(self) -> None:
        """Cleanup AI clients."""
        if self.primary_client and hasattr(self.primary_client, '__aexit__'):
            try:
                await self.primary_client.__aexit__(None, None, None)
            except Exception as e:
                self.logger.error(f"Error cleaning up primary client: {e}")
        
        if self.fallback_client and hasattr(self.fallback_client, '__aexit__'):
            try:
                await self.fallback_client.__aexit__(None, None, None)
            except Exception as e:
                self.logger.error(f"Error cleaning up fallback client: {e}")
    
    async def generate_description(self, bookmark, existing_content: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
        """
        Generate description using the available AI client.
        
        Args:
            bookmark: Bookmark object to process
            existing_content: Existing content to enhance
            
        Returns:
            Tuple of (enhanced_description, metadata)
        """
        last_error = None
        
        # Try primary client
        if self.primary_client:
            try:
                result = await self.primary_client.generate_description(bookmark, existing_content)
                if result and result[0]:  # Check if we got a valid description
                    return result
            except Exception as e:
                self.logger.warning(f"Primary AI client ({self.current_provider}) failed: {e}")
                last_error = e
        
        # Try fallback client
        if self.fallback_client and self.enable_fallback:
            try:
                self.logger.info(f"Falling back to {self.fallback_provider}")
                result = await self.fallback_client.generate_description(bookmark, existing_content)
                if result and result[0]:  # Check if we got a valid description
                    return result
            except Exception as e:
                self.logger.warning(f"Fallback AI client ({self.fallback_provider}) failed: {e}")
                last_error = e
        
        # If all AI fails, create a basic description
        title = getattr(bookmark, 'title', '') or 'Untitled'
        basic_description = f"Bookmark for {title}"
        
        metadata = {
            "provider": "fallback",
            "error": str(last_error) if last_error else "No AI clients available",
            "success": False,
        }
        
        return basic_description, metadata
    
    async def generate_descriptions_batch(self, bookmarks, existing_content=None):
        """
        Generate descriptions for a batch of bookmarks.
        
        Args:
            bookmarks: List of bookmark objects
            existing_content: List of existing content to enhance
            
        Returns:
            List of tuples (enhanced_description, metadata)
        """
        # For now, process individually
        # TODO: Implement true batch processing for supported providers
        results = []
        
        for i, bookmark in enumerate(bookmarks):
            content = existing_content[i] if existing_content and i < len(existing_content) else None
            result = await self.generate_description(bookmark, content)
            results.append(result)
        
        return results
    
    def get_current_provider(self) -> str:
        """Get the name of the currently active provider."""
        return self.current_provider or "none"
    
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
        Get usage statistics from the active client.
        
        Returns:
            Dictionary with usage statistics
        """
        if self.primary_client and hasattr(self.primary_client, 'get_usage_statistics'):
            return self.primary_client.get_usage_statistics()
        
        return {
            "provider": self.current_provider or "none",
            "total_requests": 0,
            "total_cost_usd": 0.0,
        }