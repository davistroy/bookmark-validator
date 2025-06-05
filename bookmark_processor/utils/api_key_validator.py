"""
API Key Validation Module

Validates API keys for different providers without exposing them in logs or errors.
"""

import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class APIKeyValidator:
    """Validates API keys for different cloud AI providers."""
    
    # Key patterns for basic format validation
    KEY_PATTERNS = {
        'claude': {
            'pattern': r'^sk-ant-api\d{2}-[\w\-]{40,}$',
            'min_length': 50,
            'prefix': 'sk-ant-api'
        },
        'openai': {
            'pattern': r'^sk-[a-zA-Z0-9]{48,}$',
            'min_length': 50,
            'prefix': 'sk-'
        }
    }
    
    @classmethod
    def validate_format(cls, provider: str, api_key: str) -> Tuple[bool, Optional[str]]:
        """
        Validate API key format for a provider.
        
        Args:
            provider: Provider name (claude or openai)
            api_key: API key to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not api_key:
            return False, "API key is empty"
        
        if provider not in cls.KEY_PATTERNS:
            return False, f"Unknown provider: {provider}"
        
        pattern_info = cls.KEY_PATTERNS[provider]
        
        # Check minimum length
        if len(api_key) < pattern_info['min_length']:
            return False, f"API key too short (minimum {pattern_info['min_length']} characters)"
        
        # Check prefix
        if not api_key.startswith(pattern_info['prefix']):
            return False, f"API key should start with '{pattern_info['prefix']}'"
        
        # Check pattern
        if not re.match(pattern_info['pattern'], api_key):
            return False, "API key format is invalid"
        
        return True, None
    
    @classmethod
    def sanitize_for_logging(cls, api_key: str) -> str:
        """
        Sanitize API key for safe logging.
        
        Args:
            api_key: API key to sanitize
            
        Returns:
            Sanitized version showing only first/last few characters
        """
        if not api_key or len(api_key) < 10:
            return "***"
        
        # Show first 6 and last 3 characters
        return f"{api_key[:6]}...{api_key[-3:]}"
    
    @classmethod
    def validate_configuration(cls, config) -> Tuple[bool, Optional[str]]:
        """
        Validate API configuration including keys.
        
        Args:
            config: Configuration object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        ai_engine = config.get_ai_engine()
        
        # Local AI doesn't need API keys
        if ai_engine == "local":
            return True, None
        
        # Check if API key exists for selected engine
        if not config.has_api_key(ai_engine):
            return False, f"No API key configured for {ai_engine}. Please add {ai_engine}_api_key to your configuration."
        
        # Validate key format
        api_key = config.get_api_key(ai_engine)
        is_valid, error_msg = cls.validate_format(ai_engine, api_key)
        
        if not is_valid:
            sanitized_key = cls.sanitize_for_logging(api_key)
            logger.debug(f"API key validation failed for {ai_engine}: {sanitized_key}")
            return False, f"Invalid {ai_engine} API key format: {error_msg}"
        
        return True, None
    
    @classmethod
    def mask_in_error_message(cls, message: str, api_keys: list) -> str:
        """
        Mask any API keys that might appear in error messages.
        
        Args:
            message: Error message that might contain API keys
            api_keys: List of API keys to mask
            
        Returns:
            Message with API keys masked
        """
        masked_message = message
        for key in api_keys:
            if key and key in masked_message:
                masked_message = masked_message.replace(key, cls.sanitize_for_logging(key))
        return masked_message