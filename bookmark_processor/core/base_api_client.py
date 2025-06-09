"""
Base API Client for Cloud AI Services

This module provides a common interface for API clients with shared functionality
for HTTP requests, error handling, rate limiting, and resource management.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import httpx

from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.utils.api_key_validator import APIKeyValidator


class APIClientError(Exception):
    """Base exception for API client errors."""

    pass


class RateLimitError(APIClientError):
    """Raised when API rate limits are exceeded."""

    pass


class AuthenticationError(APIClientError):
    """Raised when API authentication fails."""

    pass


class ServiceUnavailableError(APIClientError):
    """Raised when API service is temporarily unavailable."""

    pass


class BaseAPIClient(ABC):
    """
    Abstract base class for cloud AI API clients.

    Provides common functionality for HTTP requests, error handling,
    retry logic, and resource management.
    """

    def __init__(
        self,
        api_key: str,
        timeout: int = 30,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        """
        Initialize the API client.

        Args:
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize HTTP client (will be created in __aenter__)
        self._client: Optional[httpx.AsyncClient] = None

        # Request statistics
        self.request_count = 0
        self.error_count = 0
        self.retry_count = 0

    async def __aenter__(self) -> "BaseAPIClient":
        """Async context manager entry."""
        await self._initialize_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self._cleanup_client()

    async def _initialize_client(self) -> None:
        """Initialize the HTTP client with appropriate configuration."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            verify=True,  # Always verify SSL certificates
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
                keepalive_expiry=30.0,
            ),
        )

    async def _cleanup_client(self) -> None:
        """Clean up HTTP client resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_common_headers(self) -> Dict[str, str]:
        """
        Get common headers for all requests.

        Returns:
            Dictionary of common headers
        """
        return {
            "User-Agent": "BookmarkProcessor/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers.

        Returns:
            Dictionary of authentication headers
        """
        # Default implementation - subclasses should override
        return {}

    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate retry delay using exponential backoff with jitter.

        Args:
            attempt: Current retry attempt number (0-based)

        Returns:
            Delay in seconds
        """
        delay = min(self.base_delay * (2**attempt), self.max_delay)
        # Add jitter to prevent thundering herd
        jitter = delay * 0.1 * (0.5 - asyncio.get_event_loop().time() % 1)
        return max(delay + jitter, 0.1)

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if a request should be retried.

        Args:
            exception: The exception that occurred
            attempt: Current attempt number (0-based)

        Returns:
            True if the request should be retried
        """
        if attempt >= self.max_retries:
            return False

        # Retry on specific HTTP status codes
        if isinstance(exception, httpx.HTTPStatusError):
            status_code = exception.response.status_code
            # Retry on rate limits, server errors, and some client errors
            return status_code in {429, 500, 502, 503, 504, 408, 423}

        # Retry on network errors
        if isinstance(
            exception, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout)
        ):
            return True

        return False

    def _sanitize_error_message(self, message: str) -> str:
        """
        Sanitize error message to remove API keys and sensitive information.

        Args:
            message: Original error message

        Returns:
            Sanitized error message
        """
        return APIKeyValidator.mask_in_error_message(message, [self.api_key])

    def _get_mock_response(
        self, method: str, url: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate mock response for test mode.

        Args:
            method: HTTP method
            url: Request URL
            data: Request data

        Returns:
            Mock response data
        """
        # Default mock response - subclasses should override for specific formats
        return {
            "content": [
                {"text": "Mock AI generated description for testing purposes."}
            ],
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "test_mode": True,
        }

    async def _make_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with retry logic and error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            data: Request data (will be JSON encoded)
            headers: Additional headers

        Returns:
            Response data as dictionary

        Raises:
            APIClientError: On API errors
            RateLimitError: On rate limit exceeded
            AuthenticationError: On authentication failure
            ServiceUnavailableError: On service unavailability
        """
        import os

        # Test mode guard - prevent real API calls during testing
        if os.getenv("BOOKMARK_PROCESSOR_TEST_MODE") == "true":
            return self._get_mock_response(method, url, data)

        if not self._client:
            raise APIClientError("Client not initialized - use async context manager")

        # Prepare headers
        request_headers = self._get_common_headers()
        request_headers.update(self._get_auth_headers())
        if headers:
            request_headers.update(headers)

        attempt = 0
        while attempt <= self.max_retries:
            try:
                self.request_count += 1

                # Make the request
                response = await self._client.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=request_headers,
                )

                # Handle HTTP errors
                if response.status_code == 401:
                    raise AuthenticationError("Invalid API key or unauthorized access")
                elif response.status_code == 429:
                    self.retry_count += 1
                    raise RateLimitError("Rate limit exceeded")
                elif response.status_code >= 500:
                    raise ServiceUnavailableError(
                        f"Service unavailable: {response.status_code}"
                    )

                response.raise_for_status()

                # Parse JSON response
                try:
                    return response.json()
                except ValueError as e:
                    raise APIClientError(f"Invalid JSON response: {e}")

            except Exception as e:
                self.error_count += 1

                # Check if we should retry
                if self._should_retry(e, attempt):
                    delay = self._calculate_retry_delay(attempt)
                    sanitized_msg = self._sanitize_error_message(str(e))
                    self.logger.warning(
                        f"Request failed (attempt {attempt + 1}/"
                        f"{self.max_retries + 1}): {sanitized_msg}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue

                # Log error and re-raise
                sanitized_msg = self._sanitize_error_message(str(e))
                self.logger.error(f"Request failed permanently: {sanitized_msg}")

                # Convert to appropriate exception type
                if isinstance(e, httpx.HTTPStatusError):
                    if e.response.status_code == 401:
                        raise AuthenticationError(sanitized_msg)
                    elif e.response.status_code == 429:
                        raise RateLimitError(sanitized_msg)
                    elif e.response.status_code >= 500:
                        raise ServiceUnavailableError(sanitized_msg)
                    else:
                        raise APIClientError(sanitized_msg)
                elif isinstance(
                    e, (RateLimitError, AuthenticationError, ServiceUnavailableError)
                ):
                    raise  # Re-raise our custom exceptions
                else:
                    raise APIClientError(sanitized_msg)

    def get_statistics(self) -> Dict[str, int]:
        """
        Get client statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "success_rate": (
                (self.request_count - self.error_count)
                / max(self.request_count, 1)
                * 100
            ),
        }

    # Abstract methods that subclasses must implement

    @abstractmethod
    async def generate_description(
        self,
        bookmark: Bookmark,
        existing_content: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate an enhanced description for a bookmark.

        Args:
            bookmark: Bookmark object to process
            existing_content: Existing content to enhance (note/excerpt)

        Returns:
            Tuple of (enhanced_description, metadata)
        """
        pass

    @abstractmethod
    async def generate_descriptions_batch(
        self,
        bookmarks: List[Bookmark],
        existing_content: Optional[List[str]] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate enhanced descriptions for a batch of bookmarks.

        Args:
            bookmarks: List of bookmark objects to process
            existing_content: List of existing content to enhance

        Returns:
            List of tuples (enhanced_description, metadata)
        """
        pass

    @abstractmethod
    def get_cost_per_request(self) -> float:
        """
        Get the estimated cost per request for this API.

        Returns:
            Cost in USD per request
        """
        pass

    @abstractmethod
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """
        Get rate limit information for this API.

        Returns:
            Dictionary with rate limit details
        """
        pass
