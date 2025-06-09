"""
Claude API Client Implementation

This module provides a client for the Anthropic Claude API with bookmark-specific
functionality for generating enhanced descriptions.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from bookmark_processor.core.base_api_client import BaseAPIClient
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.utils.rate_limiter import get_rate_limiter


class ClaudeAPIClient(BaseAPIClient):
    """
    Claude API client for generating bookmark descriptions.

    Uses Claude 3 Haiku for cost-effective description generation.
    """

    # Claude API configuration
    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"
    MODEL = "claude-3-haiku-20240307"  # Cost-effective model

    # Pricing (as of 2024) - in USD per 1K tokens
    COST_PER_1K_INPUT_TOKENS = 0.00025  # $0.25 per million input tokens
    COST_PER_1K_OUTPUT_TOKENS = 0.00125  # $1.25 per million output tokens

    def __init__(self, api_key: str, timeout: int = 30):
        """
        Initialize Claude API client.

        Args:
            api_key: Claude API key
            timeout: Request timeout in seconds
        """
        super().__init__(api_key, timeout)

        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0

        # Rate limiter
        self.rate_limiter = get_rate_limiter("claude")

        self.logger = logging.getLogger(__name__)

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get Claude-specific authentication headers."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
        }

    def _create_bookmark_prompt(
        self,
        bookmark: Bookmark,
        existing_content: Optional[str] = None,
    ) -> str:
        """
        Create an optimized prompt for bookmark description generation.

        Claude-optimized prompt focuses on clear instructions and examples.

        Args:
            bookmark: Bookmark object to process
            existing_content: Existing content to enhance

        Returns:
            Formatted prompt string
        """
        # Gather available information
        title = getattr(bookmark, "title", "") or "Untitled"
        url = getattr(bookmark, "url", "") or "No URL"
        existing_note = getattr(bookmark, "note", "") or ""
        existing_excerpt = getattr(bookmark, "excerpt", "") or ""

        # Use provided content or fallback to bookmark content
        content_to_enhance = existing_content or existing_note or existing_excerpt

        # Extract domain for context
        try:
            from urllib.parse import urlparse

            domain = urlparse(url).netloc or "unknown domain"
        except Exception:
            domain = "unknown domain"

        # Create optimized prompt for Claude
        prompt = (
            f"Create a concise bookmark description (100-150 chars) that captures "
            f"the key value.\n\n"
            f"Title: {title}\n"
            f"Domain: {domain}\n"
            f"Content: {content_to_enhance or 'None'}\n\n"
            f"Focus on: What problem does this solve? What can users learn/do?\n"
            f"Avoid: Generic phrases, \"This is a website about...\"\n"
            f"Style: Direct, actionable, specific\n\n"
            f"Description:"
        )

        return prompt

    def _create_batch_prompt(
        self,
        bookmarks: List[Bookmark],
        existing_content: Optional[List[str]] = None,
    ) -> str:
        """
        Create a prompt for batch processing multiple bookmarks.

        Args:
            bookmarks: List of bookmark objects
            existing_content: List of existing content to enhance

        Returns:
            Formatted batch prompt
        """
        prompt = (
            "Create bookmark descriptions (100-150 chars each) that capture key "
            "value and purpose.\n\n"
            "Focus on: What problem solved? What can users learn/do?\n"
            "Format: Just numbered descriptions, no extra text.\n\n"
            "Bookmarks:\n"
        )

        for i, bookmark in enumerate(bookmarks):
            title = getattr(bookmark, "title", "") or "Untitled"
            url = getattr(bookmark, "url", "") or "No URL"
            existing_note = getattr(bookmark, "note", "") or ""
            existing_excerpt = getattr(bookmark, "excerpt", "") or ""

            # Extract domain for context
            try:
                from urllib.parse import urlparse

                domain = urlparse(url).netloc or "unknown"
            except Exception:
                domain = "unknown"

            # Use provided content or fallback
            content = (
                existing_content[i]
                if existing_content and i < len(existing_content)
                else existing_note or existing_excerpt or "None"
            )

            # Truncate content to save tokens
            content_short = content[:100] + ("..." if len(content) > 100 else "")

            prompt += (
                f"{i + 1}. Title: {title} | Domain: {domain} | "
                f"Content: {content_short}\n"
            )

        prompt += """
Descriptions:"""

        return prompt

    async def generate_description(
        self,
        bookmark: Bookmark,
        existing_content: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate an enhanced description for a bookmark.

        Args:
            bookmark: Bookmark object to process
            existing_content: Existing content to enhance

        Returns:
            Tuple of (enhanced_description, metadata)
        """
        # Wait for rate limit
        if not await self.rate_limiter.acquire(timeout=60):
            raise Exception("Rate limit timeout for Claude API")

        try:
            # Create the prompt
            prompt = self._create_bookmark_prompt(bookmark, existing_content)

            # Prepare request data
            request_data = {
                "model": self.MODEL,
                "max_tokens": 200,  # Allow some buffer for response
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,  # Slightly creative but focused
            }

            # Make the API request
            response = await self._make_request(
                method="POST",
                url=self.BASE_URL,
                data=request_data,
            )

            # Extract response data
            content = response.get("content", [])
            if not content:
                raise ValueError("Empty response from Claude API")

            description = content[0].get("text", "").strip()

            # Track token usage
            usage = response.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.request_count += 1

            # Calculate cost for this request
            input_cost = (input_tokens / 1000) * self.COST_PER_1K_INPUT_TOKENS
            output_cost = (output_tokens / 1000) * self.COST_PER_1K_OUTPUT_TOKENS
            total_cost = input_cost + output_cost

            # Create metadata
            metadata = {
                "provider": "claude",
                "model": self.MODEL,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": total_cost,
                "success": True,
            }

            return description, metadata

        except Exception as e:
            self.logger.error(f"Error generating description with Claude: {e}")

            # Return error metadata
            metadata = {
                "provider": "claude",
                "model": self.MODEL,
                "error": str(e),
                "success": False,
            }

            raise Exception(f"Claude API error: {e}") from e

    async def generate_descriptions_batch(
        self,
        bookmarks: List[Bookmark],
        existing_content: Optional[List[str]] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate enhanced descriptions for a batch of bookmarks.

        Args:
            bookmarks: List of bookmark objects
            existing_content: List of existing content to enhance

        Returns:
            List of tuples (enhanced_description, metadata)
        """
        # For Claude, we'll process individually to better handle rate limits
        # and get more precise error handling
        results = []

        for i, bookmark in enumerate(bookmarks):
            try:
                content = (
                    existing_content[i]
                    if existing_content and i < len(existing_content)
                    else None
                )
                result = await self.generate_description(bookmark, content)
                results.append(result)

            except Exception as e:
                self.logger.error(f"Error processing bookmark {i}: {e}")

                # Add error result
                error_metadata = {
                    "provider": "claude",
                    "model": self.MODEL,
                    "error": str(e),
                    "success": False,
                }
                results.append(("", error_metadata))

        return results

    def get_cost_per_request(self) -> float:
        """
        Get the estimated cost per request for this API.

        Returns:
            Estimated cost in USD per request
        """
        # Estimate based on typical bookmark description requests
        # Assume ~200 input tokens and ~50 output tokens per request
        estimated_input_tokens = 200
        estimated_output_tokens = 50

        input_cost = (estimated_input_tokens / 1000) * self.COST_PER_1K_INPUT_TOKENS
        output_cost = (estimated_output_tokens / 1000) * self.COST_PER_1K_OUTPUT_TOKENS

        return input_cost + output_cost

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """
        Get rate limit information for this API.

        Returns:
            Dictionary with rate limit details
        """
        return {
            "provider": "claude",
            "requests_per_minute": 50,
            "burst_size": 10,
            "model": self.MODEL,
            "status": self.rate_limiter.get_status(),
        }

    def _get_mock_response(
        self, method: str, url: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate Claude-specific mock response for test mode.

        Args:
            method: HTTP method
            url: Request URL
            data: Request data

        Returns:
            Claude-formatted mock response
        """
        return {
            "content": [
                {"text": "Mock Claude AI-generated description for testing purposes."}
            ],
            "usage": {"input_tokens": 150, "output_tokens": 30},
            "model": self.MODEL,
            "test_mode": True,
        }

    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get detailed usage statistics.

        Returns:
            Dictionary with usage data
        """
        total_cost = (
            self.total_input_tokens / 1000
        ) * self.COST_PER_1K_INPUT_TOKENS + (
            self.total_output_tokens / 1000
        ) * self.COST_PER_1K_OUTPUT_TOKENS

        avg_input_tokens = self.total_input_tokens / max(self.request_count, 1)
        avg_output_tokens = self.total_output_tokens / max(self.request_count, 1)
        avg_cost_per_request = total_cost / max(self.request_count, 1)

        return {
            "provider": "claude",
            "model": self.MODEL,
            "total_requests": self.request_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": total_cost,
            "avg_input_tokens_per_request": avg_input_tokens,
            "avg_output_tokens_per_request": avg_output_tokens,
            "avg_cost_per_request": avg_cost_per_request,
            "rate_limit_status": self.rate_limiter.get_status(),
        }

    def reset_statistics(self) -> None:
        """Reset usage statistics."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0
        self.logger.info("Claude API client statistics reset")
