"""
OpenAI API Client Implementation

This module provides a client for the OpenAI API with bookmark-specific
functionality for generating enhanced descriptions using GPT models.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from bookmark_processor.core.base_api_client import BaseAPIClient
from bookmark_processor.core.data_models import Bookmark
from bookmark_processor.utils.rate_limiter import get_rate_limiter


class OpenAIAPIClient(BaseAPIClient):
    """
    OpenAI API client for generating bookmark descriptions.

    Uses GPT-3.5-turbo for cost-effective description generation.
    """

    # OpenAI API configuration
    BASE_URL = "https://api.openai.com/v1/chat/completions"
    MODEL = "gpt-3.5-turbo"  # Cost-effective model

    # Pricing (as of 2024) - in USD per 1K tokens
    COST_PER_1K_INPUT_TOKENS = 0.0015  # $1.50 per million input tokens
    COST_PER_1K_OUTPUT_TOKENS = 0.002  # $2.00 per million output tokens

    def __init__(self, api_key: str, timeout: int = 30):
        """
        Initialize OpenAI API client.

        Args:
            api_key: OpenAI API key
            timeout: Request timeout in seconds
        """
        super().__init__(api_key, timeout)

        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0

        # Rate limiter
        self.rate_limiter = get_rate_limiter("openai")

        self.logger = logging.getLogger(__name__)

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get OpenAI-specific authentication headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    def _create_bookmark_prompt(
        self,
        bookmark: Bookmark,
        existing_content: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Create optimized messages for bookmark description generation.

        Args:
            bookmark: Bookmark object to process
            existing_content: Existing content to enhance

        Returns:
            List of message dictionaries for OpenAI chat completion
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
        except:
            domain = "unknown domain"

        # Optimized system message for GPT-3.5-turbo
        system_message = {
            "role": "system",
            "content": "Create concise bookmark descriptions (100-150 chars). Focus on value and purpose. Avoid generic phrases.",
        }

        # Compact user message to save tokens
        user_content = f"""Title: {title}
Domain: {domain}
Content: {content_to_enhance or 'None'}

What problem does this solve? What can users learn/do?
Description:"""

        user_message = {"role": "user", "content": user_content}

        return [system_message, user_message]

    def _create_batch_messages(
        self,
        bookmarks: List[Bookmark],
        existing_content: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Create messages for batch processing multiple bookmarks.

        Args:
            bookmarks: List of bookmark objects
            existing_content: List of existing content to enhance

        Returns:
            List of message dictionaries for OpenAI chat completion
        """
        # Optimized system message for batch processing
        system_message = {
            "role": "system",
            "content": "Create bookmark descriptions (100-150 chars each). Focus on value/purpose. Format: numbered list only.",
        }

        # Build compact user message
        user_content = "Bookmarks:\n"

        for i, bookmark in enumerate(bookmarks):
            title = getattr(bookmark, "title", "") or "Untitled"
            url = getattr(bookmark, "url", "") or "No URL"
            existing_note = getattr(bookmark, "note", "") or ""
            existing_excerpt = getattr(bookmark, "excerpt", "") or ""

            # Extract domain for context
            try:
                from urllib.parse import urlparse

                domain = urlparse(url).netloc or "unknown"
            except:
                domain = "unknown"

            # Use provided content or fallback
            content = (
                existing_content[i]
                if existing_content and i < len(existing_content)
                else existing_note or existing_excerpt or "None"
            )

            # Truncate content to save tokens
            content_short = content[:80] + ("..." if len(content) > 80 else "")

            user_content += f"{i + 1}. {title} | {domain} | {content_short}\n"

        user_content += "\nDescriptions:"

        user_message = {"role": "user", "content": user_content}

        return [system_message, user_message]

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
            raise Exception("Rate limit timeout for OpenAI API")

        try:
            # Create the messages
            messages = self._create_bookmark_prompt(bookmark, existing_content)

            # Prepare request data
            request_data = {
                "model": self.MODEL,
                "messages": messages,
                "max_tokens": 200,  # Allow some buffer for response
                "temperature": 0.3,  # Slightly creative but focused
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            }

            # Make the API request
            response = await self._make_request(
                method="POST",
                url=self.BASE_URL,
                data=request_data,
            )

            # Extract response data
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("Empty response from OpenAI API")

            message = choices[0].get("message", {})
            description = message.get("content", "").strip()

            # Track token usage
            usage = response.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.request_count += 1

            # Calculate cost for this request
            input_cost = (input_tokens / 1000) * self.COST_PER_1K_INPUT_TOKENS
            output_cost = (output_tokens / 1000) * self.COST_PER_1K_OUTPUT_TOKENS
            total_cost = input_cost + output_cost

            # Create metadata
            metadata = {
                "provider": "openai",
                "model": self.MODEL,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": total_cost,
                "success": True,
            }

            return description, metadata

        except Exception as e:
            self.logger.error(f"Error generating description with OpenAI: {e}")

            # Return error metadata
            metadata = {
                "provider": "openai",
                "model": self.MODEL,
                "error": str(e),
                "success": False,
            }

            raise Exception(f"OpenAI API error: {e}") from e

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
        # OpenAI can handle batch processing better than Claude
        # But we'll still limit batch size to avoid token limits
        batch_size = 20  # Process up to 20 bookmarks at once
        results = []

        for i in range(0, len(bookmarks), batch_size):
            batch_bookmarks = bookmarks[i : i + batch_size]
            batch_content = (
                existing_content[i : i + batch_size] if existing_content else None
            )

            try:
                # Wait for rate limit
                if not await self.rate_limiter.acquire(timeout=60):
                    raise Exception("Rate limit timeout for OpenAI API")

                # Create batch messages
                messages = self._create_batch_messages(batch_bookmarks, batch_content)

                # Prepare request data
                request_data = {
                    "model": self.MODEL,
                    "messages": messages,
                    "max_tokens": 300 * len(batch_bookmarks),  # Scale with batch size
                    "temperature": 0.3,
                }

                # Make the API request
                response = await self._make_request(
                    method="POST",
                    url=self.BASE_URL,
                    data=request_data,
                )

                # Extract and parse response
                choices = response.get("choices", [])
                if not choices:
                    raise ValueError("Empty response from OpenAI API")

                content = choices[0].get("message", {}).get("content", "")
                descriptions = self._parse_batch_response(content, len(batch_bookmarks))

                # Track token usage
                usage = response.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.request_count += 1

                # Calculate cost for this batch
                input_cost = (input_tokens / 1000) * self.COST_PER_1K_INPUT_TOKENS
                output_cost = (output_tokens / 1000) * self.COST_PER_1K_OUTPUT_TOKENS
                total_cost = input_cost + output_cost

                # Create results with metadata
                for j, description in enumerate(descriptions):
                    metadata = {
                        "provider": "openai",
                        "model": self.MODEL,
                        "input_tokens": input_tokens
                        // len(batch_bookmarks),  # Approximate
                        "output_tokens": output_tokens // len(batch_bookmarks),
                        "cost_usd": total_cost / len(batch_bookmarks),
                        "success": True,
                        "batch_size": len(batch_bookmarks),
                    }
                    results.append((description, metadata))

            except Exception as e:
                self.logger.error(f"Error processing batch {i//batch_size + 1}: {e}")

                # Add error results for this batch
                for j in range(len(batch_bookmarks)):
                    error_metadata = {
                        "provider": "openai",
                        "model": self.MODEL,
                        "error": str(e),
                        "success": False,
                    }
                    results.append(("", error_metadata))

        return results

    def _parse_batch_response(
        self, response_content: str, expected_count: int
    ) -> List[str]:
        """
        Parse batch response to extract individual descriptions.

        Args:
            response_content: Raw response content
            expected_count: Expected number of descriptions

        Returns:
            List of parsed descriptions
        """
        descriptions = []
        lines = response_content.strip().split("\n")

        for line in lines:
            line = line.strip()
            # Look for numbered responses like "1. Description here"
            if line and (line[0].isdigit() or line.startswith(("1.", "2.", "3."))):
                # Remove the number prefix
                desc = line.split(".", 1)[-1].strip()
                if desc:
                    descriptions.append(desc)

        # If we don't have enough descriptions, pad with empty ones
        while len(descriptions) < expected_count:
            descriptions.append("")

        # If we have too many, truncate
        return descriptions[:expected_count]

    def get_cost_per_request(self) -> float:
        """
        Get the estimated cost per request for this API.

        Returns:
            Estimated cost in USD per request
        """
        # Estimate based on typical bookmark description requests
        # Assume ~250 input tokens and ~60 output tokens per request
        estimated_input_tokens = 250
        estimated_output_tokens = 60

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
            "provider": "openai",
            "requests_per_minute": 60,
            "burst_size": 20,
            "model": self.MODEL,
            "status": self.rate_limiter.get_status(),
        }

    def _get_mock_response(
        self, method: str, url: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate OpenAI-specific mock response for test mode.

        Args:
            method: HTTP method
            url: Request URL
            data: Request data

        Returns:
            OpenAI-formatted mock response
        """
        return {
            "choices": [
                {
                    "message": {
                        "content": "Mock OpenAI-generated description for testing purposes."
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 25,
                "total_tokens": 145,
            },
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
            "provider": "openai",
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
        self.logger.info("OpenAI API client statistics reset")
