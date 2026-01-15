"""
Ollama AI Plugin

Provides AI processing capabilities using local Ollama models
for description generation and content summarization.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from ..base import AIProcessorPlugin, PluginHook

# Import Bookmark for type hints
try:
    from ...core.data_models import Bookmark
except ImportError:
    Bookmark = Any


class OllamaAIPlugin(AIProcessorPlugin):
    """
    Plugin that uses local Ollama for AI processing.

    Provides description generation using locally-running Ollama
    models like llama2, mistral, etc.
    """

    DEFAULT_ENDPOINT = "http://localhost:11434"
    DEFAULT_MODEL = "llama2"
    DEFAULT_TIMEOUT = 60.0

    def __init__(self):
        super().__init__()
        self._endpoint: str = self.DEFAULT_ENDPOINT
        self._model: str = self.DEFAULT_MODEL
        self._timeout: float = self.DEFAULT_TIMEOUT
        self._available: Optional[bool] = None
        self._processed_count: int = 0
        self._total_time: float = 0.0
        self._system_prompt: str = ""

    @property
    def name(self) -> str:
        return "ollama-ai"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "AI processing using local Ollama models for description generation"

    @property
    def author(self) -> str:
        return "Bookmark Processor Team"

    @property
    def provides(self) -> List[str]:
        return ["ai_processing", "description_generation", "summarization"]

    @property
    def hooks(self) -> List[PluginHook]:
        return [
            PluginHook.PRE_AI_PROCESS,
            PluginHook.POST_AI_PROCESS,
            PluginHook.AI_FALLBACK,
        ]

    def on_load(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration."""
        super().on_load(config)

        self._endpoint = config.get("endpoint", self.DEFAULT_ENDPOINT)
        self._model = config.get("model", self.DEFAULT_MODEL)
        self._timeout = config.get("timeout", self.DEFAULT_TIMEOUT)

        # Custom system prompt
        self._system_prompt = config.get(
            "system_prompt",
            "You are a helpful assistant that generates concise, informative "
            "descriptions for web bookmarks. Keep descriptions under 150 characters.",
        )

        # Temperature for generation
        self._temperature = config.get("temperature", 0.7)

        # Max tokens
        self._max_tokens = config.get("max_tokens", 150)

        # Reset availability check
        self._available = None

        self._logger.info(
            f"Ollama AI plugin loaded (endpoint={self._endpoint}, model={self._model})"
        )

    def generate_description(
        self, bookmark: "Bookmark", content: str
    ) -> str:
        """
        Generate a description for a bookmark using Ollama.

        Args:
            bookmark: The bookmark to process
            content: Fetched content from the URL

        Returns:
            Generated description string
        """
        if not self.is_available():
            raise RuntimeError("Ollama is not available")

        start_time = time.time()

        try:
            # Prepare the prompt
            title = bookmark.get_effective_title() if hasattr(bookmark, 'get_effective_title') else str(bookmark)
            url = bookmark.url if hasattr(bookmark, 'url') else str(bookmark)

            # Truncate content for prompt
            content_preview = content[:1500] if content else ""

            prompt = self._build_prompt(title, url, content_preview)

            # Call Ollama API
            response = self._call_ollama(prompt)

            # Extract description from response
            description = self._extract_description(response)

            self._processed_count += 1
            self._total_time += time.time() - start_time

            return description

        except Exception as e:
            self._logger.error(f"Error generating description: {e}")
            raise

    def is_available(self) -> bool:
        """
        Check if Ollama is available.

        Returns:
            True if Ollama server is reachable and model is available
        """
        if not REQUESTS_AVAILABLE:
            return False

        if self._available is not None:
            return self._available

        try:
            # Check if Ollama is running
            response = requests.get(
                f"{self._endpoint}/api/tags",
                timeout=5.0,
            )

            if response.status_code != 200:
                self._available = False
                return False

            # Check if model is available
            data = response.json()
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]

            self._available = self._model in models

            if not self._available:
                self._logger.warning(
                    f"Model {self._model} not found. Available models: {models}"
                )

            return self._available

        except requests.RequestException as e:
            self._logger.debug(f"Ollama not available: {e}")
            self._available = False
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the configured model."""
        return {
            "name": self.name,
            "version": self.version,
            "model": self._model,
            "endpoint": self._endpoint,
            "available": self.is_available(),
            "processed_count": self._processed_count,
            "average_time": (
                self._total_time / self._processed_count
                if self._processed_count > 0
                else 0
            ),
        }

    def estimate_cost(self, content_length: int) -> float:
        """
        Estimate cost of processing.

        Local Ollama is free, so always returns 0.
        """
        return 0.0

    def _build_prompt(self, title: str, url: str, content: str) -> str:
        """Build the prompt for description generation."""
        return f"""Based on the following webpage information, write a concise description (under 150 characters) that summarizes what this page is about.

Title: {title}
URL: {url}

Content preview:
{content}

Description:"""

    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """
        Call the Ollama API.

        Args:
            prompt: The prompt to send

        Returns:
            API response data
        """
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library not available")

        response = requests.post(
            f"{self._endpoint}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "system": self._system_prompt,
                "stream": False,
                "options": {
                    "temperature": self._temperature,
                    "num_predict": self._max_tokens,
                },
            },
            timeout=self._timeout,
        )

        response.raise_for_status()
        return response.json()

    def _extract_description(self, response: Dict[str, Any]) -> str:
        """Extract and clean description from API response."""
        text = response.get("response", "")

        # Clean up the response
        description = text.strip()

        # Remove any "Description:" prefix if present
        if description.lower().startswith("description:"):
            description = description[12:].strip()

        # Remove quotes if present
        if description.startswith('"') and description.endswith('"'):
            description = description[1:-1]

        # Truncate if too long
        if len(description) > 150:
            # Try to truncate at a sentence boundary
            truncated = description[:147]
            last_period = truncated.rfind(".")
            if last_period > 100:
                description = truncated[: last_period + 1]
            else:
                description = truncated + "..."

        return description

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate plugin configuration."""
        errors = []

        if "endpoint" in config:
            endpoint = config["endpoint"]
            if not endpoint.startswith("http"):
                errors.append("endpoint must be a valid HTTP URL")

        if "model" in config:
            if not isinstance(config["model"], str):
                errors.append("model must be a string")

        if "timeout" in config:
            timeout = config["timeout"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                errors.append("timeout must be a positive number")

        if "temperature" in config:
            temp = config["temperature"]
            if not isinstance(temp, (int, float)) or not 0 <= temp <= 2:
                errors.append("temperature must be between 0 and 2")

        return errors

    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        return {
            "processed_count": self._processed_count,
            "total_time": self._total_time,
            "average_time": (
                self._total_time / self._processed_count
                if self._processed_count > 0
                else 0
            ),
            "model": self._model,
            "endpoint": self._endpoint,
        }

    # Hook methods
    def on_pre_ai_process(
        self, bookmark: "Bookmark", content: str
    ) -> tuple["Bookmark", str]:
        """Called before AI processing."""
        return bookmark, content

    def on_post_ai_process(
        self, bookmark: "Bookmark", description: str
    ) -> str:
        """Called after AI processing."""
        return description

    def on_ai_fallback(
        self, bookmark: "Bookmark", error: Exception
    ) -> Optional[str]:
        """
        Called when AI processing fails.

        Can return a fallback description or None.
        """
        # Simple fallback: use title as description
        if hasattr(bookmark, 'get_effective_title'):
            title = bookmark.get_effective_title()
            if title:
                return title[:150]
        return None

    def list_available_models(self) -> List[str]:
        """List available Ollama models."""
        if not REQUESTS_AVAILABLE:
            return []

        try:
            response = requests.get(
                f"{self._endpoint}/api/tags",
                timeout=5.0,
            )

            if response.status_code == 200:
                data = response.json()
                return [m.get("name", "") for m in data.get("models", [])]

        except requests.RequestException:
            pass

        return []

    def pull_model(self, model_name: Optional[str] = None) -> bool:
        """
        Pull a model from Ollama.

        Args:
            model_name: Model to pull (defaults to configured model)

        Returns:
            True if successful
        """
        if not REQUESTS_AVAILABLE:
            return False

        model = model_name or self._model

        try:
            response = requests.post(
                f"{self._endpoint}/api/pull",
                json={"name": model},
                timeout=300.0,  # Pulling can take a while
            )

            return response.status_code == 200

        except requests.RequestException as e:
            self._logger.error(f"Error pulling model: {e}")
            return False


__all__ = ["OllamaAIPlugin"]
