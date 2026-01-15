"""
Example Plugins

Built-in example plugins demonstrating the plugin architecture.
"""

from .paywall_detector import PaywallDetectorPlugin
from .ollama_ai import OllamaAIPlugin

__all__ = [
    "PaywallDetectorPlugin",
    "OllamaAIPlugin",
]
