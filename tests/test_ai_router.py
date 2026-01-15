"""
Tests for AI Router (Hybrid AI Processing) functionality.

Phase 3.1: Tests for AIRouter, HybridAIConfig, and routing decisions.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from bookmark_processor.core.ai_router import (
    AIRouter,
    HybridAIConfig,
    RoutingDecision,
)
from bookmark_processor.core.content_analyzer import ContentData
from bookmark_processor.core.data_models import Bookmark


@pytest.fixture
def sample_bookmark():
    """Create a sample bookmark for testing."""
    return Bookmark(
        url="https://example.com/article",
        title="Example Article",
        created=datetime.now(),
        tags=["python", "tutorial"],
    )


@pytest.fixture
def sample_content():
    """Create sample content data for testing."""
    return ContentData(
        url="https://example.com/article",
        title="Example Article",
        meta_description="A tutorial about Python",
        word_count=500,
        content_categories=["tutorial", "programming"],
        headings=["Introduction", "Getting Started"],
    )


@pytest.fixture
def simple_content():
    """Create simple content data (low word count)."""
    return ContentData(
        url="https://example.com/simple",
        title="Simple Page",
        meta_description="A simple page",
        word_count=100,
        content_categories=["general"],
        headings=[],
    )


@pytest.fixture
def technical_content():
    """Create technical content data."""
    return ContentData(
        url="https://docs.example.com/api",
        title="API Documentation",
        meta_description="Technical API reference",
        word_count=2000,
        content_categories=["documentation"],
        content_type="documentation",
        headings=["API Reference", "Endpoints", "Authentication"],
    )


class TestHybridAIConfig:
    """Test HybridAIConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HybridAIConfig()

        assert config.mode == "hybrid"
        assert config.escalation_threshold == 0.7
        assert config.budget_cap == 5.00
        assert config.simple_threshold == 200
        assert "documentation" in config.cloud_required_types
        assert "research" in config.cloud_required_types
        assert config.track_costs is True
        assert config.local_model == "facebook/bart-large-cnn"
        assert config.cloud_provider == "claude"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HybridAIConfig(
            mode="cloud",
            escalation_threshold=0.8,
            budget_cap=10.00,
            simple_threshold=100,
            cloud_required_types=["research"],
            cloud_provider="openai",
        )

        assert config.mode == "cloud"
        assert config.escalation_threshold == 0.8
        assert config.budget_cap == 10.00
        assert config.simple_threshold == 100
        assert config.cloud_required_types == ["research"]
        assert config.cloud_provider == "openai"

    def test_to_dict(self):
        """Test config serialization."""
        config = HybridAIConfig(mode="local", budget_cap=3.00)
        data = config.to_dict()

        assert data["mode"] == "local"
        assert data["budget_cap"] == 3.00
        assert "escalation_threshold" in data

    def test_from_dict(self):
        """Test config deserialization."""
        data = {
            "mode": "hybrid",
            "budget_cap": 7.50,
            "escalation_threshold": 0.6,
        }
        config = HybridAIConfig.from_dict(data)

        assert config.mode == "hybrid"
        assert config.budget_cap == 7.50
        assert config.escalation_threshold == 0.6


class TestRoutingDecision:
    """Test RoutingDecision dataclass."""

    def test_routing_decision_creation(self):
        """Test creating a routing decision."""
        decision = RoutingDecision(
            engine="cloud",
            reason="Complex content requires cloud AI",
            confidence=0.9,
            estimated_cost=0.001,
        )

        assert decision.engine == "cloud"
        assert decision.reason == "Complex content requires cloud AI"
        assert decision.confidence == 0.9
        assert decision.estimated_cost == 0.001

    def test_to_dict(self):
        """Test routing decision serialization."""
        decision = RoutingDecision(
            engine="local",
            reason="Simple content",
            confidence=0.95,
        )
        data = decision.to_dict()

        assert data["engine"] == "local"
        assert data["reason"] == "Simple content"
        assert data["confidence"] == 0.95


class TestAIRouter:
    """Test AIRouter class."""

    def test_router_initialization_default(self):
        """Test router initialization with defaults."""
        router = AIRouter()

        assert router.config.mode == "hybrid"
        assert router.local is None
        assert router.cloud is None
        assert router.stats["local_processed"] == 0
        assert router.stats["cloud_processed"] == 0

    def test_router_initialization_custom_config(self):
        """Test router with custom config."""
        config = HybridAIConfig(mode="local", budget_cap=2.00)
        router = AIRouter(config=config)

        assert router.config.mode == "local"
        assert router.config.budget_cap == 2.00

    def test_route_local_mode(self, sample_bookmark, sample_content):
        """Test routing in local-only mode."""
        config = HybridAIConfig(mode="local")
        router = AIRouter(config=config)

        decision = router.route(sample_bookmark, sample_content)

        assert decision.engine == "local"
        assert "Local-only mode" in decision.reason
        assert decision.confidence == 1.0
        assert decision.estimated_cost == 0.0

    def test_route_cloud_mode_without_cloud(self, sample_bookmark, sample_content):
        """Test cloud mode without cloud processor falls back to local."""
        config = HybridAIConfig(mode="cloud")
        router = AIRouter(config=config)  # No cloud processor

        decision = router.route(sample_bookmark, sample_content)

        assert decision.engine == "local"
        assert "not available" in decision.reason

    def test_route_cloud_mode_budget_exhausted(self, sample_bookmark, sample_content):
        """Test cloud mode with exhausted budget falls back to local."""
        config = HybridAIConfig(mode="cloud", budget_cap=0.00)
        mock_cloud = MagicMock()
        mock_cloud.is_available = True

        mock_cost_tracker = MagicMock()
        mock_cost_tracker.session_cost = 1.00

        router = AIRouter(
            cloud_processor=mock_cloud,
            config=config,
            cost_tracker=mock_cost_tracker,
        )

        decision = router.route(sample_bookmark, sample_content)

        assert decision.engine == "local"
        assert "budget exhausted" in decision.reason.lower()

    def test_route_hybrid_simple_content(self, sample_bookmark, simple_content):
        """Test hybrid mode routes simple content to local."""
        config = HybridAIConfig(mode="hybrid", simple_threshold=200)
        mock_cloud = MagicMock()
        mock_cloud.is_available = True

        router = AIRouter(cloud_processor=mock_cloud, config=config)

        decision = router.route(sample_bookmark, simple_content)

        assert decision.engine == "local"
        assert "Simple content" in decision.reason

    def test_route_hybrid_technical_content(self, sample_bookmark, technical_content):
        """Test hybrid mode routes technical content to cloud."""
        config = HybridAIConfig(
            mode="hybrid",
            cloud_required_types=["documentation"],
        )
        mock_cloud = MagicMock()
        mock_cloud.is_available = True

        router = AIRouter(cloud_processor=mock_cloud, config=config)

        decision = router.route(sample_bookmark, technical_content)

        assert decision.engine == "cloud"
        assert "documentation" in decision.reason.lower()

    def test_route_hybrid_low_confidence_escalation(self, sample_bookmark, sample_content):
        """Test hybrid mode escalates low confidence to cloud."""
        config = HybridAIConfig(mode="hybrid", escalation_threshold=0.7)
        mock_cloud = MagicMock()
        mock_cloud.is_available = True

        router = AIRouter(cloud_processor=mock_cloud, config=config)

        # Route with low local confidence
        decision = router.route(sample_bookmark, sample_content, local_confidence=0.5)

        assert decision.engine == "cloud"
        assert "Low local confidence" in decision.reason
        assert router.stats["escalated_to_cloud"] == 1

    def test_route_hybrid_default_to_local(self, sample_bookmark, sample_content):
        """Test hybrid mode defaults to local for normal content."""
        config = HybridAIConfig(mode="hybrid")
        mock_cloud = MagicMock()
        mock_cloud.is_available = True

        router = AIRouter(cloud_processor=mock_cloud, config=config)

        # Normal content, normal confidence
        decision = router.route(sample_bookmark, sample_content, local_confidence=0.8)

        assert decision.engine == "local"
        assert "Default routing" in decision.reason

    def test_get_statistics(self, sample_bookmark):
        """Test statistics retrieval."""
        router = AIRouter()

        # Simulate some processing
        router.stats["local_processed"] = 10
        router.stats["cloud_processed"] = 5
        router.stats["escalated_to_cloud"] = 2
        router.stats["budget_limited"] = 1
        router.stats["total_cost"] = 0.01

        stats = router.get_statistics()

        assert stats["total_processed"] == 15
        assert stats["local_processed"] == 10
        assert stats["cloud_processed"] == 5
        assert stats["local_percentage"] == pytest.approx(66.67, rel=0.1)
        assert stats["cloud_percentage"] == pytest.approx(33.33, rel=0.1)

    def test_reset_statistics(self):
        """Test statistics reset."""
        router = AIRouter()
        router.stats["local_processed"] = 10
        router.stats["cloud_processed"] = 5

        router.reset_statistics()

        assert router.stats["local_processed"] == 0
        assert router.stats["cloud_processed"] == 0
        assert router.stats["total_cost"] == 0.0

    def test_detect_content_type_documentation(self):
        """Test content type detection for documentation."""
        router = AIRouter()

        content = ContentData(
            url="https://docs.example.com",
            title="API Documentation",
            meta_description="API reference manual",
            word_count=500,
        )

        content_type = router._detect_content_type(content)
        assert content_type == "documentation"

    def test_detect_content_type_research(self):
        """Test content type detection for research."""
        router = AIRouter()

        content = ContentData(
            url="https://arxiv.org/paper",
            title="Research Paper on Machine Learning",
            meta_description="A study on neural networks",
            word_count=500,
        )

        content_type = router._detect_content_type(content)
        assert content_type == "research"

    def test_detect_content_type_tutorial(self):
        """Test content type detection for tutorial."""
        router = AIRouter()

        content = ContentData(
            url="https://example.com/tutorial",
            title="Python Tutorial for Beginners",
            meta_description="Learn Python step by step",
            word_count=500,
        )

        content_type = router._detect_content_type(content)
        assert content_type == "tutorial"

    def test_detect_content_type_from_categories(self):
        """Test content type detection from categories."""
        router = AIRouter()

        content = ContentData(
            url="https://example.com",
            title="Something",
            meta_description="Something else",
            word_count=500,
            content_categories=["technical"],
        )

        content_type = router._detect_content_type(content)
        assert content_type == "technical"


class TestAIRouterIntegration:
    """Integration tests for AIRouter with mock processors."""

    def test_process_bookmark_with_local(self, sample_bookmark):
        """Test processing bookmark with local processor."""
        mock_local = MagicMock()
        mock_local.process_bookmark.return_value = sample_bookmark

        config = HybridAIConfig(mode="local")
        router = AIRouter(local_processor=mock_local, config=config)

        result = router.process_bookmark(sample_bookmark)

        assert result == sample_bookmark
        assert router.stats["local_processed"] == 1
        mock_local.process_bookmark.assert_called_once()

    def test_process_batch(self, sample_bookmark):
        """Test batch processing."""
        bookmarks = [
            Bookmark(url=f"https://example.com/{i}", title=f"Article {i}", created=datetime.now())
            for i in range(5)
        ]

        mock_local = MagicMock()
        mock_local.process_bookmark.side_effect = lambda b: b

        config = HybridAIConfig(mode="local")
        router = AIRouter(local_processor=mock_local, config=config)

        results = router.process_batch(bookmarks)

        assert len(results) == 5
        assert router.stats["local_processed"] == 5

    def test_process_batch_with_progress_callback(self, sample_bookmark):
        """Test batch processing with progress callback."""
        bookmarks = [
            Bookmark(url=f"https://example.com/{i}", title=f"Article {i}", created=datetime.now())
            for i in range(3)
        ]

        mock_local = MagicMock()
        mock_local.process_bookmark.side_effect = lambda b: b

        progress_calls = []
        def progress_callback(current, total):
            progress_calls.append((current, total))

        config = HybridAIConfig(mode="local")
        router = AIRouter(local_processor=mock_local, config=config)

        router.process_batch(bookmarks, progress_callback=progress_callback)

        assert len(progress_calls) == 3
        assert progress_calls[0] == (1, 3)
        assert progress_calls[2] == (3, 3)
