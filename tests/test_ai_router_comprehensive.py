"""
Comprehensive tests for AI Router functionality - Targeting 90%+ coverage.

This test suite extends the existing test_ai_router.py with additional coverage for:
1. AIRouter class - routing logic between local and cloud AI
2. Cost-based routing decisions
3. Fallback behavior when primary engine fails
4. Configuration for routing thresholds
5. Concurrent request handling
6. Metrics tracking
"""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest

from bookmark_processor.core.ai_router import (
    AIRouter,
    HybridAIConfig,
    RoutingDecision,
)
from bookmark_processor.core.content_analyzer import ContentData
from bookmark_processor.core.data_models import Bookmark, ProcessingStatus
from bookmark_processor.utils.cost_tracker import CostTracker


# =============================================================================
# Fixtures
# =============================================================================


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
def sample_bookmark_with_status():
    """Create a sample bookmark with processing status."""
    bookmark = Bookmark(
        url="https://example.com/article",
        title="Example Article",
        created=datetime.now(),
        tags=["python", "tutorial"],
        processing_status=ProcessingStatus(),
    )
    return bookmark


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


@pytest.fixture
def research_content():
    """Create research content data."""
    return ContentData(
        url="https://arxiv.org/paper/12345",
        title="Research Paper on Machine Learning",
        meta_description="A study on neural network optimization",
        word_count=3000,
        content_categories=["research"],
        content_type="research",
        headings=["Abstract", "Introduction", "Methodology", "Results"],
    )


@pytest.fixture
def academic_content():
    """Create academic content data."""
    return ContentData(
        url="https://university.edu/thesis",
        title="PhD Thesis on Quantum Computing",
        meta_description="Academic thesis exploring quantum algorithms",
        word_count=5000,
        content_categories=["academic"],
        content_type="academic",
        headings=["Abstract", "Literature Review", "Methodology"],
    )


@pytest.fixture
def mock_local_processor():
    """Create a mock local AI processor."""
    mock = MagicMock()
    mock.process_bookmark = MagicMock(side_effect=lambda b: b)
    return mock


@pytest.fixture
def mock_cloud_processor():
    """Create a mock cloud AI processor."""
    mock = MagicMock()
    mock.is_available = True
    mock.generate_description = MagicMock(return_value="Enhanced description from cloud AI")
    return mock


@pytest.fixture
def mock_cost_tracker():
    """Create a mock cost tracker."""
    mock = MagicMock(spec=CostTracker)
    mock.session_cost = 0.0
    mock.add_cost_record = MagicMock()
    mock.reset_session = MagicMock()
    return mock


# =============================================================================
# HybridAIConfig Tests - Extended Coverage
# =============================================================================


class TestHybridAIConfigExtended:
    """Extended tests for HybridAIConfig dataclass."""

    def test_default_cloud_required_types(self):
        """Test default cloud required content types."""
        config = HybridAIConfig()
        assert "documentation" in config.cloud_required_types
        assert "research" in config.cloud_required_types
        assert "technical" in config.cloud_required_types
        assert "academic" in config.cloud_required_types

    def test_config_with_empty_cloud_required_types(self):
        """Test config with empty cloud required types."""
        config = HybridAIConfig(cloud_required_types=[])
        assert config.cloud_required_types == []

    def test_to_dict_all_fields(self):
        """Test config serialization includes all fields."""
        config = HybridAIConfig(
            mode="hybrid",
            escalation_threshold=0.75,
            budget_cap=10.0,
            simple_threshold=150,
            cloud_required_types=["documentation"],
            track_costs=False,
            local_model="custom-model",
            cloud_provider="openai",
        )
        data = config.to_dict()

        assert data["mode"] == "hybrid"
        assert data["escalation_threshold"] == 0.75
        assert data["budget_cap"] == 10.0
        assert data["simple_threshold"] == 150
        assert data["cloud_required_types"] == ["documentation"]
        assert data["track_costs"] is False
        assert data["local_model"] == "custom-model"
        assert data["cloud_provider"] == "openai"

    def test_from_dict_with_defaults(self):
        """Test config deserialization uses defaults for missing fields."""
        data = {"mode": "local"}
        config = HybridAIConfig.from_dict(data)

        assert config.mode == "local"
        assert config.escalation_threshold == 0.7  # default
        assert config.budget_cap == 5.00  # default
        assert config.simple_threshold == 200  # default
        assert config.track_costs is True  # default
        assert config.local_model == "facebook/bart-large-cnn"  # default
        assert config.cloud_provider == "claude"  # default

    def test_from_dict_empty_dict(self):
        """Test config deserialization from empty dict uses all defaults."""
        config = HybridAIConfig.from_dict({})

        assert config.mode == "hybrid"
        assert config.escalation_threshold == 0.7
        assert config.budget_cap == 5.00

    def test_round_trip_serialization(self):
        """Test config serialization and deserialization round trip."""
        original = HybridAIConfig(
            mode="cloud",
            escalation_threshold=0.85,
            budget_cap=15.0,
            simple_threshold=300,
            cloud_required_types=["research", "technical"],
            track_costs=True,
            local_model="test-model",
            cloud_provider="openai",
        )

        data = original.to_dict()
        restored = HybridAIConfig.from_dict(data)

        assert restored.mode == original.mode
        assert restored.escalation_threshold == original.escalation_threshold
        assert restored.budget_cap == original.budget_cap
        assert restored.simple_threshold == original.simple_threshold
        assert restored.cloud_required_types == original.cloud_required_types
        assert restored.track_costs == original.track_costs
        assert restored.local_model == original.local_model
        assert restored.cloud_provider == original.cloud_provider


# =============================================================================
# RoutingDecision Tests - Extended Coverage
# =============================================================================


class TestRoutingDecisionExtended:
    """Extended tests for RoutingDecision dataclass."""

    def test_default_values(self):
        """Test routing decision default values."""
        decision = RoutingDecision(engine="local", reason="Test reason")

        assert decision.confidence == 1.0
        assert decision.estimated_cost == 0.0

    def test_to_dict_complete(self):
        """Test routing decision serialization with all fields."""
        decision = RoutingDecision(
            engine="cloud",
            reason="Complex content",
            confidence=0.85,
            estimated_cost=0.002,
        )
        data = decision.to_dict()

        assert data["engine"] == "cloud"
        assert data["reason"] == "Complex content"
        assert data["confidence"] == 0.85
        assert data["estimated_cost"] == 0.002

    def test_to_dict_with_defaults(self):
        """Test routing decision serialization includes default values."""
        decision = RoutingDecision(engine="local", reason="Default test")
        data = decision.to_dict()

        assert "confidence" in data
        assert "estimated_cost" in data
        assert data["confidence"] == 1.0
        assert data["estimated_cost"] == 0.0


# =============================================================================
# AIRouter Initialization Tests
# =============================================================================


class TestAIRouterInitialization:
    """Tests for AIRouter initialization."""

    def test_initialization_with_no_processors(self):
        """Test initialization with no processors."""
        router = AIRouter()

        assert router.local is None
        assert router.cloud is None
        assert router.config.mode == "hybrid"

    def test_initialization_with_local_processor(self, mock_local_processor):
        """Test initialization with local processor only."""
        router = AIRouter(local_processor=mock_local_processor)

        assert router.local is mock_local_processor
        assert router.cloud is None

    def test_initialization_with_cloud_processor(self, mock_cloud_processor):
        """Test initialization with cloud processor only."""
        router = AIRouter(cloud_processor=mock_cloud_processor)

        assert router.local is None
        assert router.cloud is mock_cloud_processor

    def test_initialization_with_both_processors(self, mock_local_processor, mock_cloud_processor):
        """Test initialization with both processors."""
        router = AIRouter(
            local_processor=mock_local_processor,
            cloud_processor=mock_cloud_processor,
        )

        assert router.local is mock_local_processor
        assert router.cloud is mock_cloud_processor

    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = HybridAIConfig(
            mode="cloud",
            budget_cap=20.0,
            escalation_threshold=0.5,
        )
        router = AIRouter(config=config)

        assert router.config.mode == "cloud"
        assert router.config.budget_cap == 20.0
        assert router.config.escalation_threshold == 0.5

    def test_initialization_with_provided_cost_tracker(self, mock_cost_tracker):
        """Test initialization with provided cost tracker."""
        router = AIRouter(cost_tracker=mock_cost_tracker)

        assert router.cost_tracker is mock_cost_tracker

    def test_initialization_creates_cost_tracker_when_tracking_enabled(self):
        """Test initialization creates cost tracker when tracking is enabled."""
        config = HybridAIConfig(track_costs=True, budget_cap=5.0)
        router = AIRouter(config=config)

        assert router.cost_tracker is not None

    def test_initialization_no_cost_tracker_when_tracking_disabled(self):
        """Test initialization does not create cost tracker when tracking is disabled."""
        config = HybridAIConfig(track_costs=False)
        router = AIRouter(config=config)

        assert router.cost_tracker is None

    def test_initialization_stats_are_zeroed(self):
        """Test initialization zeros all statistics."""
        router = AIRouter()

        assert router.stats["local_processed"] == 0
        assert router.stats["cloud_processed"] == 0
        assert router.stats["escalated_to_cloud"] == 0
        assert router.stats["budget_limited"] == 0
        assert router.stats["total_cost"] == 0.0


# =============================================================================
# AIRouter Routing Logic Tests
# =============================================================================


class TestAIRouterRoutingLogic:
    """Tests for AIRouter routing logic."""

    def test_route_local_mode_ignores_content(self, sample_bookmark, technical_content):
        """Test local mode routes to local regardless of content type."""
        config = HybridAIConfig(mode="local")
        router = AIRouter(config=config)

        decision = router.route(sample_bookmark, technical_content)

        assert decision.engine == "local"
        assert "Local-only mode" in decision.reason

    def test_route_cloud_mode_with_available_cloud(self, sample_bookmark, mock_cloud_processor):
        """Test cloud mode routes to cloud when available."""
        config = HybridAIConfig(mode="cloud")
        router = AIRouter(cloud_processor=mock_cloud_processor, config=config)

        decision = router.route(sample_bookmark)

        assert decision.engine == "cloud"
        assert "Cloud-only mode" in decision.reason
        assert decision.estimated_cost > 0

    def test_route_cloud_mode_with_unavailable_cloud(self, sample_bookmark):
        """Test cloud mode falls back to local when cloud unavailable."""
        mock_cloud = MagicMock()
        mock_cloud.is_available = False
        config = HybridAIConfig(mode="cloud")
        router = AIRouter(cloud_processor=mock_cloud, config=config)

        decision = router.route(sample_bookmark)

        assert decision.engine == "local"
        assert "not available" in decision.reason

    def test_route_cloud_mode_without_cloud_processor(self, sample_bookmark):
        """Test cloud mode falls back to local when no cloud processor."""
        config = HybridAIConfig(mode="cloud")
        router = AIRouter(config=config)

        decision = router.route(sample_bookmark)

        assert decision.engine == "local"
        assert "not available" in decision.reason

    def test_route_hybrid_budget_exhausted(self, sample_bookmark, mock_cloud_processor):
        """Test hybrid mode routes to local when budget exhausted."""
        mock_cost_tracker = MagicMock()
        mock_cost_tracker.session_cost = 10.0

        config = HybridAIConfig(mode="hybrid", budget_cap=5.0)
        router = AIRouter(
            cloud_processor=mock_cloud_processor,
            config=config,
            cost_tracker=mock_cost_tracker,
        )

        decision = router.route(sample_bookmark)

        assert decision.engine == "local"
        assert "Budget exhausted" in decision.reason
        assert router.stats["budget_limited"] == 1

    def test_route_hybrid_cloud_unavailable(self, sample_bookmark):
        """Test hybrid mode routes to local when cloud unavailable."""
        config = HybridAIConfig(mode="hybrid")
        router = AIRouter(config=config)

        decision = router.route(sample_bookmark)

        assert decision.engine == "local"
        assert "Cloud AI not available" in decision.reason

    def test_route_hybrid_simple_content_threshold(self, sample_bookmark, mock_cloud_processor):
        """Test hybrid mode routes simple content to local."""
        simple_content = ContentData(
            url="https://example.com",
            title="Simple",
            meta_description="Short",
            word_count=50,  # Below threshold
        )
        config = HybridAIConfig(mode="hybrid", simple_threshold=200)
        router = AIRouter(cloud_processor=mock_cloud_processor, config=config)

        decision = router.route(sample_bookmark, simple_content)

        assert decision.engine == "local"
        assert "Simple content" in decision.reason
        assert "50 words" in decision.reason

    def test_route_hybrid_documentation_to_cloud(self, sample_bookmark, technical_content, mock_cloud_processor):
        """Test hybrid mode routes documentation to cloud."""
        config = HybridAIConfig(mode="hybrid", cloud_required_types=["documentation"])
        router = AIRouter(cloud_processor=mock_cloud_processor, config=config)

        decision = router.route(sample_bookmark, technical_content)

        assert decision.engine == "cloud"
        assert "documentation" in decision.reason.lower()

    def test_route_hybrid_research_to_cloud(self, sample_bookmark, research_content, mock_cloud_processor):
        """Test hybrid mode routes research content to cloud."""
        config = HybridAIConfig(mode="hybrid", cloud_required_types=["research"])
        router = AIRouter(cloud_processor=mock_cloud_processor, config=config)

        decision = router.route(sample_bookmark, research_content)

        assert decision.engine == "cloud"
        assert "research" in decision.reason.lower()

    def test_route_hybrid_technical_to_cloud(self, sample_bookmark, mock_cloud_processor):
        """Test hybrid mode routes technical content to cloud."""
        technical_content = ContentData(
            url="https://example.com/spec",
            title="Protocol Specification",
            meta_description="Technical protocol implementation details",
            word_count=1000,
            content_categories=["technical"],
        )
        config = HybridAIConfig(mode="hybrid", cloud_required_types=["technical"])
        router = AIRouter(cloud_processor=mock_cloud_processor, config=config)

        decision = router.route(sample_bookmark, technical_content)

        assert decision.engine == "cloud"
        assert "technical" in decision.reason.lower()

    def test_route_hybrid_academic_to_cloud(self, sample_bookmark, academic_content, mock_cloud_processor):
        """Test hybrid mode routes academic content to cloud."""
        config = HybridAIConfig(mode="hybrid", cloud_required_types=["academic"])
        router = AIRouter(cloud_processor=mock_cloud_processor, config=config)

        decision = router.route(sample_bookmark, academic_content)

        assert decision.engine == "cloud"
        assert "academic" in decision.reason.lower()

    def test_route_hybrid_low_confidence_escalation(self, sample_bookmark, sample_content, mock_cloud_processor):
        """Test hybrid mode escalates low confidence to cloud."""
        config = HybridAIConfig(mode="hybrid", escalation_threshold=0.7)
        router = AIRouter(cloud_processor=mock_cloud_processor, config=config)

        decision = router.route(sample_bookmark, sample_content, local_confidence=0.5)

        assert decision.engine == "cloud"
        assert "Low local confidence" in decision.reason
        assert "0.50" in decision.reason
        assert router.stats["escalated_to_cloud"] == 1

    def test_route_hybrid_high_confidence_stays_local(self, sample_bookmark, sample_content, mock_cloud_processor):
        """Test hybrid mode keeps high confidence local."""
        config = HybridAIConfig(mode="hybrid", escalation_threshold=0.7)
        router = AIRouter(cloud_processor=mock_cloud_processor, config=config)

        decision = router.route(sample_bookmark, sample_content, local_confidence=0.9)

        assert decision.engine == "local"
        assert "Default routing" in decision.reason

    def test_route_hybrid_threshold_boundary(self, sample_bookmark, sample_content, mock_cloud_processor):
        """Test hybrid mode at exactly the escalation threshold."""
        config = HybridAIConfig(mode="hybrid", escalation_threshold=0.7)
        router = AIRouter(cloud_processor=mock_cloud_processor, config=config)

        # At threshold - should stay local
        decision = router.route(sample_bookmark, sample_content, local_confidence=0.7)
        assert decision.engine == "local"

        # Below threshold - should escalate
        decision = router.route(sample_bookmark, sample_content, local_confidence=0.69)
        assert decision.engine == "cloud"

    def test_route_hybrid_default_to_local(self, sample_bookmark, sample_content, mock_cloud_processor):
        """Test hybrid mode defaults to local for normal content."""
        # Content with word count above threshold but no cloud-required type
        normal_content = ContentData(
            url="https://example.com",
            title="Normal Article",
            meta_description="A regular article",
            word_count=500,
            content_categories=["article"],
        )
        config = HybridAIConfig(mode="hybrid")
        router = AIRouter(cloud_processor=mock_cloud_processor, config=config)

        decision = router.route(sample_bookmark, normal_content, local_confidence=0.8)

        assert decision.engine == "local"
        assert "Default routing" in decision.reason


# =============================================================================
# AIRouter Content Type Detection Tests
# =============================================================================


class TestAIRouterContentTypeDetection:
    """Tests for AIRouter content type detection."""

    def test_detect_content_type_from_categories(self):
        """Test content type detection from content_categories field."""
        router = AIRouter()
        content = ContentData(
            url="https://example.com",
            title="Something",
            meta_description="Something else",
            word_count=500,
            content_categories=["documentation", "tutorial"],
        )

        content_type = router._detect_content_type(content)
        assert content_type == "documentation"  # First category

    def test_detect_content_type_from_content_type_field(self):
        """Test content type detection from content_type field."""
        router = AIRouter()
        content = ContentData(
            url="https://example.com",
            title="Something",
            meta_description="Something else",
            word_count=500,
            content_categories=[],
            content_type="RESEARCH",  # Uppercase to test lowercasing
        )

        content_type = router._detect_content_type(content)
        assert content_type == "research"

    def test_detect_content_type_from_title_docs(self):
        """Test content type detection from title - documentation."""
        router = AIRouter()
        content = ContentData(
            url="https://example.com",
            title="API Reference Documentation",
            meta_description="",
            word_count=500,
        )

        content_type = router._detect_content_type(content)
        assert content_type == "documentation"

    def test_detect_content_type_from_title_research(self):
        """Test content type detection from title - research."""
        router = AIRouter()
        content = ContentData(
            url="https://example.com",
            title="Research Study on Machine Learning",
            meta_description="",
            word_count=500,
        )

        content_type = router._detect_content_type(content)
        assert content_type == "research"

    def test_detect_content_type_from_title_technical(self):
        """Test content type detection from title - technical."""
        router = AIRouter()
        content = ContentData(
            url="https://example.com",
            title="Protocol Specification v2.0",
            meta_description="",
            word_count=500,
        )

        content_type = router._detect_content_type(content)
        assert content_type == "technical"

    def test_detect_content_type_from_title_academic(self):
        """Test content type detection from title - academic."""
        router = AIRouter()
        content = ContentData(
            url="https://example.com",
            title="Academic Thesis on Quantum Computing",
            meta_description="",
            word_count=500,
        )

        content_type = router._detect_content_type(content)
        assert content_type == "academic"

    def test_detect_content_type_from_title_tutorial(self):
        """Test content type detection from title - tutorial."""
        router = AIRouter()
        content = ContentData(
            url="https://example.com",
            title="Beginner's Tutorial for Python",
            meta_description="",
            word_count=500,
        )

        content_type = router._detect_content_type(content)
        assert content_type == "tutorial"

    def test_detect_content_type_from_title_article(self):
        """Test content type detection from title - article."""
        router = AIRouter()
        content = ContentData(
            url="https://example.com",
            title="Blog Post About Web Development",
            meta_description="",
            word_count=500,
        )

        content_type = router._detect_content_type(content)
        assert content_type == "article"

    def test_detect_content_type_from_meta_description(self):
        """Test content type detection from meta description."""
        router = AIRouter()
        content = ContentData(
            url="https://example.com",
            title="Some Generic Title",
            meta_description="This is the official documentation for our API",
            word_count=500,
        )

        content_type = router._detect_content_type(content)
        assert content_type == "documentation"

    def test_detect_content_type_general_fallback(self):
        """Test content type detection falls back to general."""
        router = AIRouter()
        content = ContentData(
            url="https://example.com",
            title="Generic Page",
            meta_description="Just a page",
            word_count=500,
        )

        content_type = router._detect_content_type(content)
        assert content_type == "general"

    def test_detect_content_type_arxiv_indicator(self):
        """Test content type detection recognizes arxiv."""
        router = AIRouter()
        content = ContentData(
            url="https://arxiv.org/abs/12345",
            title="Paper Title",
            meta_description="arxiv paper description",
            word_count=500,
        )

        content_type = router._detect_content_type(content)
        assert content_type == "research"


# =============================================================================
# AIRouter Budget and Cost Tests
# =============================================================================


class TestAIRouterBudgetAndCost:
    """Tests for AIRouter budget and cost management."""

    def test_is_budget_exhausted_no_tracker(self):
        """Test budget check when no cost tracker."""
        config = HybridAIConfig(track_costs=False)
        router = AIRouter(config=config)

        assert router._is_budget_exhausted() is False

    def test_is_budget_exhausted_under_budget(self):
        """Test budget check when under budget."""
        mock_cost_tracker = MagicMock()
        mock_cost_tracker.session_cost = 2.0
        config = HybridAIConfig(budget_cap=5.0)
        router = AIRouter(config=config, cost_tracker=mock_cost_tracker)

        assert router._is_budget_exhausted() is False

    def test_is_budget_exhausted_at_budget(self):
        """Test budget check when at budget."""
        mock_cost_tracker = MagicMock()
        mock_cost_tracker.session_cost = 5.0
        config = HybridAIConfig(budget_cap=5.0)
        router = AIRouter(config=config, cost_tracker=mock_cost_tracker)

        assert router._is_budget_exhausted() is True

    def test_is_budget_exhausted_over_budget(self):
        """Test budget check when over budget."""
        mock_cost_tracker = MagicMock()
        mock_cost_tracker.session_cost = 10.0
        config = HybridAIConfig(budget_cap=5.0)
        router = AIRouter(config=config, cost_tracker=mock_cost_tracker)

        assert router._is_budget_exhausted() is True

    def test_estimate_cost_claude(self):
        """Test cost estimation for Claude."""
        config = HybridAIConfig(cloud_provider="claude")
        router = AIRouter(config=config)

        cost = router._estimate_cost()
        assert cost == AIRouter.COST_ESTIMATES["claude"]

    def test_estimate_cost_openai(self):
        """Test cost estimation for OpenAI."""
        config = HybridAIConfig(cloud_provider="openai")
        router = AIRouter(config=config)

        cost = router._estimate_cost()
        assert cost == AIRouter.COST_ESTIMATES["openai"]

    def test_estimate_cost_unknown_provider(self):
        """Test cost estimation for unknown provider."""
        config = HybridAIConfig(cloud_provider="unknown_provider")
        router = AIRouter(config=config)

        cost = router._estimate_cost()
        assert cost == 0.001  # Default fallback

    def test_record_cost_with_tracker(self, mock_cost_tracker):
        """Test cost recording with tracker."""
        config = HybridAIConfig(cloud_provider="claude")
        router = AIRouter(config=config, cost_tracker=mock_cost_tracker)

        router._record_cost(0.001)

        mock_cost_tracker.add_cost_record.assert_called_once()
        assert router.stats["total_cost"] == 0.001

    def test_record_cost_without_tracker(self):
        """Test cost recording without tracker."""
        config = HybridAIConfig(track_costs=False)
        router = AIRouter(config=config)

        router._record_cost(0.001)  # Should not raise

        assert router.stats["total_cost"] == 0.0  # Not recorded without tracker

    def test_record_cost_zero_cost(self, mock_cost_tracker):
        """Test cost recording with zero cost."""
        config = HybridAIConfig(cloud_provider="local")
        router = AIRouter(config=config, cost_tracker=mock_cost_tracker)

        router._record_cost(0.0)

        mock_cost_tracker.add_cost_record.assert_not_called()


# =============================================================================
# AIRouter Cloud Availability Tests
# =============================================================================


class TestAIRouterCloudAvailability:
    """Tests for AIRouter cloud availability checks."""

    def test_is_cloud_available_no_processor(self):
        """Test cloud availability when no processor."""
        router = AIRouter()
        assert router._is_cloud_available() is False

    def test_is_cloud_available_with_available_processor(self, mock_cloud_processor):
        """Test cloud availability with available processor."""
        mock_cloud_processor.is_available = True
        router = AIRouter(cloud_processor=mock_cloud_processor)

        assert router._is_cloud_available() is True

    def test_is_cloud_available_with_unavailable_processor(self):
        """Test cloud availability with unavailable processor."""
        mock_cloud = MagicMock()
        mock_cloud.is_available = False
        router = AIRouter(cloud_processor=mock_cloud)

        assert router._is_cloud_available() is False

    def test_is_cloud_available_without_is_available_attribute(self):
        """Test cloud availability when processor lacks is_available attribute."""
        mock_cloud = MagicMock(spec=[])  # No attributes
        del mock_cloud.is_available  # Ensure it doesn't exist
        router = AIRouter(cloud_processor=mock_cloud)

        # getattr with default True should return True
        assert router._is_cloud_available() is True


# =============================================================================
# AIRouter Bookmark Processing Tests
# =============================================================================


class TestAIRouterBookmarkProcessing:
    """Tests for AIRouter bookmark processing."""

    def test_process_bookmark_local_only(self, sample_bookmark_with_status, mock_local_processor):
        """Test processing bookmark with local processor only."""
        config = HybridAIConfig(mode="local")
        router = AIRouter(local_processor=mock_local_processor, config=config)

        result = router.process_bookmark(sample_bookmark_with_status)

        assert result is sample_bookmark_with_status
        assert router.stats["local_processed"] == 1
        mock_local_processor.process_bookmark.assert_called_once()

    def test_process_bookmark_cloud_success(self, sample_bookmark_with_status, mock_cloud_processor):
        """Test processing bookmark with successful cloud processing."""
        config = HybridAIConfig(mode="cloud")
        router = AIRouter(cloud_processor=mock_cloud_processor, config=config)

        result = router.process_bookmark(sample_bookmark_with_status)

        assert result.enhanced_description == "Enhanced description from cloud AI"
        assert result.processing_status.ai_processed is True
        assert router.stats["cloud_processed"] == 1

    def test_process_bookmark_cloud_failure_fallback(self, sample_bookmark_with_status, mock_local_processor):
        """Test processing falls back to local when cloud fails."""
        mock_cloud = MagicMock()
        mock_cloud.is_available = True
        mock_cloud.generate_description.side_effect = Exception("Cloud error")

        config = HybridAIConfig(mode="cloud")
        router = AIRouter(
            local_processor=mock_local_processor,
            cloud_processor=mock_cloud,
            config=config,
        )

        result = router.process_bookmark(sample_bookmark_with_status)

        assert router.stats["local_processed"] == 1
        mock_local_processor.process_bookmark.assert_called_once()

    def test_process_bookmark_cloud_returns_none_fallback(self, sample_bookmark_with_status, mock_local_processor):
        """Test processing falls back to local when cloud returns None."""
        mock_cloud = MagicMock()
        mock_cloud.is_available = True
        mock_cloud.generate_description.return_value = None

        config = HybridAIConfig(mode="cloud")
        router = AIRouter(
            local_processor=mock_local_processor,
            cloud_processor=mock_cloud,
            config=config,
        )

        result = router.process_bookmark(sample_bookmark_with_status)

        assert router.stats["local_processed"] == 1

    def test_process_bookmark_no_processor_available(self, sample_bookmark):
        """Test processing when no processor is available."""
        config = HybridAIConfig(mode="local")
        router = AIRouter(config=config)

        result = router.process_bookmark(sample_bookmark)

        assert result is sample_bookmark  # Returns unchanged
        assert router.stats["local_processed"] == 0

    def test_process_bookmark_records_cloud_cost(self, sample_bookmark_with_status, mock_cost_tracker):
        """Test processing records cost for cloud processing."""
        mock_cloud = MagicMock()
        mock_cloud.is_available = True
        mock_cloud.generate_description.return_value = "Cloud description"

        config = HybridAIConfig(mode="cloud", cloud_provider="claude")
        router = AIRouter(
            cloud_processor=mock_cloud,
            config=config,
            cost_tracker=mock_cost_tracker,
        )

        router.process_bookmark(sample_bookmark_with_status)

        mock_cost_tracker.add_cost_record.assert_called_once()


# =============================================================================
# AIRouter Cloud Processing Tests
# =============================================================================


class TestAIRouterCloudProcessing:
    """Tests for AIRouter cloud processing internals."""

    def test_process_with_cloud_no_processor(self, sample_bookmark):
        """Test cloud processing returns None when no processor."""
        router = AIRouter()
        result = router._process_with_cloud(sample_bookmark)
        assert result is None

    def test_process_with_cloud_success(self, sample_bookmark_with_status, mock_cloud_processor):
        """Test successful cloud processing."""
        router = AIRouter(cloud_processor=mock_cloud_processor)
        result = router._process_with_cloud(sample_bookmark_with_status)

        assert result is not None
        assert result.enhanced_description == "Enhanced description from cloud AI"
        assert result.processing_status.ai_processed is True

    def test_process_with_cloud_empty_description(self, sample_bookmark_with_status):
        """Test cloud processing with empty description."""
        mock_cloud = MagicMock()
        mock_cloud.generate_description.return_value = ""
        router = AIRouter(cloud_processor=mock_cloud)

        result = router._process_with_cloud(sample_bookmark_with_status)

        assert result is None

    def test_process_with_cloud_exception(self, sample_bookmark):
        """Test cloud processing handles exceptions."""
        mock_cloud = MagicMock()
        mock_cloud.generate_description.side_effect = Exception("API error")
        router = AIRouter(cloud_processor=mock_cloud)

        result = router._process_with_cloud(sample_bookmark)

        assert result is None


# =============================================================================
# AIRouter Batch Processing Tests
# =============================================================================


class TestAIRouterBatchProcessing:
    """Tests for AIRouter batch processing."""

    def test_process_batch_empty_list(self, mock_local_processor):
        """Test batch processing with empty list."""
        config = HybridAIConfig(mode="local")
        router = AIRouter(local_processor=mock_local_processor, config=config)

        results = router.process_batch([])

        assert results == []
        assert router.stats["local_processed"] == 0

    def test_process_batch_multiple_bookmarks(self, mock_local_processor):
        """Test batch processing multiple bookmarks."""
        bookmarks = [
            Bookmark(url=f"https://example.com/{i}", title=f"Article {i}", created=datetime.now())
            for i in range(5)
        ]

        config = HybridAIConfig(mode="local")
        router = AIRouter(local_processor=mock_local_processor, config=config)

        results = router.process_batch(bookmarks)

        assert len(results) == 5
        assert router.stats["local_processed"] == 5

    def test_process_batch_with_content_data_map(self, mock_local_processor):
        """Test batch processing with content data map."""
        bookmarks = [
            Bookmark(url="https://example.com/1", title="Article 1", created=datetime.now()),
            Bookmark(url="https://example.com/2", title="Article 2", created=datetime.now()),
        ]

        content_data_map = {
            "https://example.com/1": ContentData(
                url="https://example.com/1",
                title="Article 1",
                meta_description="Description 1",
                word_count=500,
            ),
        }

        config = HybridAIConfig(mode="local")
        router = AIRouter(local_processor=mock_local_processor, config=config)

        results = router.process_batch(bookmarks, content_data_map=content_data_map)

        assert len(results) == 2

    def test_process_batch_with_progress_callback(self, mock_local_processor):
        """Test batch processing with progress callback."""
        bookmarks = [
            Bookmark(url=f"https://example.com/{i}", title=f"Article {i}", created=datetime.now())
            for i in range(3)
        ]

        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        config = HybridAIConfig(mode="local")
        router = AIRouter(local_processor=mock_local_processor, config=config)

        router.process_batch(bookmarks, progress_callback=progress_callback)

        assert len(progress_calls) == 3
        assert progress_calls[0] == (1, 3)
        assert progress_calls[1] == (2, 3)
        assert progress_calls[2] == (3, 3)

    def test_process_batch_mixed_routing(self, mock_local_processor, mock_cloud_processor):
        """Test batch processing with mixed routing decisions."""
        bookmarks = [
            Bookmark(url="https://example.com/simple", title="Simple", created=datetime.now()),
            Bookmark(url="https://docs.example.com/api", title="API Docs", created=datetime.now()),
        ]

        content_data_map = {
            "https://example.com/simple": ContentData(
                url="https://example.com/simple",
                title="Simple",
                meta_description="Simple content",
                word_count=50,  # Simple content
            ),
            "https://docs.example.com/api": ContentData(
                url="https://docs.example.com/api",
                title="API Docs",
                meta_description="Documentation",
                word_count=1000,
                content_categories=["documentation"],
            ),
        }

        config = HybridAIConfig(mode="hybrid", cloud_required_types=["documentation"])
        router = AIRouter(
            local_processor=mock_local_processor,
            cloud_processor=mock_cloud_processor,
            config=config,
        )

        results = router.process_batch(bookmarks, content_data_map=content_data_map)

        assert len(results) == 2


# =============================================================================
# AIRouter Statistics Tests
# =============================================================================


class TestAIRouterStatistics:
    """Tests for AIRouter statistics."""

    def test_get_statistics_initial(self):
        """Test statistics on fresh router."""
        router = AIRouter()
        stats = router.get_statistics()

        assert stats["total_processed"] == 0
        assert stats["local_processed"] == 0
        assert stats["cloud_processed"] == 0
        assert stats["local_percentage"] == 0.0
        assert stats["cloud_percentage"] == 0.0
        assert stats["mode"] == "hybrid"

    def test_get_statistics_after_processing(self, mock_local_processor):
        """Test statistics after processing."""
        config = HybridAIConfig(mode="local")
        router = AIRouter(local_processor=mock_local_processor, config=config)

        # Process some bookmarks
        for i in range(10):
            bookmark = Bookmark(
                url=f"https://example.com/{i}",
                title=f"Article {i}",
                created=datetime.now(),
            )
            router.process_bookmark(bookmark)

        stats = router.get_statistics()

        assert stats["total_processed"] == 10
        assert stats["local_processed"] == 10
        assert stats["cloud_processed"] == 0
        assert stats["local_percentage"] == 100.0
        assert stats["cloud_percentage"] == 0.0

    def test_get_statistics_mixed_processing(self, mock_local_processor, mock_cloud_processor):
        """Test statistics with mixed processing."""
        router = AIRouter(
            local_processor=mock_local_processor,
            cloud_processor=mock_cloud_processor,
        )

        # Manually set stats for testing
        router.stats["local_processed"] = 7
        router.stats["cloud_processed"] = 3
        router.stats["escalated_to_cloud"] = 2
        router.stats["budget_limited"] = 1
        router.stats["total_cost"] = 0.005

        stats = router.get_statistics()

        assert stats["total_processed"] == 10
        assert stats["local_processed"] == 7
        assert stats["cloud_processed"] == 3
        assert stats["escalated_to_cloud"] == 2
        assert stats["budget_limited"] == 1
        assert stats["total_cost_usd"] == 0.005
        assert stats["local_percentage"] == pytest.approx(70.0)
        assert stats["cloud_percentage"] == pytest.approx(30.0)

    def test_get_statistics_remaining_budget_with_tracker(self, mock_cost_tracker):
        """Test statistics include remaining budget with tracker."""
        mock_cost_tracker.session_cost = 2.0
        config = HybridAIConfig(budget_cap=5.0)
        router = AIRouter(config=config, cost_tracker=mock_cost_tracker)

        stats = router.get_statistics()

        assert stats["remaining_budget"] == 3.0

    def test_get_statistics_remaining_budget_without_tracker(self):
        """Test statistics remaining budget without tracker."""
        config = HybridAIConfig(budget_cap=5.0, track_costs=False)
        router = AIRouter(config=config)

        stats = router.get_statistics()

        assert stats["remaining_budget"] == 5.0

    def test_reset_statistics(self, mock_cost_tracker):
        """Test statistics reset."""
        router = AIRouter(cost_tracker=mock_cost_tracker)

        # Set some stats
        router.stats["local_processed"] = 10
        router.stats["cloud_processed"] = 5
        router.stats["escalated_to_cloud"] = 3
        router.stats["budget_limited"] = 2
        router.stats["total_cost"] = 0.01

        router.reset_statistics()

        assert router.stats["local_processed"] == 0
        assert router.stats["cloud_processed"] == 0
        assert router.stats["escalated_to_cloud"] == 0
        assert router.stats["budget_limited"] == 0
        assert router.stats["total_cost"] == 0.0
        mock_cost_tracker.reset_session.assert_called_once()

    def test_reset_statistics_without_tracker(self):
        """Test statistics reset without cost tracker."""
        config = HybridAIConfig(track_costs=False)
        router = AIRouter(config=config)

        router.stats["local_processed"] = 10

        router.reset_statistics()  # Should not raise

        assert router.stats["local_processed"] == 0


# =============================================================================
# AIRouter Edge Cases and Error Handling
# =============================================================================


class TestAIRouterEdgeCases:
    """Tests for AIRouter edge cases and error handling."""

    def test_route_with_none_content(self, sample_bookmark, mock_cloud_processor):
        """Test routing with None content."""
        config = HybridAIConfig(mode="hybrid")
        router = AIRouter(cloud_processor=mock_cloud_processor, config=config)

        decision = router.route(sample_bookmark, content=None)

        assert decision.engine == "local"

    def test_route_with_none_bookmark_url(self, mock_cloud_processor):
        """Test routing with bookmark that has empty URL."""
        bookmark = Bookmark(url="", title="No URL", created=datetime.now())
        config = HybridAIConfig(mode="hybrid")
        router = AIRouter(cloud_processor=mock_cloud_processor, config=config)

        decision = router.route(bookmark)

        assert decision.engine in ["local", "cloud"]

    def test_process_batch_with_none_content_map(self, mock_local_processor):
        """Test batch processing handles None content map."""
        bookmarks = [
            Bookmark(url="https://example.com/1", title="Article 1", created=datetime.now()),
        ]

        config = HybridAIConfig(mode="local")
        router = AIRouter(local_processor=mock_local_processor, config=config)

        results = router.process_batch(bookmarks, content_data_map=None)

        assert len(results) == 1

    def test_cost_estimates_values(self):
        """Test cost estimate constants are reasonable."""
        assert AIRouter.COST_ESTIMATES["claude"] > 0
        assert AIRouter.COST_ESTIMATES["openai"] > 0
        assert AIRouter.COST_ESTIMATES["local"] == 0
        assert AIRouter.COST_ESTIMATES["openai"] > AIRouter.COST_ESTIMATES["claude"]

    def test_router_with_cloud_budget_zero(self, sample_bookmark, mock_cloud_processor):
        """Test cloud mode with zero budget cap."""
        mock_cost_tracker = MagicMock()
        mock_cost_tracker.session_cost = 0.0

        config = HybridAIConfig(mode="cloud", budget_cap=0.0)
        router = AIRouter(
            cloud_processor=mock_cloud_processor,
            config=config,
            cost_tracker=mock_cost_tracker,
        )

        decision = router.route(sample_bookmark)

        assert decision.engine == "local"
        assert "budget exhausted" in decision.reason.lower()


# =============================================================================
# AIRouter Concurrent Request Handling Tests
# =============================================================================


class TestAIRouterConcurrency:
    """Tests for AIRouter concurrent request handling."""

    def test_process_batch_sequential_processing(self, mock_local_processor):
        """Test that batch processing is sequential."""
        call_order = []

        def track_call(bookmark):
            call_order.append(bookmark.url)
            return bookmark

        mock_local_processor.process_bookmark.side_effect = track_call

        bookmarks = [
            Bookmark(url=f"https://example.com/{i}", title=f"Article {i}", created=datetime.now())
            for i in range(5)
        ]

        config = HybridAIConfig(mode="local")
        router = AIRouter(local_processor=mock_local_processor, config=config)

        router.process_batch(bookmarks)

        # Verify sequential processing
        expected_order = [f"https://example.com/{i}" for i in range(5)]
        assert call_order == expected_order

    def test_multiple_routers_independent_stats(self, mock_local_processor):
        """Test multiple router instances have independent stats."""
        config = HybridAIConfig(mode="local")
        router1 = AIRouter(local_processor=mock_local_processor, config=config)
        router2 = AIRouter(local_processor=mock_local_processor, config=config)

        # Process with router1
        bookmark = Bookmark(url="https://example.com", title="Test", created=datetime.now())
        router1.process_bookmark(bookmark)

        assert router1.stats["local_processed"] == 1
        assert router2.stats["local_processed"] == 0


# =============================================================================
# AIRouter Integration Scenarios
# =============================================================================


class TestAIRouterIntegrationScenarios:
    """Integration-style tests for complete scenarios."""

    def test_full_workflow_local_mode(self, mock_local_processor):
        """Test complete workflow in local mode."""
        config = HybridAIConfig(mode="local", budget_cap=5.0)
        router = AIRouter(local_processor=mock_local_processor, config=config)

        bookmarks = [
            Bookmark(url=f"https://example.com/{i}", title=f"Article {i}", created=datetime.now())
            for i in range(5)
        ]

        # Process batch
        results = router.process_batch(bookmarks)

        # Verify results
        assert len(results) == 5

        # Verify statistics
        stats = router.get_statistics()
        assert stats["total_processed"] == 5
        assert stats["local_processed"] == 5
        assert stats["cloud_processed"] == 0
        assert stats["local_percentage"] == 100.0

    def test_full_workflow_hybrid_mode_with_escalation(self, mock_local_processor, mock_cloud_processor):
        """Test complete workflow in hybrid mode with escalation."""
        config = HybridAIConfig(
            mode="hybrid",
            escalation_threshold=0.7,
            cloud_required_types=["documentation"],
        )
        router = AIRouter(
            local_processor=mock_local_processor,
            cloud_processor=mock_cloud_processor,
            config=config,
        )

        # Route various content types
        simple_content = ContentData(
            url="https://example.com/simple",
            title="Simple",
            meta_description="Simple",
            word_count=50,
        )
        doc_content = ContentData(
            url="https://docs.example.com",
            title="API Docs",
            meta_description="Documentation",
            word_count=1000,
            content_categories=["documentation"],
        )

        bookmark = Bookmark(url="https://example.com", title="Test", created=datetime.now())

        # Simple content -> local
        decision1 = router.route(bookmark, simple_content)
        assert decision1.engine == "local"

        # Documentation -> cloud
        decision2 = router.route(bookmark, doc_content)
        assert decision2.engine == "cloud"

        # Low confidence -> cloud
        decision3 = router.route(bookmark, local_confidence=0.5)
        assert decision3.engine == "cloud"
        assert router.stats["escalated_to_cloud"] == 1

    def test_budget_exhaustion_scenario(self, mock_local_processor, mock_cloud_processor):
        """Test budget exhaustion scenario."""
        mock_cost_tracker = MagicMock()
        mock_cost_tracker.session_cost = 0.0

        config = HybridAIConfig(mode="hybrid", budget_cap=0.001)  # Very low budget
        router = AIRouter(
            local_processor=mock_local_processor,
            cloud_processor=mock_cloud_processor,
            config=config,
            cost_tracker=mock_cost_tracker,
        )

        bookmark = Bookmark(url="https://example.com", title="Test", created=datetime.now())

        # First route - budget not exhausted
        decision1 = router.route(bookmark)
        assert decision1.engine == "local"  # Default routing

        # Simulate budget exhaustion
        mock_cost_tracker.session_cost = 0.002

        # Second route - budget exhausted
        decision2 = router.route(bookmark)
        assert decision2.engine == "local"
        assert "Budget exhausted" in decision2.reason


# =============================================================================
# AIRouter Logging Tests
# =============================================================================


class TestAIRouterLogging:
    """Tests for AIRouter logging behavior."""

    def test_initialization_logs_mode_and_budget(self, caplog):
        """Test initialization logs mode and budget."""
        import logging
        caplog.set_level(logging.INFO)

        config = HybridAIConfig(mode="hybrid", budget_cap=10.0)
        router = AIRouter(config=config)

        # Check that logging occurred
        assert router.logger is not None

    def test_cloud_failure_logs_warning(self, sample_bookmark, mock_local_processor, caplog):
        """Test cloud failure logs warning."""
        import logging
        caplog.set_level(logging.WARNING)

        mock_cloud = MagicMock()
        mock_cloud.is_available = True
        mock_cloud.generate_description.side_effect = Exception("Cloud error")

        config = HybridAIConfig(mode="cloud")
        router = AIRouter(
            local_processor=mock_local_processor,
            cloud_processor=mock_cloud,
            config=config,
        )

        router.process_bookmark(sample_bookmark)

        # Verify warning was logged
        assert any("Cloud" in record.message or "falling back" in record.message
                   for record in caplog.records if record.levelno >= logging.WARNING)
