"""
Hybrid AI Routing Module

Routes bookmarks to optimal AI engine based on content complexity, budget constraints,
and content type analysis. Supports local-only, cloud-only, and hybrid modes.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .data_models import Bookmark
    from .content_analyzer import ContentData
    from .ai_processor import EnhancedAIProcessor, AIProcessingResult
    from .base_api_client import BaseAPIClient

from ..utils.cost_tracker import CostTracker


@dataclass
class HybridAIConfig:
    """Configuration for hybrid AI routing."""

    mode: str = "hybrid"  # local, cloud, hybrid
    escalation_threshold: float = 0.7  # Confidence below this escalates to cloud
    budget_cap: float = 5.00  # USD maximum budget
    simple_threshold: int = 200  # Word count threshold for simple content
    cloud_required_types: List[str] = field(default_factory=lambda: [
        "documentation", "research", "technical", "academic"
    ])
    # Track costs per session
    track_costs: bool = True
    # Default model preferences
    local_model: str = "facebook/bart-large-cnn"
    cloud_provider: str = "claude"  # claude or openai

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mode": self.mode,
            "escalation_threshold": self.escalation_threshold,
            "budget_cap": self.budget_cap,
            "simple_threshold": self.simple_threshold,
            "cloud_required_types": self.cloud_required_types,
            "track_costs": self.track_costs,
            "local_model": self.local_model,
            "cloud_provider": self.cloud_provider,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridAIConfig":
        """Create from dictionary."""
        return cls(
            mode=data.get("mode", "hybrid"),
            escalation_threshold=data.get("escalation_threshold", 0.7),
            budget_cap=data.get("budget_cap", 5.00),
            simple_threshold=data.get("simple_threshold", 200),
            cloud_required_types=data.get("cloud_required_types", [
                "documentation", "research", "technical", "academic"
            ]),
            track_costs=data.get("track_costs", True),
            local_model=data.get("local_model", "facebook/bart-large-cnn"),
            cloud_provider=data.get("cloud_provider", "claude"),
        )


@dataclass
class RoutingDecision:
    """Result of routing decision with reasoning."""

    engine: str  # "local" or "cloud"
    reason: str
    confidence: float = 1.0
    estimated_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "engine": self.engine,
            "reason": self.reason,
            "confidence": self.confidence,
            "estimated_cost": self.estimated_cost,
        }


class AIRouter:
    """Route bookmarks to optimal AI engine based on content."""

    # Average cost estimates per bookmark (USD)
    COST_ESTIMATES = {
        "claude": 0.0006,  # ~$0.0006 per bookmark
        "openai": 0.0012,  # ~$0.0012 per bookmark
        "local": 0.0,  # Free
    }

    def __init__(
        self,
        local_processor: Optional["EnhancedAIProcessor"] = None,
        cloud_processor: Optional["BaseAPIClient"] = None,
        config: Optional[HybridAIConfig] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        """
        Initialize AI router.

        Args:
            local_processor: Local AI processor instance
            cloud_processor: Cloud API client instance
            config: Hybrid AI configuration
            cost_tracker: Cost tracking instance
        """
        self.local = local_processor
        self.cloud = cloud_processor
        self.config = config or HybridAIConfig()

        # Use provided cost tracker or create new one
        if cost_tracker:
            self.cost_tracker = cost_tracker
        elif self.config.track_costs:
            self.cost_tracker = CostTracker(
                confirmation_interval=self.config.budget_cap / 2,
                warning_threshold=self.config.budget_cap * 0.8,
            )
        else:
            self.cost_tracker = None

        # Statistics
        self.stats = {
            "local_processed": 0,
            "cloud_processed": 0,
            "escalated_to_cloud": 0,
            "budget_limited": 0,
            "total_cost": 0.0,
        }

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"AI Router initialized (mode={self.config.mode}, "
            f"budget_cap=${self.config.budget_cap:.2f})"
        )

    def route(
        self,
        bookmark: "Bookmark",
        content: Optional["ContentData"] = None,
        local_confidence: Optional[float] = None,
    ) -> RoutingDecision:
        """
        Determine which AI engine to use.

        Args:
            bookmark: Bookmark to process
            content: Content data from analysis
            local_confidence: Pre-computed local confidence (for escalation)

        Returns:
            RoutingDecision with engine and reasoning
        """
        # Mode-based routing
        if self.config.mode == "local":
            return RoutingDecision(
                engine="local",
                reason="Local-only mode configured",
                confidence=1.0,
                estimated_cost=0.0,
            )

        if self.config.mode == "cloud":
            if not self._is_cloud_available():
                return RoutingDecision(
                    engine="local",
                    reason="Cloud requested but not available, falling back to local",
                    confidence=0.8,
                    estimated_cost=0.0,
                )
            if self._is_budget_exhausted():
                self.stats["budget_limited"] += 1
                return RoutingDecision(
                    engine="local",
                    reason="Cloud mode but budget exhausted, falling back to local",
                    confidence=0.8,
                    estimated_cost=0.0,
                )
            return RoutingDecision(
                engine="cloud",
                reason="Cloud-only mode configured",
                confidence=1.0,
                estimated_cost=self._estimate_cost(),
            )

        # Hybrid mode routing
        return self._route_hybrid(bookmark, content, local_confidence)

    def _route_hybrid(
        self,
        bookmark: "Bookmark",
        content: Optional["ContentData"],
        local_confidence: Optional[float],
    ) -> RoutingDecision:
        """Route in hybrid mode based on content analysis."""

        # Check 1: Budget exhausted -> local only
        if self._is_budget_exhausted():
            self.stats["budget_limited"] += 1
            return RoutingDecision(
                engine="local",
                reason="Budget exhausted",
                confidence=0.9,
                estimated_cost=0.0,
            )

        # Check 2: Cloud not available -> local only
        if not self._is_cloud_available():
            return RoutingDecision(
                engine="local",
                reason="Cloud AI not available",
                confidence=0.9,
                estimated_cost=0.0,
            )

        # Check 3: Simple content (low word count) -> local
        if content and content.word_count < self.config.simple_threshold:
            return RoutingDecision(
                engine="local",
                reason=f"Simple content ({content.word_count} words < {self.config.simple_threshold})",
                confidence=0.95,
                estimated_cost=0.0,
            )

        # Check 4: Cloud-required content types -> cloud
        if content:
            content_type = self._detect_content_type(content)
            if content_type in self.config.cloud_required_types:
                return RoutingDecision(
                    engine="cloud",
                    reason=f"Content type '{content_type}' requires cloud AI",
                    confidence=0.9,
                    estimated_cost=self._estimate_cost(),
                )

        # Check 5: Low local confidence -> escalate to cloud
        if local_confidence is not None and local_confidence < self.config.escalation_threshold:
            self.stats["escalated_to_cloud"] += 1
            return RoutingDecision(
                engine="cloud",
                reason=f"Low local confidence ({local_confidence:.2f} < {self.config.escalation_threshold})",
                confidence=0.85,
                estimated_cost=self._estimate_cost(),
            )

        # Default: use local AI
        return RoutingDecision(
            engine="local",
            reason="Default routing to local AI",
            confidence=0.9,
            estimated_cost=0.0,
        )

    def _detect_content_type(self, content: "ContentData") -> str:
        """Detect content type from content data."""
        # Check content categories from analyzer
        if content.content_categories:
            return content.content_categories[0].lower()

        # Fallback to content type field
        if content.content_type:
            return content.content_type.lower()

        # Analyze title and content for type indicators
        text_to_check = f"{content.title} {content.meta_description}".lower()

        type_indicators = {
            "documentation": ["docs", "documentation", "api", "reference", "manual"],
            "research": ["paper", "research", "study", "journal", "arxiv"],
            "technical": ["technical", "specification", "protocol", "implementation"],
            "academic": ["academic", "thesis", "dissertation", "scholarly"],
            "tutorial": ["tutorial", "guide", "how-to", "walkthrough"],
            "article": ["article", "blog", "post", "news"],
        }

        for content_type, indicators in type_indicators.items():
            if any(indicator in text_to_check for indicator in indicators):
                return content_type

        return "general"

    def _is_budget_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        if not self.cost_tracker:
            return False
        return self.cost_tracker.session_cost >= self.config.budget_cap

    def _is_cloud_available(self) -> bool:
        """Check if cloud AI is available."""
        if not self.cloud:
            return False
        return getattr(self.cloud, "is_available", True)

    def _estimate_cost(self) -> float:
        """Estimate cost for cloud processing."""
        return self.COST_ESTIMATES.get(self.config.cloud_provider, 0.001)

    def process_bookmark(
        self,
        bookmark: "Bookmark",
        content: Optional["ContentData"] = None,
    ) -> "Bookmark":
        """
        Process a bookmark using the optimal AI engine.

        Args:
            bookmark: Bookmark to process
            content: Optional content data

        Returns:
            Processed bookmark
        """
        # Get routing decision
        decision = self.route(bookmark, content)

        # Process based on decision
        if decision.engine == "cloud" and self.cloud:
            try:
                result = self._process_with_cloud(bookmark)
                if result:
                    self.stats["cloud_processed"] += 1
                    self._record_cost(decision.estimated_cost)
                    return result
                # Fallback to local if cloud fails
                self.logger.warning(f"Cloud processing failed for {bookmark.url}, falling back to local")
            except Exception as e:
                self.logger.warning(f"Cloud error for {bookmark.url}: {e}, falling back to local")

        # Local processing
        if self.local:
            result = self.local.process_bookmark(bookmark)
            self.stats["local_processed"] += 1
            return result

        # No processor available
        self.logger.error("No AI processor available")
        return bookmark

    def _process_with_cloud(self, bookmark: "Bookmark") -> Optional["Bookmark"]:
        """Process bookmark with cloud AI."""
        if not self.cloud:
            return None

        try:
            # Cloud clients have generate_description method
            description = self.cloud.generate_description(bookmark)
            if description:
                bookmark.enhanced_description = description
                bookmark.processing_status.ai_processed = True
                return bookmark
        except Exception as e:
            self.logger.debug(f"Cloud processing error: {e}")

        return None

    def _record_cost(self, cost: float) -> None:
        """Record cost to tracker."""
        if self.cost_tracker and cost > 0:
            self.cost_tracker.add_cost_record(
                provider=self.config.cloud_provider,
                model=f"{self.config.cloud_provider}-default",
                input_tokens=150,  # Estimated
                output_tokens=50,  # Estimated
                cost_usd=cost,
                operation_type="description_generation",
            )
            self.stats["total_cost"] += cost

    def process_batch(
        self,
        bookmarks: List["Bookmark"],
        content_data_map: Optional[Dict[str, "ContentData"]] = None,
        progress_callback=None,
    ) -> List["Bookmark"]:
        """
        Process multiple bookmarks with optimal routing.

        Args:
            bookmarks: List of bookmarks to process
            content_data_map: Optional mapping of URLs to content data
            progress_callback: Optional callback for progress updates

        Returns:
            List of processed bookmarks
        """
        if content_data_map is None:
            content_data_map = {}

        results = []
        total = len(bookmarks)

        for i, bookmark in enumerate(bookmarks):
            content = content_data_map.get(bookmark.url)
            processed = self.process_bookmark(bookmark, content)
            results.append(processed)

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total = self.stats["local_processed"] + self.stats["cloud_processed"]

        stats = {
            "total_processed": total,
            "local_processed": self.stats["local_processed"],
            "cloud_processed": self.stats["cloud_processed"],
            "escalated_to_cloud": self.stats["escalated_to_cloud"],
            "budget_limited": self.stats["budget_limited"],
            "total_cost_usd": self.stats["total_cost"],
            "mode": self.config.mode,
            "budget_cap": self.config.budget_cap,
        }

        if total > 0:
            stats["local_percentage"] = (self.stats["local_processed"] / total) * 100
            stats["cloud_percentage"] = (self.stats["cloud_processed"] / total) * 100
        else:
            stats["local_percentage"] = 0.0
            stats["cloud_percentage"] = 0.0

        if self.cost_tracker:
            stats["remaining_budget"] = max(0, self.config.budget_cap - self.cost_tracker.session_cost)
        else:
            stats["remaining_budget"] = self.config.budget_cap

        return stats

    def reset_statistics(self) -> None:
        """Reset routing statistics."""
        self.stats = {
            "local_processed": 0,
            "cloud_processed": 0,
            "escalated_to_cloud": 0,
            "budget_limited": 0,
            "total_cost": 0.0,
        }
        if self.cost_tracker:
            self.cost_tracker.reset_session()
