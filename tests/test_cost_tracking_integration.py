"""
Unit tests for cost tracking functionality in the EnhancedBatchProcessor.

Tests verify proper cost estimation, budget checking, user confirmation workflows,
and integration with the CostTracker utility.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Optional
from dataclasses import dataclass

from bookmark_processor.core.url_validator import (
    EnhancedBatchProcessor,
    BatchConfig,
    CostBreakdown,
    BatchResult,
    ValidationResult
)
from bookmark_processor.utils.cost_tracker import CostTracker


class MockValidator:
    """Mock validator for testing batch processing (implements BatchProcessorInterface)."""

    def process_batch(self, items: List[str], batch_id: str) -> BatchResult:
        """Mock batch processing that returns a BatchResult."""
        results = [
            ValidationResult(url=url, is_valid=True, status_code=200)
            for url in items
        ]
        return BatchResult(
            batch_id=batch_id,
            items_processed=len(items),
            items_successful=len(items),
            items_failed=0,
            processing_time=0.1 * len(items),
            average_item_time=0.1,
            error_rate=0.0,
            results=results
        )

    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size."""
        return 100

    def estimate_processing_time(self, item_count: int) -> float:
        """Estimate processing time."""
        return item_count * 0.1


class TestCostTrackingIntegration:
    """Test cost tracking integration in EnhancedBatchProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_validator = MockValidator()
        self.cost_tracker = CostTracker(
            confirmation_interval=5.0,
            auto_save=False,  # Don't save during tests
            warning_threshold=2.0
        )
        
    def test_cost_tracking_disabled_by_default(self):
        """Test that cost tracking is disabled by default."""
        config = BatchConfig()
        processor = EnhancedBatchProcessor(
            self.mock_validator,
            config
        )

        assert not config.enable_cost_tracking
        assert processor.cost_tracker is None
        assert processor.total_session_cost == 0.0
        
    def test_cost_tracking_enabled_with_config(self):
        """Test cost tracking can be enabled via configuration."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
            cost_confirmation_threshold=2.0,
            budget_limit=10.0
        )
        processor = EnhancedBatchProcessor(
            self.mock_validator,
            config
        )
        processor.cost_tracker = self.cost_tracker

        assert config.enable_cost_tracking
        assert processor.cost_tracker is not None
        assert processor.config.cost_per_url_validation == 0.001
        assert processor.config.cost_confirmation_threshold == 2.0
        assert processor.config.budget_limit == 10.0
        
    def test_cost_estimation_calculation(self):
        """Test cost estimation algorithm."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001
        )
        processor = EnhancedBatchProcessor(
            self.mock_validator,
            config
        )
        processor.cost_tracker = self.cost_tracker

        # Test cost estimation for different batch sizes
        small_batch_cost = processor.estimate_batch_cost(10)
        large_batch_cost = processor.estimate_batch_cost(200)  # >100 for bulk discount

        # Verify basic cost calculation
        assert small_batch_cost.batch_size == 10
        assert large_batch_cost.batch_size == 200

        # Large batches (>100) should have bulk discount factor
        assert "bulk_discount" in large_batch_cost.cost_factors
        assert large_batch_cost.cost_factors["bulk_discount"] < 0  # It's a negative discount

        # Small batches (<10) have premium factor
        very_small_batch_cost = processor.estimate_batch_cost(5)
        assert "small_batch_premium" in very_small_batch_cost.cost_factors
        assert very_small_batch_cost.cost_factors["small_batch_premium"] > 0
            
    def test_cost_factors_application(self):
        """Test that cost factors are properly applied."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001
        )
        processor = EnhancedBatchProcessor(
            self.mock_validator,
            config
        )
        processor.cost_tracker = self.cost_tracker

        # Test different batch sizes to verify cost factors
        small_cost = processor.estimate_batch_cost(3)  # Should have premium
        medium_cost = processor.estimate_batch_cost(50)  # Base cost
        large_cost = processor.estimate_batch_cost(200)  # Should have discount

        # Verify cost factors are applied correctly
        # Small batches have premium (higher per-item cost)
        assert small_cost.estimated_cost_per_item > medium_cost.estimated_cost_per_item
        # Large batches have discount (lower per-item cost)
        assert large_cost.estimated_cost_per_item < medium_cost.estimated_cost_per_item

        # Total cost should reflect factors
        assert small_cost.total_estimated_cost == pytest.approx(small_cost.estimated_cost_per_item * 3, rel=1e-6)
        assert large_cost.total_estimated_cost == pytest.approx(large_cost.estimated_cost_per_item * 200, rel=1e-6)
        
    def test_budget_limit_enforcement(self):
        """Test budget limit checking."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
            budget_limit=5.0
        )
        processor = EnhancedBatchProcessor(
            self.mock_validator,
            config
        )
        processor.cost_tracker = self.cost_tracker

        # Add some existing cost
        processor.total_session_cost = 4.5

        # Test budget check with cost that would exceed limit
        with patch('builtins.input', return_value='n'):
            result = processor._check_budget_and_confirm_sync(1.0)  # Would exceed 5.0 limit
            assert not result

        # Test budget check with acceptable cost
        result = processor._check_budget_and_confirm_sync(0.4)  # Within limit
        assert result
        
    def test_cost_confirmation_threshold(self):
        """Test cost confirmation threshold behavior."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
            cost_confirmation_threshold=1.0
        )
        processor = EnhancedBatchProcessor(
            self.mock_validator,
            config
        )
        processor.cost_tracker = self.cost_tracker

        # Test below threshold (no confirmation needed)
        result = processor._check_budget_and_confirm_sync(0.5)
        assert result

        # Test above threshold (confirmation needed)
        with patch('builtins.input', return_value='y'):
            result = processor._check_budget_and_confirm_sync(1.5)
            assert result

        with patch('builtins.input', return_value='n'):
            result = processor._check_budget_and_confirm_sync(1.5)
            assert not result
            
    def test_cost_tracker_integration(self):
        """Test integration with CostTracker utility."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
            cost_confirmation_threshold=1.0
        )
        processor = EnhancedBatchProcessor(
            self.mock_validator,
            config
        )
        processor.cost_tracker = self.cost_tracker

        # Mock cost tracker methods
        self.cost_tracker.get_confirmation_prompt = Mock(return_value="Test prompt: ")

        # Test confirmation with cost tracker
        with patch('builtins.input', return_value='y'):
            result = processor._check_budget_and_confirm_sync(1.5)
            assert result

        # Verify cost tracker was used
        self.cost_tracker.get_confirmation_prompt.assert_called_once()
        
    def test_actual_cost_recording(self):
        """Test recording of actual costs after processing."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001
        )
        processor = EnhancedBatchProcessor(
            self.mock_validator,
            config
        )
        processor.cost_tracker = self.cost_tracker

        # Mock cost tracker's add_cost_record method
        self.cost_tracker.add_cost_record = Mock()

        # Record a batch cost
        batch_id = "test_batch_1"
        actual_cost = 0.05
        processor.record_batch_cost(batch_id, actual_cost)

        # Verify cost was recorded
        assert processor.total_session_cost == actual_cost
        assert (batch_id, actual_cost) in processor.batch_cost_history

        # Verify cost tracker was called
        self.cost_tracker.add_cost_record.assert_called_once_with(
            provider="url_validation",
            model="batch_processor",
            input_tokens=0,
            output_tokens=0,
            cost_usd=actual_cost,
            operation_type="url_validation_batch",
            bookmark_count=1,
            success=True
        )
        
    def test_cost_statistics_generation(self):
        """Test generation of cost statistics."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001
        )
        processor = EnhancedBatchProcessor(
            self.mock_validator,
            config
        )
        processor.cost_tracker = self.cost_tracker

        # Add some cost history
        processor.record_batch_cost("batch1", 0.05)
        processor.record_batch_cost("batch2", 0.03)

        # Get statistics
        stats = processor.get_cost_statistics()

        # Verify statistics structure
        assert "total_session_cost" in stats
        assert "batch_count" in stats
        assert "average_batch_cost" in stats  # Note: actual key is average_batch_cost, not average_cost_per_batch

        # Verify values
        assert stats["total_session_cost"] == 0.08
        assert stats["batch_count"] == 2
        assert stats["average_batch_cost"] == 0.04
        
    def test_cost_trend_calculation(self):
        """Test cost trend analysis."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001
        )
        processor = EnhancedBatchProcessor(
            self.mock_validator,
            config
        )
        processor.cost_tracker = self.cost_tracker

        # Test insufficient data
        trend = processor._calculate_cost_trend()
        assert trend == "insufficient_data"

        # Add increasing cost data - need at least 10 entries for comparison
        # First 5 entries are "early" costs, last 5 are "recent" costs
        early_costs = [0.01, 0.01, 0.01, 0.01, 0.01]  # Average: 0.01
        recent_costs = [0.10, 0.10, 0.10, 0.10, 0.10]  # Average: 0.10 (10x increase)
        for i, cost in enumerate(early_costs + recent_costs):
            processor.record_batch_cost(f"batch{i}", cost)

        trend = processor._calculate_cost_trend()
        assert trend == "increasing"

        # Test decreasing trend - clear and add new data
        processor.batch_cost_history.clear()
        processor.total_session_cost = 0.0
        early_costs = [0.10, 0.10, 0.10, 0.10, 0.10]  # Average: 0.10
        recent_costs = [0.01, 0.01, 0.01, 0.01, 0.01]  # Average: 0.01 (90% decrease)
        for i, cost in enumerate(early_costs + recent_costs):
            processor.batch_cost_history.append((f"batch{i}", cost))

        trend = processor._calculate_cost_trend()
        assert trend == "decreasing"
        
    def test_budget_exceeded_with_user_override(self):
        """Test budget limit with user override option."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
            budget_limit=1.0
        )
        processor = EnhancedBatchProcessor(
            self.mock_validator,
            config
        )
        processor.cost_tracker = self.cost_tracker

        processor.total_session_cost = 0.8

        # Test user chooses to continue despite budget limit
        with patch('builtins.input', return_value='y'):
            result = processor._check_budget_and_confirm_sync(0.5)  # Would exceed limit
            assert result

        # Test user chooses to stop
        with patch('builtins.input', return_value='n'):
            result = processor._check_budget_and_confirm_sync(0.5)  # Would exceed limit
            assert not result
            
    def test_keyboard_interrupt_during_confirmation(self):
        """Test handling of keyboard interrupt during confirmation."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
            cost_confirmation_threshold=1.0
        )
        processor = EnhancedBatchProcessor(
            self.mock_validator,
            config
        )
        processor.cost_tracker = self.cost_tracker

        # Test KeyboardInterrupt during confirmation
        with patch('builtins.input', side_effect=KeyboardInterrupt):
            result = processor._check_budget_and_confirm_sync(1.5)
            assert not result

        # Test EOFError during confirmation
        with patch('builtins.input', side_effect=EOFError):
            result = processor._check_budget_and_confirm_sync(1.5)
            assert not result
            
    def test_batch_processing_with_cost_tracking(self):
        """Test end-to-end batch processing with cost tracking."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001,
            optimal_batch_size=5,  # Use optimal_batch_size instead of batch_size
            enable_async_processing=False  # Use sync mode for simpler testing
        )
        processor = EnhancedBatchProcessor(
            self.mock_validator,
            config
        )
        processor.cost_tracker = self.cost_tracker

        # Add test items
        test_urls = [f"https://example{i}.com" for i in range(10)]

        # Mock user confirmation for any prompts
        with patch('builtins.input', return_value='y'):
            # Add items (may prompt for confirmation)
            added = processor.add_items(test_urls)
            assert added

            # Process all items
            results = processor.process_all()

        # Verify results
        assert len(results) == 10
        assert processor.total_session_cost > 0
        assert len(processor.batch_cost_history) > 0

        # Verify cost tracking occurred
        stats = processor.get_cost_statistics()
        assert stats["total_session_cost"] > 0
        assert stats["batch_count"] > 0
        
    def test_failed_batch_cost_recording(self):
        """Test cost recording for failed batches."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001
        )

        # Create processor with failing validator (needs process_batch method)
        failing_validator = Mock()
        failing_validator.process_batch.side_effect = Exception("Validation failed")

        processor = EnhancedBatchProcessor(
            failing_validator,
            config
        )
        processor.cost_tracker = self.cost_tracker

        # Process a batch that will fail
        test_urls = ["https://example1.com", "https://example2.com"]

        # Call _process_single_batch directly to simulate failure
        batch_result = processor._process_single_batch("test_batch", test_urls)

        # Verify failed batch still records partial cost
        assert batch_result.actual_cost is not None
        assert batch_result.actual_cost > 0
        assert batch_result.items_failed == len(test_urls)
        assert batch_result.items_successful == 0
        
    @pytest.mark.parametrize("batch_size,expected_factors", [
        (1, ["small_batch_premium"]),
        (10, []),  # Base case, no special factors (10 is at the boundary)
        (50, []),  # Medium case, no special factors
        (200, ["bulk_discount"]),  # >100 gets bulk discount
    ])
    def test_cost_factors_by_batch_size(self, batch_size, expected_factors):
        """Test that appropriate cost factors are applied based on batch size."""
        config = BatchConfig(
            enable_cost_tracking=True,
            cost_per_url_validation=0.001
        )
        processor = EnhancedBatchProcessor(
            self.mock_validator,
            config
        )
        processor.cost_tracker = self.cost_tracker

        cost_breakdown = processor.estimate_batch_cost(batch_size)

        # Verify expected factors are present
        for factor in expected_factors:
            assert factor in cost_breakdown.cost_factors, f"Expected {factor} in {cost_breakdown.cost_factors}"

        # Verify cost calculation consistency
        assert cost_breakdown.batch_size == batch_size
        assert cost_breakdown.total_estimated_cost > 0
        assert cost_breakdown.estimated_cost_per_item > 0


class TestCostBreakdown:
    """Test CostBreakdown dataclass functionality."""
    
    def test_cost_breakdown_creation(self):
        """Test CostBreakdown creation and serialization."""
        breakdown = CostBreakdown(
            operation_type="url_validation",
            batch_size=50,
            estimated_cost_per_item=0.001,
            total_estimated_cost=0.05,
            cost_factors={"bulk_discount": 0.9}
        )
        
        assert breakdown.operation_type == "url_validation"
        assert breakdown.batch_size == 50
        assert breakdown.estimated_cost_per_item == 0.001
        assert breakdown.total_estimated_cost == 0.05
        assert breakdown.cost_factors["bulk_discount"] == 0.9
        
        # Test serialization
        breakdown_dict = breakdown.to_dict()
        assert isinstance(breakdown_dict, dict)
        assert breakdown_dict["operation_type"] == "url_validation"
        assert breakdown_dict["batch_size"] == 50
        assert "timestamp" in breakdown_dict


if __name__ == "__main__":
    pytest.main([__file__])