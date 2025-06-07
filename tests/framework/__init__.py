"""
Integration Test Framework for Bookmark Processor

This package provides a comprehensive testing framework for integration tests,
including test environment setup, scenario management, and result validation.
"""

from .test_environment import (
    IntegrationTestEnvironment,
    NetworkSimulator,
    TestEnvironmentManager,
)
from .test_fixtures import DataGenerator, IntegrationTestFixtures, MockServiceManager
from .test_scenarios import ScenarioRunner, StandardScenarios, TestScenario
from .test_validators import (
    CheckpointValidator,
    CompositeValidator,
    ErrorValidator,
    ResultValidator,
)

__all__ = [
    "IntegrationTestEnvironment",
    "TestEnvironmentManager",
    "NetworkSimulator",
    "TestScenario",
    "ScenarioRunner",
    "StandardScenarios",
    "ResultValidator",
    "CheckpointValidator",
    "ErrorValidator",
    "CompositeValidator",
    "IntegrationTestFixtures",
    "DataGenerator",
    "MockServiceManager",
]
