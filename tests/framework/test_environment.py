"""
Test Environment Management for Integration Tests

Provides comprehensive test environment setup, configuration management,
and cleanup for integration testing scenarios.
"""

import os
import shutil
import tempfile
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import time

from bookmark_processor.config.configuration import Configuration


@dataclass
class EnvironmentConfig:
    """Configuration for test environment setup."""
    test_id: str
    base_dir: Path
    enable_logging: bool = True
    enable_checkpoints: bool = True
    enable_mock_services: bool = True
    network_simulation: str = 'fast'  # fast, slow, unstable, offline
    cleanup_on_exit: bool = True
    persist_artifacts: bool = False


class IntegrationTestEnvironment:
    """Manages a complete integration test environment."""
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.directories = {}
        self.files = {}
        self.services = {}
        self._original_env = {}
        self._is_setup = False
        self._setup_complete = False
    
    def setup(self) -> None:
        """Set up the complete test environment."""
        if self._setup_complete:
            return
        
        self.logger.info(f"Setting up integration test environment: {self.config.test_id}")
        
        # Create directory structure
        self._create_directories()
        
        # Set up configuration files
        self._create_configuration_files()
        
        # Set up environment variables
        self._setup_environment_variables()
        
        # Initialize mock services if enabled
        if self.config.enable_mock_services:
            self._setup_mock_services()
        
        # Create test data files
        self._create_test_data_files()
        
        self._setup_complete = True
        self.logger.info("Integration test environment setup complete")
    
    def cleanup(self) -> None:
        """Clean up the test environment."""
        if not self._setup_complete:
            return
        
        self.logger.info(f"Cleaning up integration test environment: {self.config.test_id}")
        
        # Restore environment variables
        self._restore_environment_variables()
        
        # Stop mock services
        self._cleanup_mock_services()
        
        # Remove directories if cleanup is enabled
        if self.config.cleanup_on_exit and not self.config.persist_artifacts:
            self._cleanup_directories()
        elif self.config.persist_artifacts:
            self.logger.info(f"Test artifacts preserved in: {self.config.base_dir}")
        
        self._setup_complete = False
        self.logger.info("Integration test environment cleanup complete")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the test environment."""
        logger = logging.getLogger(f"integration_test.{self.config.test_id}")
        
        if not self.config.enable_logging:
            logger.setLevel(logging.CRITICAL)
            return logger
        
        logger.setLevel(logging.DEBUG)
        
        # Create logs directory
        logs_dir = self.config.base_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        log_file = logs_dir / f"{self.config.test_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _create_directories(self) -> None:
        """Create the directory structure for testing."""
        base_dir = self.config.base_dir
        
        directories = {
            'base': base_dir,
            'input': base_dir / "input",
            'output': base_dir / "output", 
            'config': base_dir / "config",
            'logs': base_dir / "logs",
            'checkpoints': base_dir / "checkpoints",
            'data': base_dir / "data",
            'artifacts': base_dir / "artifacts",
            'temp': base_dir / "temp"
        }
        
        for name, path in directories.items():
            path.mkdir(parents=True, exist_ok=True)
            self.directories[name] = path
            self.logger.debug(f"Created directory: {name} -> {path}")
    
    def _create_configuration_files(self) -> None:
        """Create configuration files for testing."""
        config_dir = self.directories['config']
        
        # Main configuration file
        main_config = config_dir / "test_config.ini"
        config_content = self._generate_test_config_content()
        main_config.write_text(config_content)
        self.files['main_config'] = main_config
        
        # Network simulation config
        network_config = config_dir / "network_config.json"
        network_data = self._generate_network_config()
        network_config.write_text(json.dumps(network_data, indent=2))
        self.files['network_config'] = network_config
        
        # AI mock config
        ai_config = config_dir / "ai_mock_config.json"
        ai_data = self._generate_ai_mock_config()
        ai_config.write_text(json.dumps(ai_data, indent=2))
        self.files['ai_config'] = ai_config
        
        self.logger.debug("Created configuration files")
    
    def _generate_test_config_content(self) -> str:
        """Generate test configuration file content."""
        network_settings = NetworkSimulator.get_network_settings(self.config.network_simulation)
        
        return f"""
[network]
timeout = {network_settings['timeout']}
max_retries = {network_settings['retries']}
default_delay = {network_settings['delay']}
max_concurrent_requests = 5
user_agent_rotation = true

[processing]
batch_size = 10
max_tags_per_bookmark = 5
target_unique_tags = 50
ai_model = facebook/bart-large-cnn
max_description_length = 150
use_existing_content = true

[ai]
default_engine = mock
mock_response_delay = 0.1
mock_success_rate = 0.9

[checkpoint]
enabled = {str(self.config.enable_checkpoints).lower()}
save_interval = 5
checkpoint_dir = {self.directories['checkpoints']}
auto_cleanup = false

[logging]
log_level = DEBUG
log_file = {self.directories['logs'] / 'processor.log'}
console_output = true
performance_logging = true

[test]
test_mode = true
test_id = {self.config.test_id}
mock_network = true
simulate_errors = false
"""
    
    def _generate_network_config(self) -> Dict[str, Any]:
        """Generate network simulation configuration."""
        settings = NetworkSimulator.get_network_settings(self.config.network_simulation)
        
        return {
            "simulation_type": self.config.network_simulation,
            "settings": settings,
            "error_patterns": {
                "connection_errors": ["connection_timeout", "connection_refused"],
                "http_errors": [404, 500, 503],
                "network_delays": {
                    "min_delay": settings['delay'] * 0.5,
                    "max_delay": settings['delay'] * 2.0
                }
            },
            "response_patterns": {
                "success_responses": {
                    "status_code": 200,
                    "content_type": "text/html",
                    "mock_content": True
                },
                "redirect_responses": {
                    "status_codes": [301, 302, 307],
                    "max_redirects": 3
                }
            }
        }
    
    def _generate_ai_mock_config(self) -> Dict[str, Any]:
        """Generate AI mock service configuration."""
        return {
            "mock_ai_service": {
                "enabled": True,
                "response_delay": 0.1,
                "success_rate": 0.9,
                "models": {
                    "description_generator": {
                        "model_name": "mock-description-model",
                        "max_length": 150,
                        "response_patterns": [
                            "AI-generated description for {title}",
                            "Enhanced description: {excerpt}",
                            "Intelligent summary of {url}"
                        ]
                    },
                    "tag_generator": {
                        "model_name": "mock-tag-model",
                        "max_tags": 5,
                        "tag_patterns": [
                            "technology", "ai", "research", "development",
                            "programming", "software", "web", "tutorial",
                            "documentation", "example"
                        ]
                    }
                }
            },
            "cloud_ai_mock": {
                "claude": {
                    "enabled": True,
                    "api_key": "mock-claude-key",
                    "rate_limit": 10,
                    "response_delay": 0.5
                },
                "openai": {
                    "enabled": True,
                    "api_key": "mock-openai-key",
                    "rate_limit": 15,
                    "response_delay": 0.3
                }
            }
        }
    
    def _setup_environment_variables(self) -> None:
        """Set up environment variables for testing."""
        # Store original environment
        env_vars = [
            'BOOKMARK_PROCESSOR_CONFIG',
            'BOOKMARK_PROCESSOR_CHECKPOINT_DIR',
            'BOOKMARK_PROCESSOR_TEST_MODE',
            'BOOKMARK_PROCESSOR_LOG_LEVEL',
            'TRANSFORMERS_CACHE',
            'CLAUDE_API_KEY',
            'OPENAI_API_KEY'
        ]
        
        for var in env_vars:
            if var in os.environ:
                self._original_env[var] = os.environ[var]
        
        # Set test environment variables
        test_env = {
            'BOOKMARK_PROCESSOR_CONFIG': str(self.files['main_config']),
            'BOOKMARK_PROCESSOR_CHECKPOINT_DIR': str(self.directories['checkpoints']),
            'BOOKMARK_PROCESSOR_TEST_MODE': 'true',
            'BOOKMARK_PROCESSOR_LOG_LEVEL': 'DEBUG',
            'TRANSFORMERS_CACHE': str(self.directories['temp']),
            'CLAUDE_API_KEY': 'mock-claude-key',
            'OPENAI_API_KEY': 'mock-openai-key'
        }
        
        for var, value in test_env.items():
            os.environ[var] = value
        
        self.logger.debug("Set up test environment variables")
    
    def _restore_environment_variables(self) -> None:
        """Restore original environment variables."""
        # Remove test variables
        test_vars = [
            'BOOKMARK_PROCESSOR_CONFIG',
            'BOOKMARK_PROCESSOR_CHECKPOINT_DIR', 
            'BOOKMARK_PROCESSOR_TEST_MODE',
            'BOOKMARK_PROCESSOR_LOG_LEVEL',
            'TRANSFORMERS_CACHE',
            'CLAUDE_API_KEY',
            'OPENAI_API_KEY'
        ]
        
        for var in test_vars:
            if var in os.environ:
                del os.environ[var]
        
        # Restore original values
        for var, value in self._original_env.items():
            os.environ[var] = value
        
        self.logger.debug("Restored original environment variables")
    
    def _setup_mock_services(self) -> None:
        """Set up mock services for testing."""
        # This will be expanded based on specific service needs
        self.services['network_simulator'] = NetworkSimulator(self.config.network_simulation)
        self.logger.debug("Set up mock services")
    
    def _cleanup_mock_services(self) -> None:
        """Clean up mock services."""
        for service_name, service in self.services.items():
            if hasattr(service, 'cleanup'):
                service.cleanup()
        self.services.clear()
        self.logger.debug("Cleaned up mock services")
    
    def _create_test_data_files(self) -> None:
        """Create standard test data files."""
        data_dir = self.directories['data']
        
        # Create user agents file
        user_agents_file = data_dir / "user_agents.txt"
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        ]
        user_agents_file.write_text("\n".join(user_agents))
        self.files['user_agents'] = user_agents_file
        
        # Create site delays file  
        site_delays_file = data_dir / "site_delays.json"
        site_delays = {
            "google.com": 2.0,
            "github.com": 1.5,
            "youtube.com": 2.0,
            "linkedin.com": 2.0,
            "default": 0.5
        }
        site_delays_file.write_text(json.dumps(site_delays, indent=2))
        self.files['site_delays'] = site_delays_file
        
        self.logger.debug("Created test data files")
    
    def _cleanup_directories(self) -> None:
        """Clean up test directories."""
        try:
            if self.config.base_dir.exists():
                shutil.rmtree(self.config.base_dir)
                self.logger.debug(f"Removed test directory: {self.config.base_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to remove test directory: {e}")
    
    def get_directory(self, name: str) -> Path:
        """Get a directory path by name."""
        return self.directories.get(name)
    
    def get_file(self, name: str) -> Path:
        """Get a file path by name."""
        return self.files.get(name)
    
    def create_test_file(self, name: str, content: str, directory: str = 'temp') -> Path:
        """Create a test file with given content."""
        file_path = self.directories[directory] / name
        file_path.write_text(content)
        self.files[name] = file_path
        return file_path
    
    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


class TestEnvironmentManager:
    """Manages multiple test environments and provides factory methods."""
    
    def __init__(self):
        self.environments: Dict[str, IntegrationTestEnvironment] = {}
        self.logger = logging.getLogger("test_environment_manager")
    
    def create_environment(
        self,
        test_id: str,
        network_simulation: str = 'fast',
        enable_checkpoints: bool = True,
        persist_artifacts: bool = False
    ) -> IntegrationTestEnvironment:
        """Create a new test environment."""
        
        # Create temporary base directory
        base_dir = Path(tempfile.mkdtemp(prefix=f"bookmark_test_{test_id}_"))
        
        config = EnvironmentConfig(
            test_id=test_id,
            base_dir=base_dir,
            network_simulation=network_simulation,
            enable_checkpoints=enable_checkpoints,
            persist_artifacts=persist_artifacts
        )
        
        env = IntegrationTestEnvironment(config)
        self.environments[test_id] = env
        
        return env
    
    def get_environment(self, test_id: str) -> Optional[IntegrationTestEnvironment]:
        """Get an existing test environment."""
        return self.environments.get(test_id)
    
    def cleanup_environment(self, test_id: str) -> None:
        """Clean up a specific test environment."""
        env = self.environments.get(test_id)
        if env:
            env.cleanup()
            del self.environments[test_id]
    
    def cleanup_all(self) -> None:
        """Clean up all test environments."""
        for test_id in list(self.environments.keys()):
            self.cleanup_environment(test_id)
    
    @contextmanager
    def temporary_environment(
        self,
        test_id: str,
        **kwargs
    ) -> Generator[IntegrationTestEnvironment, None, None]:
        """Create a temporary test environment that auto-cleans."""
        env = self.create_environment(test_id, **kwargs)
        try:
            with env:
                yield env
        finally:
            self.cleanup_environment(test_id)


class NetworkSimulator:
    """Simulates different network conditions for testing."""
    
    NETWORK_PROFILES = {
        'fast': {
            'delay': 0.1,
            'timeout': 5,
            'retries': 1,
            'error_rate': 0.01,
            'connection_errors': 0.005
        },
        'slow': {
            'delay': 2.0,
            'timeout': 30,
            'retries': 3,
            'error_rate': 0.05,
            'connection_errors': 0.02
        },
        'unstable': {
            'delay': 1.0,
            'timeout': 10,
            'retries': 5,
            'error_rate': 0.15,
            'connection_errors': 0.1
        },
        'offline': {
            'delay': 0,
            'timeout': 1,
            'retries': 0,
            'error_rate': 1.0,
            'connection_errors': 1.0
        }
    }
    
    def __init__(self, profile: str = 'fast'):
        self.profile = profile
        self.settings = self.NETWORK_PROFILES.get(profile, self.NETWORK_PROFILES['fast'])
        self.logger = logging.getLogger(f"network_simulator.{profile}")
    
    @classmethod
    def get_network_settings(cls, profile: str) -> Dict[str, Any]:
        """Get network settings for a given profile."""
        return cls.NETWORK_PROFILES.get(profile, cls.NETWORK_PROFILES['fast'])
    
    def simulate_request_delay(self) -> None:
        """Simulate network delay for a request."""
        delay = self.settings['delay']
        if delay > 0:
            time.sleep(delay)
    
    def should_simulate_error(self) -> bool:
        """Determine if an error should be simulated."""
        import random
        return random.random() < self.settings['error_rate']
    
    def should_simulate_connection_error(self) -> bool:
        """Determine if a connection error should be simulated."""
        import random
        return random.random() < self.settings['connection_errors']
    
    def get_mock_response_config(self) -> Dict[str, Any]:
        """Get configuration for mock HTTP responses."""
        return {
            'timeout': self.settings['timeout'],
            'max_retries': self.settings['retries'],
            'simulate_delays': True,
            'base_delay': self.settings['delay'],
            'error_rate': self.settings['error_rate']
        }
    
    def cleanup(self) -> None:
        """Clean up network simulator resources."""
        pass  # No cleanup needed for basic simulator