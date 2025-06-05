"""
Configuration management for the Bookmark Processor.

This module handles loading and merging configuration from multiple sources:
1. Default configuration
2. User configuration file
3. Command-line arguments
"""

import configparser
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional


class Configuration:
    """Manages application configuration with multiple sources."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to user configuration file
        """
        self.config = configparser.ConfigParser()
        self._load_default_config()
        
        if config_path:
            self._load_user_config(config_path)
    
    def _get_default_config_path(self) -> Path:
        """Get path to default configuration file."""
        if getattr(sys, 'frozen', False):
            # Running as PyInstaller executable
            app_dir = Path(sys.executable).parent
            config_path = app_dir / 'config' / 'default_config.ini'
        else:
            # Running as Python script
            config_path = Path(__file__).parent / 'default_config.ini'
        
        return config_path
    
    def _load_default_config(self) -> None:
        """Load default configuration."""
        default_path = self._get_default_config_path()
        if default_path.exists():
            self.config.read(default_path)
        else:
            # Fallback to hardcoded defaults if file not found
            self._set_hardcoded_defaults()
    
    def _set_hardcoded_defaults(self) -> None:
        """Set hardcoded default values."""
        # Network settings
        self.config['network'] = {
            'timeout': '30',
            'max_retries': '3',
            'default_delay': '0.5',
            'max_concurrent_requests': '10',
            'user_agent_rotation': 'true',
            'google_delay': '2.0',
            'github_delay': '1.5',
            'youtube_delay': '2.0',
            'linkedin_delay': '2.0'
        }
        
        # Processing settings
        self.config['processing'] = {
            'batch_size': '100',
            'max_tags_per_bookmark': '5',
            'target_unique_tags': '150',
            'ai_model': 'facebook/bart-large-cnn',
            'max_description_length': '150',
            'use_existing_content': 'true'
        }
        
        # Checkpoint settings
        self.config['checkpoint'] = {
            'enabled': 'true',
            'save_interval': '50',
            'checkpoint_dir': '.bookmark_checkpoints',
            'auto_cleanup': 'true'
        }
        
        # Output settings
        self.config['output'] = {
            'output_format': 'raindrop_import',
            'preserve_folder_structure': 'true',
            'include_timestamps': 'true',
            'error_log_detailed': 'true'
        }
        
        # Logging settings
        self.config['logging'] = {
            'log_level': 'INFO',
            'log_file': 'bookmark_processor.log',
            'console_output': 'true',
            'performance_logging': 'true'
        }
        
        # Executable settings
        self.config['executable'] = {
            'model_cache_dir': '%APPDATA%/BookmarkProcessor/models',
            'temp_dir': '%TEMP%/BookmarkProcessor',
            'cleanup_on_exit': 'true'
        }
    
    def _load_user_config(self, config_path: Path) -> None:
        """Load user configuration file."""
        if config_path.exists():
            self.config.read(config_path)
    
    def update_from_args(self, args: Dict[str, Any]) -> None:
        """
        Update configuration from command-line arguments.
        
        Args:
            args: Dictionary of validated arguments
        """
        # Update processing settings
        if 'batch_size' in args:
            self.config.set('processing', 'batch_size', str(args['batch_size']))
        
        if 'max_retries' in args:
            self.config.set('network', 'max_retries', str(args['max_retries']))
        
        # Update checkpoint settings
        if args.get('clear_checkpoints'):
            self.config.set('checkpoint', 'enabled', 'false')
        elif args.get('resume'):
            self.config.set('checkpoint', 'enabled', 'true')
        
        # Update logging settings
        if args.get('verbose'):
            self.config.set('logging', 'log_level', 'DEBUG')
            self.config.set('logging', 'console_output', 'true')
    
    def get(self, section: str, option: str, fallback: Any = None) -> str:
        """Get configuration value."""
        return self.config.get(section, option, fallback=fallback)
    
    def getint(self, section: str, option: str, fallback: int = 0) -> int:
        """Get configuration value as integer."""
        return self.config.getint(section, option, fallback=fallback)
    
    def getfloat(self, section: str, option: str, fallback: float = 0.0) -> float:
        """Get configuration value as float."""
        return self.config.getfloat(section, option, fallback=fallback)
    
    def getboolean(self, section: str, option: str, fallback: bool = False) -> bool:
        """Get configuration value as boolean."""
        return self.config.getboolean(section, option, fallback=fallback)
    
    def get_model_cache_dir(self) -> Path:
        """Get AI model cache directory with environment variable expansion."""
        cache_dir = self.get('executable', 'model_cache_dir')
        cache_dir = os.path.expandvars(cache_dir)
        return Path(cache_dir)
    
    def get_checkpoint_dir(self) -> Path:
        """Get checkpoint directory."""
        return Path(self.get('checkpoint', 'checkpoint_dir', '.bookmark_checkpoints'))