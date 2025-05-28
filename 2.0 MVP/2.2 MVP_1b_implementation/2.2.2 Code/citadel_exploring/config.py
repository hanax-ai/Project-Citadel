
"""
Configuration management for Citadel.

This module provides configuration management functionality for the Citadel project.
"""

import os
import json
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for Citadel."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file. If None, default config is used.
        """
        self._config: Dict[str, Any] = {}
        self._config_path = config_path
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self._load_default_config()

    def _load_default_config(self) -> None:
        """Load default configuration."""
        self._config = {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": None
            },
            "api": {
                "timeout": 30,
                "retries": 3
            }
        }

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from a file.

        Args:
            config_path: Path to the configuration file.
        """
        with open(config_path, 'r') as f:
            self._config = json.load(f)
        self._config_path = config_path

    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Save configuration to a file.

        Args:
            config_path: Path to save the configuration file. If None, uses the path
                         from which the configuration was loaded.
        """
        save_path = config_path or self._config_path
        if not save_path:
            raise ValueError("No config path specified")
            
        with open(save_path, 'w') as f:
            json.dump(self._config, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: The configuration key.
            default: Default value if key is not found.

        Returns:
            The configuration value.
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: The configuration key.
            value: The configuration value.
        """
        self._config[key] = value
