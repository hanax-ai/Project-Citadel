
"""
Utility functions for Citadel.

This module provides utility functions for the Citadel project.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Union


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> callable:
    """
    Retry decorator for functions that might fail.

    Args:
        max_attempts: Maximum number of retry attempts.
        delay: Initial delay between retries in seconds.
        backoff_factor: Backoff factor for delay between retries.
        exceptions: Tuple of exceptions to catch and retry.

    Returns:
        Decorated function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise e
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
        return wrapper
    return decorator


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON data from a file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Parsed JSON data.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save.
        file_path: Path to the JSON file.
        indent: Indentation level for the JSON file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def ensure_dir(directory: str) -> None:
    """
    Ensure that a directory exists.

    Args:
        directory: Directory path.
    """
    os.makedirs(directory, exist_ok=True)


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = '',
    separator: str = '.'
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten.
        parent_key: Parent key for nested dictionaries.
        separator: Separator for keys.

    Returns:
        Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, separator).items())
        else:
            items.append((new_key, v))
    return dict(items)
