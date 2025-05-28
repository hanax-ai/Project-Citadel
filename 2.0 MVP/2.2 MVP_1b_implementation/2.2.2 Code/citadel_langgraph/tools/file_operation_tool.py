
"""
File operation tool for agent workflows.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List, Union

from citadel_core.logging import get_logger
from .tool_registry import BaseTool


class FileOperationTool(BaseTool):
    """
    Tool for performing file operations.
    
    This tool allows agents to read, write, and manipulate files.
    """
    
    name = "file_operation"
    description = "Perform file operations such as reading, writing, and listing files"
    
    def __init__(
        self,
        base_directory: Optional[str] = None,
        allowed_extensions: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the file operation tool.
        
        Args:
            base_directory: Base directory for file operations.
            allowed_extensions: List of allowed file extensions.
            logger: Logger instance.
        """
        self.logger = logger or get_logger("citadel.langgraph.tools.file_operation")
        self.base_directory = base_directory or os.getcwd()
        self.allowed_extensions = allowed_extensions or [".txt", ".json", ".csv", ".md"]
    
    def __call__(
        self,
        operation: str,
        path: Optional[str] = None,
        content: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform a file operation.
        
        Args:
            operation: The operation to perform (read, write, list, etc.).
            path: The file path.
            content: The content to write (for write operations).
            **kwargs: Additional operation-specific parameters.
            
        Returns:
            The result of the operation.
        """
        self.logger.info(f"Performing file operation: {operation}")
        
        try:
            # Validate and normalize the path
            if path:
                path = self._normalize_path(path)
            
            # Perform the requested operation
            if operation == "read":
                return self._read_file(path)
            elif operation == "write":
                return self._write_file(path, content)
            elif operation == "list":
                return self._list_files(path)
            elif operation == "exists":
                return self._file_exists(path)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as e:
            self.logger.error(f"Error performing file operation: {str(e)}")
            return {
                "operation": operation,
                "path": path,
                "error": str(e),
            }
    
    def _normalize_path(self, path: str) -> str:
        """
        Normalize and validate a file path.
        
        Args:
            path: The file path.
            
        Returns:
            The normalized path.
            
        Raises:
            ValueError: If the path is invalid or not allowed.
        """
        # Make the path absolute
        if not os.path.isabs(path):
            path = os.path.join(self.base_directory, path)
        
        # Normalize the path
        path = os.path.normpath(path)
        
        # Check if the path is within the base directory
        if not path.startswith(self.base_directory):
            raise ValueError(f"Path must be within the base directory: {self.base_directory}")
        
        # Check if the file extension is allowed
        if self.allowed_extensions and os.path.splitext(path)[1] not in self.allowed_extensions:
            raise ValueError(f"File extension not allowed. Allowed extensions: {self.allowed_extensions}")
        
        return path
    
    def _read_file(self, path: str) -> Dict[str, Any]:
        """
        Read a file.
        
        Args:
            path: The file path.
            
        Returns:
            The file content.
        """
        if not os.path.exists(path):
            return {
                "operation": "read",
                "path": path,
                "error": "File not found",
            }
        
        with open(path, "r") as f:
            content = f.read()
        
        return {
            "operation": "read",
            "path": path,
            "content": content,
        }
    
    def _write_file(self, path: str, content: str) -> Dict[str, Any]:
        """
        Write to a file.
        
        Args:
            path: The file path.
            content: The content to write.
            
        Returns:
            The result of the operation.
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w") as f:
            f.write(content)
        
        return {
            "operation": "write",
            "path": path,
            "success": True,
        }
    
    def _list_files(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        List files in a directory.
        
        Args:
            path: The directory path.
            
        Returns:
            List of files.
        """
        directory = path or self.base_directory
        
        if not os.path.exists(directory):
            return {
                "operation": "list",
                "path": directory,
                "error": "Directory not found",
            }
        
        if not os.path.isdir(directory):
            return {
                "operation": "list",
                "path": directory,
                "error": "Path is not a directory",
            }
        
        files = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            files.append({
                "name": item,
                "path": item_path,
                "is_directory": os.path.isdir(item_path),
                "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None,
            })
        
        return {
            "operation": "list",
            "path": directory,
            "files": files,
        }
    
    def _file_exists(self, path: str) -> Dict[str, Any]:
        """
        Check if a file exists.
        
        Args:
            path: The file path.
            
        Returns:
            Whether the file exists.
        """
        return {
            "operation": "exists",
            "path": path,
            "exists": os.path.exists(path),
            "is_directory": os.path.isdir(path) if os.path.exists(path) else None,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool to a dictionary representation.
        
        Returns:
            Dictionary representation of the tool.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "operation": {
                    "type": "string",
                    "description": "The operation to perform (read, write, list, exists)",
                    "enum": ["read", "write", "list", "exists"],
                },
                "path": {
                    "type": "string",
                    "description": "The file or directory path",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write (for write operations)",
                },
            },
        }
