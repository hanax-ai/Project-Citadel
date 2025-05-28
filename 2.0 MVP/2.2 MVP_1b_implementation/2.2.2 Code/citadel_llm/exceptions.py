
"""
Custom exceptions for the Citadel LLM package.

This module defines custom exceptions for error handling in the Citadel LLM package.
"""

from typing import Optional, Dict, Any


class CitadelLLMError(Exception):
    """Base exception for all Citadel LLM errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message.
            details: Additional error details.
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ModelNotFoundError(CitadelLLMError):
    """Exception raised when a model is not found."""
    
    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            model_name: Name of the model that was not found.
            details: Additional error details.
        """
        message = f"Model '{model_name}' not found"
        super().__init__(message, details)
        self.model_name = model_name


class ModelLoadError(CitadelLLMError):
    """Exception raised when a model fails to load."""
    
    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            model_name: Name of the model that failed to load.
            details: Additional error details.
        """
        message = f"Failed to load model '{model_name}'"
        super().__init__(message, details)
        self.model_name = model_name


class GatewayConnectionError(CitadelLLMError):
    """Exception raised when connection to the gateway fails."""
    
    def __init__(self, gateway_url: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            gateway_url: URL of the gateway that failed to connect.
            details: Additional error details.
        """
        message = f"Failed to connect to gateway at '{gateway_url}'"
        super().__init__(message, details)
        self.gateway_url = gateway_url


class GatewayTimeoutError(CitadelLLMError):
    """Exception raised when a gateway request times out."""
    
    def __init__(self, gateway_url: str, timeout: float, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            gateway_url: URL of the gateway that timed out.
            timeout: Timeout value in seconds.
            details: Additional error details.
        """
        message = f"Request to gateway at '{gateway_url}' timed out after {timeout} seconds"
        super().__init__(message, details)
        self.gateway_url = gateway_url
        self.timeout = timeout


class GatewayResponseError(CitadelLLMError):
    """Exception raised when the gateway returns an error response."""
    
    def __init__(self, status_code: int, response_text: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            status_code: HTTP status code.
            response_text: Response text from the gateway.
            details: Additional error details.
        """
        message = f"Gateway returned error response: {status_code} - {response_text}"
        super().__init__(message, details)
        self.status_code = status_code
        self.response_text = response_text


class InvalidRequestError(CitadelLLMError):
    """Exception raised when a request is invalid."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message.
            details: Additional error details.
        """
        super().__init__(message, details)


class PromptTemplateError(CitadelLLMError):
    """Exception raised when there is an error with a prompt template."""
    
    def __init__(self, template_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            template_name: Name of the template that caused the error.
            message: Error message.
            details: Additional error details.
        """
        full_message = f"Error in prompt template '{template_name}': {message}"
        super().__init__(full_message, details)
        self.template_name = template_name
