
"""
Ollama Gateway client for the Citadel LLM package.

This module provides the OllamaGateway class for interacting with the Ollama API.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, AsyncIterator, Union

import httpx

from citadel_core.logging import get_logger
from citadel_core.utils import retry

from .exceptions import (
    GatewayConnectionError,
    GatewayTimeoutError,
    GatewayResponseError,
    ModelNotFoundError,
    InvalidRequestError
)


class OllamaGateway:
    """Client for interacting with the Ollama API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Ollama gateway client.
        
        Args:
            base_url: Base URL of the Ollama API.
            timeout: Request timeout in seconds.
            logger: Logger instance.
        """
        self.base_url = base_url
        self.timeout = timeout
        self.logger = logger or get_logger("citadel.llm.gateway")
        
        # Default headers
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], httpx.Response]:
        """
        Make a request to the Ollama API.
        
        Args:
            method: HTTP method.
            endpoint: API endpoint.
            data: Request data.
            stream: Whether to stream the response.
            
        Returns:
            If stream is False, returns the response as a dictionary.
            If stream is True, returns the raw response object.
            
        Raises:
            GatewayConnectionError: If connection to the gateway fails.
            GatewayTimeoutError: If the request times out.
            GatewayResponseError: If the gateway returns an error response.
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=self.headers, params=data)
                else:  # POST
                    if stream:
                        response = await client.post(
                            url,
                            headers=self.headers,
                            json=data,
                            stream=True
                        )
                        if response.status_code != 200:
                            error_text = await response.aread()
                            raise GatewayResponseError(
                                response.status_code,
                                error_text.decode("utf-8")
                            )
                        return response
                    else:
                        response = await client.post(url, headers=self.headers, json=data)
                
                if response.status_code != 200:
                    raise GatewayResponseError(response.status_code, response.text)
                
                return response.json()
        
        except httpx.ConnectError as e:
            self.logger.error(f"Failed to connect to Ollama API: {e}")
            raise GatewayConnectionError(self.base_url, {"error": str(e)})
        
        except httpx.TimeoutException as e:
            self.logger.error(f"Request to Ollama API timed out: {e}")
            raise GatewayTimeoutError(self.base_url, self.timeout, {"error": str(e)})
        
        except httpx.HTTPError as e:
            self.logger.error(f"HTTP error from Ollama API: {e}")
            raise GatewayResponseError(
                getattr(e, "status_code", 500),
                str(e),
                {"error": str(e)}
            )
    
    @retry(max_attempts=3, delay=1.0, backoff_factor=2.0, exceptions=(GatewayConnectionError,))
    async def generate(
        self,
        prompt: str,
        model: str,
        options: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
        format: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        stream: bool = False,
        raw: bool = False,
        keep_alive: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt.
            model: Model name.
            options: Model parameters.
            system: System message.
            format: Response format.
            template: Prompt template.
            context: Context tokens from previous responses.
            stream: Whether to stream the response.
            raw: Whether to send the prompt without formatting.
            keep_alive: Duration to keep the model loaded.
            
        Returns:
            Response from the Ollama API.
            
        Raises:
            GatewayConnectionError: If connection to the gateway fails.
            GatewayTimeoutError: If the request times out.
            GatewayResponseError: If the gateway returns an error response.
            ModelNotFoundError: If the model is not found.
            InvalidRequestError: If the request is invalid.
        """
        data = {
            "prompt": prompt,
            "model": model,
            "stream": False  # We handle streaming separately
        }
        
        if options:
            data["options"] = options
        
        if system:
            data["system"] = system
        
        if format:
            data["format"] = format
        
        if template:
            data["template"] = template
        
        if context:
            data["context"] = context
        
        if raw:
            data["raw"] = True
        
        if keep_alive:
            data["keep_alive"] = keep_alive
        
        try:
            return await self._make_request("POST", "/api/generate", data)
        except GatewayResponseError as e:
            if e.status_code == 404:
                raise ModelNotFoundError(model, {"error": e.response_text})
            elif e.status_code == 400:
                raise InvalidRequestError(e.response_text, {"error": e.response_text})
            else:
                raise
    
    async def generate_stream(
        self,
        prompt: str,
        model: str,
        options: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
        format: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        raw: bool = False,
        keep_alive: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Generate text from a prompt with streaming.
        
        Args:
            prompt: Input prompt.
            model: Model name.
            options: Model parameters.
            system: System message.
            format: Response format.
            template: Prompt template.
            context: Context tokens from previous responses.
            raw: Whether to send the prompt without formatting.
            keep_alive: Duration to keep the model loaded.
            
        Yields:
            Chunks of generated text.
            
        Raises:
            GatewayConnectionError: If connection to the gateway fails.
            GatewayTimeoutError: If the request times out.
            GatewayResponseError: If the gateway returns an error response.
            ModelNotFoundError: If the model is not found.
            InvalidRequestError: If the request is invalid.
        """
        data = {
            "prompt": prompt,
            "model": model,
            "stream": True
        }
        
        if options:
            data["options"] = options
        
        if system:
            data["system"] = system
        
        if format:
            data["format"] = format
        
        if template:
            data["template"] = template
        
        if context:
            data["context"] = context
        
        if raw:
            data["raw"] = True
        
        if keep_alive:
            data["keep_alive"] = keep_alive
        
        try:
            response = await self._make_request("POST", "/api/generate", data, stream=True)
            
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                try:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to decode JSON from stream: {line}")
                    continue
        
        except GatewayResponseError as e:
            if e.status_code == 404:
                raise ModelNotFoundError(model, {"error": e.response_text})
            elif e.status_code == 400:
                raise InvalidRequestError(e.response_text, {"error": e.response_text})
            else:
                raise
    
    @retry(max_attempts=3, delay=1.0, backoff_factor=2.0, exceptions=(GatewayConnectionError,))
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        options: Optional[Dict[str, Any]] = None,
        format: Optional[str] = None,
        stream: bool = False,
        keep_alive: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a chat response.
        
        Args:
            messages: List of messages in the conversation.
            model: Model name.
            options: Model parameters.
            format: Response format.
            stream: Whether to stream the response.
            keep_alive: Duration to keep the model loaded.
            
        Returns:
            Response from the Ollama API.
            
        Raises:
            GatewayConnectionError: If connection to the gateway fails.
            GatewayTimeoutError: If the request times out.
            GatewayResponseError: If the gateway returns an error response.
            ModelNotFoundError: If the model is not found.
            InvalidRequestError: If the request is invalid.
        """
        data = {
            "messages": messages,
            "model": model,
            "stream": False  # We handle streaming separately
        }
        
        if options:
            data["options"] = options
        
        if format:
            data["format"] = format
        
        if keep_alive:
            data["keep_alive"] = keep_alive
        
        try:
            return await self._make_request("POST", "/api/chat", data)
        except GatewayResponseError as e:
            if e.status_code == 404:
                raise ModelNotFoundError(model, {"error": e.response_text})
            elif e.status_code == 400:
                raise InvalidRequestError(e.response_text, {"error": e.response_text})
            else:
                raise
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        options: Optional[Dict[str, Any]] = None,
        format: Optional[str] = None,
        keep_alive: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Generate a chat response with streaming.
        
        Args:
            messages: List of messages in the conversation.
            model: Model name.
            options: Model parameters.
            format: Response format.
            keep_alive: Duration to keep the model loaded.
            
        Yields:
            Chunks of generated text.
            
        Raises:
            GatewayConnectionError: If connection to the gateway fails.
            GatewayTimeoutError: If the request times out.
            GatewayResponseError: If the gateway returns an error response.
            ModelNotFoundError: If the model is not found.
            InvalidRequestError: If the request is invalid.
        """
        data = {
            "messages": messages,
            "model": model,
            "stream": True
        }
        
        if options:
            data["options"] = options
        
        if format:
            data["format"] = format
        
        if keep_alive:
            data["keep_alive"] = keep_alive
        
        try:
            response = await self._make_request("POST", "/api/chat", data, stream=True)
            
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                try:
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to decode JSON from stream: {line}")
                    continue
        
        except GatewayResponseError as e:
            if e.status_code == 404:
                raise ModelNotFoundError(model, {"error": e.response_text})
            elif e.status_code == 400:
                raise InvalidRequestError(e.response_text, {"error": e.response_text})
            else:
                raise
    
    @retry(max_attempts=3, delay=1.0, backoff_factor=2.0, exceptions=(GatewayConnectionError,))
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of models.
            
        Raises:
            GatewayConnectionError: If connection to the gateway fails.
            GatewayTimeoutError: If the request times out.
            GatewayResponseError: If the gateway returns an error response.
        """
        response = await self._make_request("GET", "/api/tags")
        return response.get("models", [])
    
    @retry(max_attempts=3, delay=1.0, backoff_factor=2.0, exceptions=(GatewayConnectionError,))
    async def model_info(self, model: str) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model: Model name.
            
        Returns:
            Model information.
            
        Raises:
            GatewayConnectionError: If connection to the gateway fails.
            GatewayTimeoutError: If the request times out.
            GatewayResponseError: If the gateway returns an error response.
            ModelNotFoundError: If the model is not found.
        """
        try:
            return await self._make_request("POST", "/api/show", {"name": model})
        except GatewayResponseError as e:
            if e.status_code == 404:
                raise ModelNotFoundError(model, {"error": e.response_text})
            else:
                raise
    
    @retry(max_attempts=3, delay=1.0, backoff_factor=2.0, exceptions=(GatewayConnectionError,))
    async def pull_model(self, model: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Pull a model from the Ollama library.
        
        Args:
            model: Model name.
            
        Yields:
            Progress updates.
            
        Raises:
            GatewayConnectionError: If connection to the gateway fails.
            GatewayTimeoutError: If the request times out.
            GatewayResponseError: If the gateway returns an error response.
        """
        try:
            response = await self._make_request("POST", "/api/pull", {"name": model}, stream=True)
            
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to decode JSON from stream: {line}")
                    continue
        
        except GatewayResponseError as e:
            if e.status_code == 404:
                raise ModelNotFoundError(model, {"error": e.response_text})
            else:
                raise
    
    @retry(max_attempts=3, delay=1.0, backoff_factor=2.0, exceptions=(GatewayConnectionError,))
    async def delete_model(self, model: str) -> Dict[str, Any]:
        """
        Delete a model.
        
        Args:
            model: Model name.
            
        Returns:
            Response from the Ollama API.
            
        Raises:
            GatewayConnectionError: If connection to the gateway fails.
            GatewayTimeoutError: If the request times out.
            GatewayResponseError: If the gateway returns an error response.
            ModelNotFoundError: If the model is not found.
        """
        try:
            return await self._make_request("DELETE", "/api/delete", {"name": model})
        except GatewayResponseError as e:
            if e.status_code == 404:
                raise ModelNotFoundError(model, {"error": e.response_text})
            else:
                raise
    
    @retry(max_attempts=3, delay=1.0, backoff_factor=2.0, exceptions=(GatewayConnectionError,))
    async def generate_embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str,
        keep_alive: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate embeddings for text.
        
        Args:
            input_text: Text to generate embeddings for.
            model: Model name.
            keep_alive: Duration to keep the model loaded.
            
        Returns:
            Response from the Ollama API.
            
        Raises:
            GatewayConnectionError: If connection to the gateway fails.
            GatewayTimeoutError: If the request times out.
            GatewayResponseError: If the gateway returns an error response.
            ModelNotFoundError: If the model is not found.
            InvalidRequestError: If the request is invalid.
        """
        data = {
            "input": input_text,
            "model": model
        }
        
        if keep_alive:
            data["keep_alive"] = keep_alive
        
        try:
            return await self._make_request("POST", "/api/embeddings", data)
        except GatewayResponseError as e:
            if e.status_code == 404:
                raise ModelNotFoundError(model, {"error": e.response_text})
            elif e.status_code == 400:
                raise InvalidRequestError(e.response_text, {"error": e.response_text})
            else:
                raise
