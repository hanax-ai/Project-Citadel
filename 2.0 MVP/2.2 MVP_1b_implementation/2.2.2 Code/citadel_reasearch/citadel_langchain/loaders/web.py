
"""
Web document loader for Project Citadel LangChain integration.
"""

import logging
import requests
from typing import Any, Dict, List, Optional, Union
from bs4 import BeautifulSoup
import time

from langchain_core.documents import Document

from citadel_core.logging import get_logger

from .base import BaseLoader


class WebLoader(BaseLoader):
    """Loader for web content."""
    
    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the web loader.
        
        Args:
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
            logger: Logger instance.
        """
        super().__init__(logger)
        
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = self._get_session()
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds
    
    def _get_session(self) -> requests.Session:
        """Create and configure a requests session."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'CitadelRevisions/1.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        return session
    
    def _rate_limit(self) -> None:
        """Implement rate limiting to avoid overloading the server."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
    
    def _safe_request(self, url: str) -> Optional[requests.Response]:
        """Make a safe HTTP request with error handling and retries."""
        self._rate_limit()
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    url=url,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response
                
            except requests.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt+1}/{self.max_retries+1}): {url}, Error: {str(e)}")
                
                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Max retries exceeded for URL: {url}")
                    return None
    
    def _parse_html(self, html_content: str) -> BeautifulSoup:
        """Parse HTML content using BeautifulSoup."""
        return BeautifulSoup(html_content, 'html.parser')
    
    def load(self, source: Union[str, List[str]], **kwargs) -> List[Document]:
        """
        Load documents from web URLs.
        
        Args:
            source: URL or list of URLs to load documents from.
            **kwargs: Additional loading parameters.
            
        Returns:
            List of loaded documents.
        """
        if isinstance(source, str):
            urls = [source]
        else:
            urls = source
        
        documents = []
        
        for url in urls:
            try:
                # Make the request
                response = self._safe_request(url)
                if response is None:
                    self.logger.warning(f"Failed to load content from URL: {url}")
                    continue
                
                # Parse the HTML
                soup = self._parse_html(response.text)
                
                # Extract the main content
                # This is a simple implementation - in a real-world scenario,
                # you would use more sophisticated content extraction
                main_content = soup.get_text(separator="\n", strip=True)
                
                # Create a document
                document = self._create_document(
                    text=main_content,
                    metadata={
                        "source": url,
                        "title": soup.title.string if soup.title else url,
                        "content_type": "text/html",
                    }
                )
                
                documents.append(document)
                
            except Exception as e:
                self.logger.error(f"Error loading content from URL {url}: {str(e)}")
        
        return documents
