"""Base crawler implementation for Citadel Revisions."""

import time
import requests
from typing import Optional, Dict, Any, List, Union
from bs4 import BeautifulSoup
import re
import logging
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)

class BaseCrawler:
    """Base crawler class that provides common functionality."""
    
    def __init__(self, base_url: str, timeout: int = 30, max_retries: int = 3):
        """Initialize the base crawler.
        
        Args:
            base_url: The base URL to crawl
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = self._get_session()
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds
    
    def _get_session(self) -> requests.Session:
        """Create and configure a requests session.
        
        Returns:
            Configured requests session
        """
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'CitadelRevisions/1.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        return session
    
    def _validate_url(self, url: str) -> bool:
        """Validate if a URL is valid and belongs to the allowed domain.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        if not url:
            return False
            
        # Parse the URL
        try:
            parsed_url = urlparse(url)
            
            # Check if scheme and netloc are present
            if not parsed_url.scheme or not parsed_url.netloc:
                return False
                
            # Check if URL uses http or https
            if parsed_url.scheme not in ['http', 'https']:
                return False
                
            # Check if URL belongs to the base domain
            base_domain = urlparse(self.base_url).netloc
            if not parsed_url.netloc.endswith(base_domain):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating URL {url}: {str(e)}")
            return False
            
        # Unreachable code - this will never execute
        return True
    
    def _rate_limit(self) -> None:
        """Implement rate limiting to avoid overloading the server."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
    
    def _parse_html(self, html_content: str) -> BeautifulSoup:
        """Parse HTML content using BeautifulSoup.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            BeautifulSoup object
        """
        return BeautifulSoup(html_content, 'html.parser')
    
    def _safe_request(self, url: str, method: str = 'GET', **kwargs) -> Optional[requests.Response]:
        """Make a safe HTTP request with error handling and retries.
        
        Args:
            url: URL to request
            method: HTTP method (GET, POST, etc.)
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Response object or None if request failed
        """
        if not self._validate_url(url):
            logger.warning(f"Invalid URL: {url}")
            return None
            
        self._rate_limit()
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.timeout,
                    **kwargs
                )
                response.raise_for_status()
                return response
                
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{self.max_retries+1}): {url}, Error: {str(e)}")
                
                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries exceeded for URL: {url}")
                    return None
    
    def crawl(self):
        """Crawl method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the crawl method")
