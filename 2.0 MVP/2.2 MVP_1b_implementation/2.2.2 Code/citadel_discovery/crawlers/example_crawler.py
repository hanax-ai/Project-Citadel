"""Example crawler implementation using crawler_utils.

This example demonstrates how to use the utility functions from crawler_utils.py
to implement a crawler with minimal code duplication.
"""

import time
import requests
from typing import Optional, Dict, Any, List, Set
from bs4 import BeautifulSoup
import logging
from urllib.parse import urljoin

from citadel_revisions import crawler_utils

logger = logging.getLogger(__name__)

class ExampleCrawler:
    """Example crawler using utility functions from crawler_utils."""
    
    def __init__(self, base_url: str, timeout: int = 30, max_retries: int = 3):
        """Initialize the example crawler.
        
        Args:
            base_url: The base URL to crawl
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = crawler_utils.get_session(
            user_agent="ExampleCrawler/1.0",
            additional_headers={"X-Crawler": "ExampleCrawler"}
        )
        self.last_request_time = 0
        self.min_request_interval = 1.5  # seconds
        self.visited_urls: Set[str] = set()
        self.content_urls: Set[str] = set()
        
    def _validate_url(self, url: str) -> bool:
        """Validate if a URL is valid for this crawler.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        # Use the utility function with custom parameters for this crawler
        return crawler_utils.validate_url(
            url=url,
            base_url=self.base_url,
            url_patterns=['/content/', '/article/'],
            excluded_patterns=['/login', '/signup', '/admin'],
            callback=lambda u: self.content_urls.add(u) if '/content/' in u else None
        )
    
    def _rate_limit(self) -> None:
        """Apply rate limiting."""
        self.last_request_time = crawler_utils.apply_rate_limiting(
            self.last_request_time, 
            self.min_request_interval
        )
    
    def _safe_request(self, url: str, method: str = 'GET', **kwargs) -> Optional[requests.Response]:
        """Make a safe HTTP request.
        
        Args:
            url: URL to request
            method: HTTP method (GET, POST, etc.)
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Response object or None if request failed
        """
        return crawler_utils.safe_request(
            url=url,
            session=self.session,
            validate_func=self._validate_url,
            rate_limit_func=self._rate_limit,
            visited_urls=self.visited_urls,
            timeout=self.timeout,
            max_retries=self.max_retries,
            method=method,
            **kwargs
        )
    
    def _extract_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract content from a page.
        
        Args:
            soup: BeautifulSoup object
            url: URL of the page
            
        Returns:
            Dictionary with extracted content
        """
        content = {'url': url}
        
        # Extract title
        title_elem = soup.find('h1') or soup.find('title')
        if title_elem:
            content['title'] = title_elem.text.strip()
        
        # Extract main content
        main_elem = soup.find('main') or soup.find('article') or soup.find('div', {'class': 'content'})
        if main_elem:
            content['content'] = main_elem.get_text(separator=' ', strip=True)
        
        return content
    
    @crawler_utils.retry_on_error(max_retries=2)
    def crawl(self, max_pages: int = 50) -> List[Dict[str, Any]]:
        """Crawl the website and extract content.
        
        Args:
            max_pages: Maximum number of pages to crawl
            
        Returns:
            List of content dictionaries
        """
        contents = []
        to_visit = [self.base_url]
        visited_count = 0
        
        while to_visit and visited_count < max_pages:
            url = to_visit.pop(0)
            
            response = self._safe_request(url)
            if not response:
                continue
                
            visited_count += 1
            
            # Parse HTML
            soup = crawler_utils.parse_html(response.text)
            
            # Extract content if this is a content page
            if url in self.content_urls:
                content = self._extract_content(soup, url)
                contents.append(content)
            
            # Find more links to follow
            links = crawler_utils.extract_links(soup, url)
            for link in links:
                if link not in self.visited_urls and link not in to_visit:
                    to_visit.append(link)
        
        return contents


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    crawler = ExampleCrawler("https://example.com")
    results = crawler.crawl(max_pages=10)
    
    print(f"Found {len(results)} content pages:")
    for result in results:
        print(f"- {result.get('title', 'No title')} ({result['url']})")
