
"""
Single Page Crawler Implementation for Project Citadel.

This crawler is designed to fetch and process a single web page.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import aiohttp
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse

from citadel.crawlers.base_crawler import BaseCrawler, CrawlResult
from citadel.utils.crawler_utils import validate_url


class SinglePageCrawler(BaseCrawler):
    """
    Crawler implementation for fetching and processing a single web page.
    
    This crawler is optimized for single-page extraction with robust error handling
    and configurable data extraction patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SinglePageCrawler with configuration.
        
        Args:
            config: Dictionary containing crawler configuration parameters including:
                - user_agent: Custom user agent string
                - timeout: Request timeout in seconds
                - max_retries: Maximum number of retry attempts
                - retry_delay: Delay between retries in seconds
                - headers: Additional HTTP headers
                - extraction_patterns: Dictionary of CSS selectors or XPath expressions
                - validate_ssl: Whether to validate SSL certificates
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.extraction_patterns = config.get('extraction_patterns', {})
        self.validate_ssl = config.get('validate_ssl', True)
        self.session = None
        self.last_request_time = 0
        
    async def __aenter__(self):
        """Set up resources when entering async context."""
        if self.session is None:
            self.session = aiohttp.ClientSession(headers=self.headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting async context."""
        if self.session is not None:
            await self.session.close()
            self.session = None
    
    async def crawl(self, url: str, **kwargs) -> CrawlResult:
        """
        Crawl a single web page and return the result.
        
        Args:
            url: The URL to crawl
            **kwargs: Additional parameters:
                - custom_headers: Dict of headers to merge with default headers
                - extraction_override: Dict of extraction patterns to override defaults
                
        Returns:
            CrawlResult containing the crawled data
        """
        if not await self.validate_url(url):
            return await self.handle_error(url, ValueError(f"Invalid URL: {url}"))
        
        custom_headers = kwargs.get('custom_headers', {})
        merged_headers = {**self.headers, **custom_headers}
        
        # Create session if not exists
        if self.session is None:
            self.session = aiohttp.ClientSession(headers=self.headers)
            
        # Apply rate limiting
        await self.handle_rate_limits()
        
        # Track request time for rate limiting
        self.last_request_time = asyncio.get_event_loop().time()
        
        try:
            for attempt in range(self.max_retries):
                try:
                    async with self.session.get(
                        url, 
                        timeout=self.timeout,
                        headers=merged_headers,
                        ssl=None if not self.validate_ssl else True
                    ) as response:
                        html = await response.text()
                        
                        if response.status >= 400:
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay * (attempt + 1))
                                continue
                            return CrawlResult(
                                url=url,
                                content=html,
                                metadata={"status_message": response.reason},
                                status_code=response.status,
                                success=False
                            )
                        
                        # Extract data from HTML
                        extraction_override = kwargs.get('extraction_override', {})
                        extraction_patterns = {**self.extraction_patterns, **extraction_override}
                        extracted_data = await self.extract_data(html, url, extraction_patterns)
                        
                        return CrawlResult(
                            url=url,
                            content=html,
                            metadata=extracted_data,
                            status_code=response.status,
                            success=True
                        )
                except asyncio.TimeoutError:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        return await self.handle_error(url, asyncio.TimeoutError(f"Request timed out after {self.timeout} seconds"))
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        return await self.handle_error(url, e)
        except Exception as e:
            return await self.handle_error(url, e)
    
    async def extract_data(self, html: str, url: str, extraction_patterns: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Extract structured data from the HTML content.
        
        Args:
            html: The HTML content to parse
            url: The URL the content was fetched from
            extraction_patterns: Optional dictionary of CSS selectors or XPath expressions
            
        Returns:
            Dictionary containing extracted data with standardized error handling
        """
        from citadel.utils.crawler_utils import ParsingError, get_soup

        if extraction_patterns is None:
            extraction_patterns = self.extraction_patterns
            
        result = {
            "url": url,
            "timestamp": asyncio.get_event_loop().time(),
            "success": True,
            "partial_success": False,
            "errors": []
        }
        
        try:
            # Use the utility function to get BeautifulSoup object with error handling
            soup = get_soup(html, 'html.parser')
            
            # Extract title with error handling
            try:
                result["title"] = soup.title.text.strip() if soup.title else ""
            except Exception as e:
                result["errors"].append({"type": "title_extraction", "message": str(e)})
                result["partial_success"] = True
            
            # Extract metadata with error handling
            try:
                meta_tags = {}
                for meta in soup.find_all('meta'):
                    name = meta.get('name') or meta.get('property')
                    if name:
                        meta_tags[name] = meta.get('content', '')
                result["meta_tags"] = meta_tags
            except Exception as e:
                result["errors"].append({"type": "meta_extraction", "message": str(e)})
                result["partial_success"] = True
                result["meta_tags"] = {}
            
            # Apply custom extraction patterns with error handling
            extraction_errors = []
            for key, selector in extraction_patterns.items():
                try:
                    if selector.startswith('//'):  # XPath
                        # For XPath, we'd need lxml, but we'll use a placeholder for now
                        result[key] = "XPath extraction requires lxml"
                    else:  # CSS selector
                        elements = soup.select(selector)
                        if elements:
                            if len(elements) == 1:
                                result[key] = elements[0].get_text(strip=True)
                            else:
                                result[key] = [el.get_text(strip=True) for el in elements]
                except Exception as e:
                    extraction_errors.append({"selector": key, "message": str(e)})
                    result["partial_success"] = True
            
            if extraction_errors:
                result["errors"].append({"type": "selector_extraction", "details": extraction_errors})
            
            return result
        except ParsingError as e:
            # Handle specific parsing errors from the utility function
            self.logger.error(f"Parsing error extracting data from {url}: {str(e)}")
            return {
                "url": url,
                "timestamp": asyncio.get_event_loop().time(),
                "success": False,
                "error_type": "parsing_error",
                "error_message": str(e),
                "errors": [{"type": "html_parsing", "message": str(e)}]
            }
        except Exception as e:
            # Handle any other unexpected errors
            self.logger.error(f"Error extracting data from {url}: {str(e)}")
            return {
                "url": url,
                "timestamp": asyncio.get_event_loop().time(),
                "success": False,
                "error_type": "extraction_error",
                "error_message": str(e),
                "errors": [{"type": "unexpected", "message": str(e)}]
            }
    
    async def handle_rate_limits(self) -> None:
        """
        Implement rate limiting to avoid overloading target servers.
        
        This method ensures a minimum delay between requests based on the
        crawler's configuration.
        """
        min_delay = self.config.get('min_request_interval', 1.0)  # Default: 1 second between requests
        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < min_delay:
            delay = min_delay - elapsed
            await asyncio.sleep(delay)
    
    def validate_url(self, url: str) -> bool:
        """Use the utility function to validate URLs."""
        # Call the imported validate_url function with appropriate parameters
        return validate_url(
            url=url,
            base_url=self.base_url,
            allowed_domains=getattr(self, 'allowed_domains', None),
            allowed_schemes=getattr(self, 'allowed_schemes', None),
            url_patterns=getattr(self, 'url_patterns', None),
            excluded_patterns=getattr(self, 'excluded_patterns', None)
        )

