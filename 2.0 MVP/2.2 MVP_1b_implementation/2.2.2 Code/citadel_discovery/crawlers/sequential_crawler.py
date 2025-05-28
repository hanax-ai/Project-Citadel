
"""
Sequential Document Crawler Implementation for Project Citadel.

This crawler is designed to fetch and process multiple documents in sequence.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import aiohttp
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse, urljoin

from citadel.crawlers.base_crawler import BaseCrawler, CrawlResult
from citadel.utils.crawler_utils import validate_url


class SequentialCrawler(BaseCrawler):
    """
    Crawler implementation for fetching and processing multiple documents in sequence.
    
    This crawler processes a list of URLs one after another, maintaining order and
    providing detailed progress tracking.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SequentialCrawler with configuration.
        
        Args:
            config: Dictionary containing crawler configuration parameters including:
                - user_agent: Custom user agent string
                - timeout: Request timeout in seconds
                - max_retries: Maximum number of retry attempts
                - retry_delay: Delay between retries in seconds
                - headers: Additional HTTP headers
                - extraction_patterns: Dictionary of CSS selectors or XPath expressions
                - validate_ssl: Whether to validate SSL certificates
                - min_request_interval: Minimum time between requests in seconds
                - progress_callback: Optional callback function for progress updates
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.extraction_patterns = config.get('extraction_patterns', {})
        self.validate_ssl = config.get('validate_ssl', True)
        self.min_request_interval = config.get('min_request_interval', 1.0)
        self.progress_callback = config.get('progress_callback')
        self.session = None
        self.last_request_time = 0
        self.current_progress = 0
        self.total_urls = 0
        
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
    
    async def crawl(self, urls: Union[str, List[str]], **kwargs) -> List[CrawlResult]:
        """
        Crawl multiple URLs in sequence and return the results.
        
        Args:
            urls: A single URL string or list of URLs to crawl
            **kwargs: Additional parameters:
                - custom_headers: Dict of headers to merge with default headers
                - extraction_override: Dict of extraction patterns to override defaults
                - continue_on_error: Whether to continue crawling if an error occurs
                
        Returns:
            List of CrawlResult objects containing the crawled data
        """
        # Handle single URL case
        if isinstance(urls, str):
            urls = [urls]
            
        self.total_urls = len(urls)
        self.current_progress = 0
        
        # Create session if not exists
        if self.session is None:
            self.session = aiohttp.ClientSession(headers=self.headers)
            
        results = []
        continue_on_error = kwargs.get('continue_on_error', True)
        
        for url in urls:
            # Update progress
            self.current_progress += 1
            if self.progress_callback:
                self.progress_callback(self.current_progress, self.total_urls, url)
                
            # Validate URL
            if not await self.validate_url(url):
                error_result = await self.handle_error(url, ValueError(f"Invalid URL: {url}"))
                results.append(error_result)
                if not continue_on_error:
                    break
                continue
                
            # Apply rate limiting
            await self.handle_rate_limits()
            
            # Track request time for rate limiting
            self.last_request_time = asyncio.get_event_loop().time()
            
            try:
                # Perform the crawl for this URL
                result = await self._crawl_single_url(url, **kwargs)
                results.append(result)
                
                # Stop if error and continue_on_error is False
                if not result.success and not continue_on_error:
                    break
                    
            except Exception as e:
                error_result = await self.handle_error(url, e)
                results.append(error_result)
                if not continue_on_error:
                    break
        
        return results
    
    async def _crawl_single_url(self, url: str, **kwargs) -> CrawlResult:
        """
        Crawl a single URL and return the result.
        
        Args:
            url: The URL to crawl
            **kwargs: Additional parameters
            
        Returns:
            CrawlResult containing the crawled data
        """
        custom_headers = kwargs.get('custom_headers', {})
        merged_headers = {**self.headers, **custom_headers}
        
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
            
            # Extract document-specific data with error handling
            try:
                result["document_type"] = self._detect_document_type(soup, url)
            except Exception as e:
                result["errors"].append({"type": "document_type_detection", "message": str(e)})
                result["partial_success"] = True
                result["document_type"] = "UNKNOWN"
            
            # Extract links with error handling
            try:
                links = []
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if href and not href.startswith('#') and not href.startswith('javascript:'):
                        full_url = urljoin(url, href)
                        links.append({
                            "url": full_url,
                            "text": a.get_text(strip=True),
                            "title": a.get('title', '')
                        })
                result["links"] = links
            except Exception as e:
                result["errors"].append({"type": "links_extraction", "message": str(e)})
                result["partial_success"] = True
                result["links"] = []
            
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
    
    def _detect_document_type(self, soup: BeautifulSoup, url: str) -> str:
        """
        Detect the type of document based on content and URL.
        
        Args:
            soup: BeautifulSoup object of the parsed HTML
            url: The URL the content was fetched from
            
        Returns:
            String indicating the detected document type
        """
        # Check URL patterns
        if url.endswith(('.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx')):
            return url.split('.')[-1].upper()
            
        # Check content type from meta tags
        for meta in soup.find_all('meta'):
            if meta.get('http-equiv', '').lower() == 'content-type':
                content_type = meta.get('content', '')
                if 'application/pdf' in content_type:
                    return 'PDF'
                elif 'application/msword' in content_type:
                    return 'DOC'
                    
        # Check for common document structures
        if soup.find('article') or soup.find('main'):
            return 'ARTICLE'
        elif soup.find('form'):
            return 'FORM'
            
        return 'HTML'
    
    async def handle_rate_limits(self) -> None:
        """
        Implement rate limiting to avoid overloading target servers.
        
        This method ensures a minimum delay between requests based on the
        crawler's configuration.
        """
        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            delay = self.min_request_interval - elapsed
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

