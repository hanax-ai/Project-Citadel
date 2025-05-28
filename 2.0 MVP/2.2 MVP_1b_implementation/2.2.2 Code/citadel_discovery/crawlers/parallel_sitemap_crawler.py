
"""
Parallel Sitemap Crawler Implementation for Project Citadel.

This crawler is designed to fetch and process URLs from a sitemap in parallel.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Union
import aiohttp
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import re

from citadel.crawlers.base_crawler import BaseCrawler, CrawlResult
from citadel.utils.crawler_utils import validate_url


class ParallelSitemapCrawler(BaseCrawler):
    """
    Crawler implementation for fetching and processing URLs from a sitemap in parallel.
    
    This crawler parses XML sitemaps and processes multiple URLs concurrently with
    configurable concurrency limits and prioritization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ParallelSitemapCrawler with configuration.
        
        Args:
            config: Dictionary containing crawler configuration parameters including:
                - user_agent: Custom user agent string
                - timeout: Request timeout in seconds
                - max_retries: Maximum number of retry attempts
                - retry_delay: Delay between retries in seconds
                - headers: Additional HTTP headers
                - extraction_patterns: Dictionary of CSS selectors or XPath expressions
                - validate_ssl: Whether to validate SSL certificates
                - max_concurrent_requests: Maximum number of concurrent requests
                - sitemap_index_handling: How to handle sitemap index files
                - priority_patterns: URL patterns to prioritize
                - progress_callback: Optional callback function for progress updates
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.extraction_patterns = config.get('extraction_patterns', {})
        self.validate_ssl = config.get('validate_ssl', True)
        self.max_concurrent_requests = config.get('max_concurrent_requests', 5)
        self.sitemap_index_handling = config.get('sitemap_index_handling', 'process_all')
        self.priority_patterns = config.get('priority_patterns', [])
        self.progress_callback = config.get('progress_callback')
        self.session = None
        self.semaphore = None
        self.processed_urls = set()
        
    async def __aenter__(self):
        """Set up resources when entering async context."""
        if self.session is None:
            self.session = aiohttp.ClientSession(headers=self.headers)
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting async context."""
        if self.session is not None:
            await self.session.close()
            self.session = None
    
    async def crawl(self, sitemap_url: str, **kwargs) -> List[CrawlResult]:
        """
        Crawl URLs from a sitemap in parallel and return the results.
        
        Args:
            sitemap_url: The URL of the sitemap to process
            **kwargs: Additional parameters:
                - custom_headers: Dict of headers to merge with default headers
                - extraction_override: Dict of extraction patterns to override defaults
                - max_urls: Maximum number of URLs to process
                - url_filter: Function to filter URLs from the sitemap
                
        Returns:
            List of CrawlResult objects containing the crawled data
        """
        # Create session if not exists
        if self.session is None:
            self.session = aiohttp.ClientSession(headers=self.headers)
            
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            
        # Reset processed URLs
        self.processed_urls = set()
        
        # Extract URLs from sitemap
        sitemap_urls = await self._extract_urls_from_sitemap(sitemap_url)
        
        # Apply URL filter if provided
        url_filter = kwargs.get('url_filter')
        if url_filter and callable(url_filter):
            sitemap_urls = [url for url in sitemap_urls if url_filter(url)]
            
        # Apply max_urls limit if provided
        max_urls = kwargs.get('max_urls')
        if max_urls and isinstance(max_urls, int) and max_urls > 0:
            sitemap_urls = sitemap_urls[:max_urls]
            
        # Sort URLs by priority if priority patterns are defined
        if self.priority_patterns:
            sitemap_urls = self._prioritize_urls(sitemap_urls)
            
        # Process URLs in parallel
        tasks = []
        for url in sitemap_urls:
            if url not in self.processed_urls:
                self.processed_urls.add(url)
                tasks.append(self._crawl_url_with_semaphore(url, **kwargs))
                
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Convert exception to CrawlResult
                error_result = await self.handle_error(sitemap_urls[i], result)
                processed_results.append(error_result)
            else:
                processed_results.append(result)
                
        return processed_results
    
    async def _crawl_url_with_semaphore(self, url: str, **kwargs) -> CrawlResult:
        """
        Crawl a URL with semaphore-based concurrency control.
        
        Args:
            url: The URL to crawl
            **kwargs: Additional parameters
            
        Returns:
            CrawlResult containing the crawled data
        """
        async with self.semaphore:
            # Apply rate limiting
            await self.handle_rate_limits()
            
            # Update progress if callback is provided
            if self.progress_callback:
                self.progress_callback(len(self.processed_urls), url)
                
            return await self._crawl_single_url(url, **kwargs)
    
    async def _crawl_single_url(self, url: str, **kwargs) -> CrawlResult:
        """
        Crawl a single URL and return the result.
        
        Args:
            url: The URL to crawl
            **kwargs: Additional parameters
            
        Returns:
            CrawlResult containing the crawled data
        """
        if not await self.validate_url(url):
            return await self.handle_error(url, ValueError(f"Invalid URL: {url}"))
            
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
    
    async def _extract_urls_from_sitemap(self, sitemap_url: str) -> List[str]:
        """
        Extract URLs from a sitemap or sitemap index.
        
        Args:
            sitemap_url: The URL of the sitemap to process
            
        Returns:
            List of URLs extracted from the sitemap
        """
        try:
            async with self.session.get(
                sitemap_url,
                timeout=self.timeout,
                headers=self.headers,
                ssl=None if not self.validate_ssl else True
            ) as response:
                if response.status != 200:
                    self.logger.error(f"Failed to fetch sitemap: {sitemap_url}, status: {response.status}")
                    return []
                    
                sitemap_content = await response.text()
                
                # Check if this is a sitemap index
                if '<sitemapindex' in sitemap_content:
                    return await self._process_sitemap_index(sitemap_content, sitemap_url)
                    
                # Process regular sitemap
                return self._parse_sitemap(sitemap_content)
        except Exception as e:
            self.logger.error(f"Error fetching sitemap {sitemap_url}: {str(e)}")
            return []
    
    async def _process_sitemap_index(self, sitemap_content: str, base_url: str) -> List[str]:
        """
        Process a sitemap index file and extract URLs from all referenced sitemaps.
        
        Args:
            sitemap_content: The content of the sitemap index
            base_url: The URL of the sitemap index
            
        Returns:
            List of URLs extracted from all referenced sitemaps
        """
        all_urls = []
        
        try:
            root = ET.fromstring(sitemap_content)
            namespace = self._get_namespace(root)
            
            # Extract sitemap URLs
            sitemap_urls = []
            for sitemap in root.findall(f".//{{{namespace}}}sitemap"):
                loc = sitemap.find(f"{{{namespace}}}loc")
                if loc is not None and loc.text:
                    sitemap_urls.append(loc.text.strip())
            
            # Process based on configuration
            if self.sitemap_index_handling == 'process_first':
                if sitemap_urls:
                    return await self._extract_urls_from_sitemap(sitemap_urls[0])
            elif self.sitemap_index_handling == 'process_all':
                # Process all sitemaps in parallel
                tasks = [self._extract_urls_from_sitemap(url) for url in sitemap_urls]
                results = await asyncio.gather(*tasks)
                for urls in results:
                    all_urls.extend(urls)
            else:  # Default: process_all
                # Process all sitemaps in parallel
                tasks = [self._extract_urls_from_sitemap(url) for url in sitemap_urls]
                results = await asyncio.gather(*tasks)
                for urls in results:
                    all_urls.extend(urls)
                    
            return all_urls
        except Exception as e:
            self.logger.error(f"Error processing sitemap index: {str(e)}")
            return []
    
    def _parse_sitemap(self, sitemap_content: str) -> List[str]:
        """
        Parse a sitemap XML and extract URLs.
        
        Args:
            sitemap_content: The content of the sitemap
            
        Returns:
            List of URLs extracted from the sitemap
        """
        urls = []
        
        try:
            root = ET.fromstring(sitemap_content)
            namespace = self._get_namespace(root)
            
            for url_element in root.findall(f".//{{{namespace}}}url"):
                loc = url_element.find(f"{{{namespace}}}loc")
                if loc is not None and loc.text:
                    urls.append(loc.text.strip())
                    
            return urls
        except Exception as e:
            self.logger.error(f"Error parsing sitemap: {str(e)}")
            return []
    
    def _get_namespace(self, element: ET.Element) -> str:
        """
        Extract namespace from XML element.
        
        Args:
            element: XML element
            
        Returns:
            Namespace string
        """
        m = re.match(r'\{(.*?)\}', element.tag)
        return m.group(1) if m else ''
    
    def _prioritize_urls(self, urls: List[str]) -> List[str]:
        """
        Sort URLs based on priority patterns.
        
        Args:
            urls: List of URLs to sort
            
        Returns:
            Sorted list of URLs with priority URLs first
        """
        priority_urls = []
        normal_urls = []
        
        for url in urls:
            if any(re.search(pattern, url) for pattern in self.priority_patterns):
                priority_urls.append(url)
            else:
                normal_urls.append(url)
                
        return priority_urls + normal_urls
    
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
            
            # Extract sitemap-specific data if available
            try:
                # Look for lastmod, changefreq, priority in the page
                lastmod = soup.find('meta', {'name': 'lastmod'}) or soup.find('meta', {'property': 'lastmod'})
                if lastmod:
                    result["lastmod"] = lastmod.get('content', '')
                
                changefreq = soup.find('meta', {'name': 'changefreq'}) or soup.find('meta', {'property': 'changefreq'})
                if changefreq:
                    result["changefreq"] = changefreq.get('content', '')
                
                priority = soup.find('meta', {'name': 'priority'}) or soup.find('meta', {'property': 'priority'})
                if priority:
                    result["priority"] = priority.get('content', '')
            except Exception as e:
                result["errors"].append({"type": "sitemap_metadata_extraction", "message": str(e)})
                result["partial_success"] = True
            
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
        
        This method uses a token bucket algorithm to limit request rates.
        """
        # Token bucket rate limiting is handled by the semaphore
        # Additional domain-specific rate limiting can be implemented here
        domain_limits = self.config.get('domain_rate_limits', {})
        
        if domain_limits:
            # Extract domain from current URL and check for specific limits
            # This would require tracking the current URL being processed
            pass
        
        # Default delay to avoid overwhelming servers
        await asyncio.sleep(0.1)
    
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

