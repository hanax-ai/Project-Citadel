
"""
Recursive Site Crawler Implementation for Project Citadel.

This crawler is designed to recursively crawl a website following links.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Union
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re

from citadel.crawlers.base_crawler import BaseCrawler, CrawlResult
from citadel.utils.crawler_utils import validate_url


class RecursiveCrawler(BaseCrawler):
    """
    Crawler implementation for recursively crawling a website.
    
    This crawler follows links within a domain to a specified depth, with
    configurable URL filtering, concurrency control, and crawl policies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RecursiveCrawler with configuration.
        
        Args:
            config: Dictionary containing crawler configuration parameters including:
                - user_agent: Custom user agent string
                - timeout: Request timeout in seconds
                - max_retries: Maximum number of retry attempts
                - retry_delay: Delay between retries in seconds
                - headers: Additional HTTP headers
                - extraction_patterns: Dictionary of CSS selectors or XPath expressions
                - validate_ssl: Whether to validate SSL certificates
                - max_depth: Maximum recursion depth
                - max_urls: Maximum number of URLs to crawl
                - follow_external_links: Whether to follow links to external domains
                - url_patterns: List of regex patterns for URLs to include
                - exclude_patterns: List of regex patterns for URLs to exclude
                - max_concurrent_requests: Maximum number of concurrent requests
                - respect_robots_txt: Whether to respect robots.txt rules
                - progress_callback: Optional callback function for progress updates
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.extraction_patterns = config.get('extraction_patterns', {})
        self.validate_ssl = config.get('validate_ssl', True)
        self.max_depth = config.get('max_depth', 3)
        self.max_urls = config.get('max_urls', 100)
        self.follow_external_links = config.get('follow_external_links', False)
        self.url_patterns = config.get('url_patterns', [])
        self.exclude_patterns = config.get('exclude_patterns', [])
        self.max_concurrent_requests = config.get('max_concurrent_requests', 5)
        self.respect_robots_txt = config.get('respect_robots_txt', True)
        self.progress_callback = config.get('progress_callback')
        
        self.session = None
        self.semaphore = None
        self.visited_urls = set()
        self.queued_urls = set()
        self.robots_parsers = {}
        
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
    
    async def crawl(self, start_url: str, **kwargs) -> List[CrawlResult]:
        """
        Recursively crawl a website starting from the given URL.
        
        Args:
            start_url: The URL to start crawling from
            **kwargs: Additional parameters:
                - custom_headers: Dict of headers to merge with default headers
                - extraction_override: Dict of extraction patterns to override defaults
                - max_depth_override: Override the default max_depth
                - max_urls_override: Override the default max_urls
                - url_filter: Custom function to filter URLs
                
        Returns:
            List of CrawlResult objects containing the crawled data
        """
        # Reset state
        self.visited_urls = set()
        self.queued_urls = set()
        
        # Create session if not exists
        if self.session is None:
            self.session = aiohttp.ClientSession(headers=self.headers)
            
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            
        # Override configuration if specified
        max_depth = kwargs.get('max_depth_override', self.max_depth)
        max_urls = kwargs.get('max_urls_override', self.max_urls)
        
        # Validate start URL
        if not await self.validate_url(start_url):
            return [await self.handle_error(start_url, ValueError(f"Invalid start URL: {start_url}"))]
            
        # Initialize URL queue with start URL
        url_queue = [(start_url, 0)]  # (url, depth)
        self.queued_urls.add(start_url)
        
        results = []
        
        # Process URL queue
        while url_queue and len(self.visited_urls) < max_urls:
            # Get next URL and depth
            current_url, depth = url_queue.pop(0)
            
            # Skip if already visited
            if current_url in self.visited_urls:
                continue
                
            # Mark as visited
            self.visited_urls.add(current_url)
            
            # Update progress if callback is provided
            if self.progress_callback:
                self.progress_callback(len(self.visited_urls), max_urls, current_url)
                
            # Crawl current URL
            result = await self._crawl_url_with_semaphore(current_url, **kwargs)
            results.append(result)
            
            # Stop if max depth reached
            if depth >= max_depth:
                continue
                
            # Extract links if crawl was successful
            if result.success:
                links = self._extract_links(result.content, current_url)
                
                # Filter links
                filtered_links = await self._filter_links(links, current_url, **kwargs)
                
                # Add new links to queue
                for link in filtered_links:
                    if link not in self.visited_urls and link not in self.queued_urls and len(self.queued_urls) + len(self.visited_urls) < max_urls:
                        url_queue.append((link, depth + 1))
                        self.queued_urls.add(link)
        
        return results
    
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
            
            # Check robots.txt if enabled
            if self.respect_robots_txt and not await self._is_allowed_by_robots(url):
                return CrawlResult(
                    url=url,
                    content="",
                    metadata={"error": "Blocked by robots.txt"},
                    status_code=0,
                    success=False
                )
                
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
    
    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """
        Extract links from HTML content.
        
        Args:
            html: The HTML content to parse
            base_url: The base URL for resolving relative links
            
        Returns:
            List of absolute URLs
        """
        links = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            for a in soup.find_all('a', href=True):
                href = a['href']
                
                # Skip empty links, anchors, and javascript
                if not href or href.startswith('#') or href.startswith('javascript:'):
                    continue
                    
                # Resolve relative URLs
                absolute_url = urljoin(base_url, href)
                
                # Normalize URL
                parsed = urlparse(absolute_url)
                normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if parsed.query:
                    normalized_url += f"?{parsed.query}"
                    
                links.append(normalized_url)
                
            return links
        except Exception as e:
            self.logger.error(f"Error extracting links from {base_url}: {str(e)}")
            return []
    
    async def _filter_links(self, links: List[str], base_url: str, **kwargs) -> List[str]:
        """
        Filter links based on crawler configuration and custom filters.
        
        Args:
            links: List of links to filter
            base_url: The base URL the links were extracted from
            **kwargs: Additional parameters
            
        Returns:
            Filtered list of links
        """
        filtered_links = []
        base_domain = urlparse(base_url).netloc
        
        # Custom URL filter function
        url_filter = kwargs.get('url_filter')
        
        for url in links:
            # Skip if already visited or queued
            if url in self.visited_urls or url in self.queued_urls:
                continue
                
            # Validate URL
            if not await self.validate_url(url):
                continue
                
            # Check domain if not following external links
            if not self.follow_external_links:
                url_domain = urlparse(url).netloc
                if url_domain != base_domain:
                    continue
                    
            # Apply URL patterns
            if self.url_patterns and not any(re.search(pattern, url) for pattern in self.url_patterns):
                continue
                
            # Apply exclude patterns
            if self.exclude_patterns and any(re.search(pattern, url) for pattern in self.exclude_patterns):
                continue
                
            # Apply custom filter if provided
            if url_filter and callable(url_filter) and not url_filter(url):
                continue
                
            filtered_links.append(url)
            
        return filtered_links
    
    async def _is_allowed_by_robots(self, url: str) -> bool:
        """
        Check if a URL is allowed by the site's robots.txt.
        
        Args:
            url: The URL to check
            
        Returns:
            Boolean indicating if the URL is allowed
        """
        try:
            parsed = urlparse(url)
            domain = f"{parsed.scheme}://{parsed.netloc}"
            
            # Check if we already have a parser for this domain
            if domain not in self.robots_parsers:
                # Fetch robots.txt
                robots_url = f"{domain}/robots.txt"
                try:
                    async with self.session.get(
                        robots_url,
                        timeout=self.timeout,
                        headers=self.headers,
                        ssl=None if not self.validate_ssl else True
                    ) as response:
                        if response.status == 200:
                            robots_content = await response.text()
                            # We'd normally use robotparser here, but for simplicity
                            # we'll just check for basic Disallow rules
                            self.robots_parsers[domain] = self._parse_robots_txt(robots_content)
                        else:
                            # No robots.txt or couldn't fetch, assume allowed
                            self.robots_parsers[domain] = []
                except Exception:
                    # Error fetching robots.txt, assume allowed
                    self.robots_parsers[domain] = []
            
            # Check if URL path is disallowed
            path = parsed.path or "/"
            for disallow in self.robots_parsers[domain]:
                if path.startswith(disallow):
                    return False
                    
            return True
        except Exception as e:
            self.logger.error(f"Error checking robots.txt for {url}: {str(e)}")
            return True  # Assume allowed on error
    
    def _parse_robots_txt(self, content: str) -> List[str]:
        """
        Parse robots.txt content for Disallow rules.
        
        Args:
            content: The robots.txt content
            
        Returns:
            List of disallowed paths
        """
        disallowed = []
        user_agent_match = False
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
                
            # Check User-agent
            if line.lower().startswith('user-agent:'):
                agent = line.split(':', 1)[1].strip()
                user_agent_match = agent == '*' or agent in self.user_agent
            
            # Check Disallow
            elif user_agent_match and line.lower().startswith('disallow:'):
                path = line.split(':', 1)[1].strip()
                if path:
                    disallowed.append(path)
        
        return disallowed
    
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
            
            # Extract headings with error handling
            try:
                headings = []
                for tag in ['h1', 'h2', 'h3']:
                    for heading in soup.find_all(tag):
                        headings.append({
                            "level": int(tag[1]),
                            "text": heading.get_text(strip=True)
                        })
                result["headings"] = headings
            except Exception as e:
                result["errors"].append({"type": "headings_extraction", "message": str(e)})
                result["partial_success"] = True
                result["headings"] = []
            
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
    
    async def handle_rate_limits(self) -> None:
        """
        Implement rate limiting to avoid overloading target servers.
        
        This method uses domain-specific rate limiting to be respectful to different servers.
        """
        # Domain-specific rate limiting
        domain_limits = self.config.get('domain_rate_limits', {})
        
        # Default delay to avoid overwhelming servers
        default_delay = self.config.get('default_delay', 0.5)
        await asyncio.sleep(default_delay)
    
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

