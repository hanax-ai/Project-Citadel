"""Improved crawler implementation with enhanced error handling.

This module demonstrates how to implement robust error handling in a web crawler
using the utilities from crawler_utils.py. It focuses on the extract_data method
to show how to handle different types of errors (parsing errors, network errors, etc.)
in a structured way.
"""

import time
import requests
from typing import Optional, Dict, Any, List, Set, Union, Tuple, TypeVar, cast
from bs4 import BeautifulSoup, Tag
import re
import logging
import json
from urllib.parse import urlparse, urljoin
from http import HTTPStatus
from datetime import datetime
import traceback

from citadel_revisions import crawler_utils

# Type variables for better type hinting
T = TypeVar('T')
DataResult = Dict[str, Any]
ErrorResult = Dict[str, Any]
ExtractResult = Union[DataResult, ErrorResult]

logger = logging.getLogger(__name__)

class ImprovedCrawler:
    """Crawler with improved error handling capabilities.
    
    This crawler implementation demonstrates best practices for error handling
    in web crawlers, particularly in the data extraction phase. It uses the
    utilities from crawler_utils.py to handle different types of errors in a
    structured and consistent way.
    """
    
    def __init__(self, base_url: str, timeout: int = 30, max_retries: int = 3):
        """Initialize the improved crawler.
        
        Args:
            base_url: The base URL to crawl
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = crawler_utils.get_session(
            user_agent="ImprovedCrawler/1.0",
            additional_headers={"X-Crawler": "ImprovedCrawler"}
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
            url_patterns=['/content/', '/article/', '/data/'],
            excluded_patterns=['/login', '/signup', '/admin'],
            callback=lambda u: self.content_urls.add(u) if '/content/' in u or '/article/' in u else None
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
    
    def _is_error_result(self, result: ExtractResult) -> bool:
        """Check if a result is an error result.
        
        Args:
            result: The result to check
            
        Returns:
            True if the result is an error result, False otherwise
        """
        return 'error' in result and 'error_type' in result
    
    def _create_error_result(self, error_type: str, error_message: str, 
                            url: str, additional_info: Optional[Dict[str, Any]] = None) -> ErrorResult:
        """Create a standardized error result.
        
        Args:
            error_type: Type of error (e.g., 'parsing_error', 'network_error')
            error_message: Human-readable error message
            url: URL that was being processed when the error occurred
            additional_info: Additional information about the error
            
        Returns:
            Standardized error result dictionary
        """
        result: ErrorResult = {
            'error': True,
            'error_type': error_type,
            'error_message': error_message,
            'url': url,
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_info:
            result['additional_info'] = additional_info
            
        return result
    
    def _handle_parsing_error(self, url: str, element_name: str, 
                             exception: Optional[Exception] = None) -> ErrorResult:
        """Handle parsing errors in a standardized way.
        
        Args:
            url: URL that was being processed
            element_name: Name of the element that failed to parse
            exception: The exception that was raised, if any
            
        Returns:
            Standardized error result
        """
        error_message = f"Failed to parse {element_name} from {url}"
        additional_info = None
        
        if exception:
            error_message += f": {str(exception)}"
            additional_info = {
                'exception_type': type(exception).__name__,
                'traceback': traceback.format_exc()
            }
            
        logger.warning(error_message)
        return self._create_error_result('parsing_error', error_message, url, additional_info)
    
    def _handle_network_error(self, url: str, status_code: Optional[int] = None,
                             exception: Optional[Exception] = None) -> ErrorResult:
        """Handle network errors in a standardized way.
        
        Args:
            url: URL that was being processed
            status_code: HTTP status code, if available
            exception: The exception that was raised, if any
            
        Returns:
            Standardized error result
        """
        error_message = f"Network error while accessing {url}"
        additional_info = {}
        
        if status_code:
            error_message += f" (Status code: {status_code})"
            additional_info['status_code'] = status_code
            
            # Add standard HTTP status message if available
            try:
                status_enum = HTTPStatus(status_code)
                additional_info['status_message'] = status_enum.phrase
            except ValueError:
                pass
        
        if exception:
            error_message += f": {str(exception)}"
            additional_info['exception_type'] = type(exception).__name__
            additional_info['traceback'] = traceback.format_exc()
            
        logger.warning(error_message)
        return self._create_error_result('network_error', error_message, url, additional_info)
    
    def _handle_timeout_error(self, url: str, timeout_value: float,
                             exception: Optional[Exception] = None) -> ErrorResult:
        """Handle timeout errors in a standardized way.
        
        Args:
            url: URL that was being processed
            timeout_value: Timeout value in seconds
            exception: The exception that was raised, if any
            
        Returns:
            Standardized error result
        """
        error_message = f"Timeout error while accessing {url} (Timeout: {timeout_value}s)"
        additional_info = {'timeout_value': timeout_value}
        
        if exception:
            error_message += f": {str(exception)}"
            additional_info['exception_type'] = type(exception).__name__
            additional_info['traceback'] = traceback.format_exc()
            
        logger.warning(error_message)
        return self._create_error_result('timeout_error', error_message, url, additional_info)
    
    def _log_and_raise(self, error_type: str, error_message: str, 
                      exception: Optional[Exception] = None) -> None:
        """Log an error and raise an exception.
        
        This method is used for critical errors that should stop the crawler.
        
        Args:
            error_type: Type of error
            error_message: Human-readable error message
            exception: The exception to raise, if any
            
        Raises:
            Exception: The provided exception or a RuntimeError with the error message
        """
        logger.error(f"{error_type}: {error_message}")
        
        if exception:
            raise exception
        else:
            raise RuntimeError(error_message)
    
    def extract_data(self, url: str, html_content: Optional[str] = None) -> ExtractResult:
        """Extract data from a webpage with comprehensive error handling.
        
        This method demonstrates best practices for error handling in web crawlers.
        It handles different types of errors (parsing errors, network errors, etc.)
        in a structured way and returns either the extracted data or a standardized
        error result.
        
        Args:
            url: URL of the webpage to extract data from
            html_content: HTML content of the webpage, if already fetched.
                          If None, the method will fetch the content.
                          
        Returns:
            A dictionary containing either:
            - The extracted data (title, content, metadata, etc.)
            - A standardized error result with error type, message, and additional info
        """
        # Initialize the result with the URL
        result: Dict[str, Any] = {'url': url}
        
        try:
            # Fetch the HTML content if not provided
            if html_content is None:
                try:
                    response = self._safe_request(url)
                    if not response:
                        return self._handle_network_error(url, None, 
                                                         Exception("Failed to fetch URL"))
                    
                    # Check for HTTP errors
                    error_message = crawler_utils.handle_http_error(response)
                    if error_message:
                        return self._handle_network_error(url, response.status_code, 
                                                         Exception(error_message))
                    
                    html_content = response.text
                except requests.Timeout as e:
                    return self._handle_timeout_error(url, self.timeout, e)
                except requests.RequestException as e:
                    return self._handle_network_error(url, getattr(e.response, 'status_code', None), e)
                except Exception as e:
                    return self._handle_network_error(url, None, e)
            
            # Parse the HTML content
            try:
                soup = crawler_utils.parse_html(html_content)
            except Exception as e:
                return self._handle_parsing_error(url, "HTML content", e)
            
            # Extract title with error handling
            try:
                title_elem = soup.find('h1') or soup.find('title')
                if title_elem:
                    result['title'] = title_elem.text.strip()
                else:
                    logger.debug(f"No title found for {url}")
                    result['title'] = "No title found"
            except Exception as e:
                # Non-critical error, log and continue
                logger.warning(f"Error extracting title from {url}: {str(e)}")
                result['title'] = "Error extracting title"
            
            # Extract main content with error handling
            try:
                main_elem = (soup.find('main') or 
                            soup.find('article') or 
                            soup.find('div', {'class': 'content'}) or
                            soup.find('div', {'id': 'content'}))
                
                if main_elem:
                    # Remove script and style elements
                    for script in main_elem(['script', 'style']):
                        script.decompose()
                    
                    result['content'] = main_elem.get_text(separator=' ', strip=True)
                else:
                    logger.debug(f"No main content found for {url}")
                    result['content'] = "No content found"
            except Exception as e:
                # This is more critical, but we'll still try to return partial results
                parsing_error = self._handle_parsing_error(url, "main content", e)
                result['content_error'] = parsing_error['error_message']
                result['content'] = "Error extracting content"
            
            # Extract metadata with error handling
            try:
                metadata = self._extract_metadata(soup, url)
                if metadata:
                    result['metadata'] = metadata
            except Exception as e:
                # Non-critical error, log and continue
                logger.warning(f"Error extracting metadata from {url}: {str(e)}")
                result['metadata'] = {"error": str(e)}
            
            # Extract links with error handling
            try:
                links = crawler_utils.extract_links(soup, url)
                result['links'] = links
            except Exception as e:
                # Non-critical error, log and continue
                logger.warning(f"Error extracting links from {url}: {str(e)}")
                result['links'] = []
            
            # Add timestamp
            result['extracted_at'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            # Catch-all for any unexpected errors
            error_message = f"Unexpected error extracting data from {url}: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            
            return self._create_error_result(
                'unexpected_error',
                error_message,
                url,
                {
                    'exception_type': type(e).__name__,
                    'traceback': traceback.format_exc()
                }
            )
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from a webpage.
        
        Args:
            soup: BeautifulSoup object
            url: URL of the webpage
            
        Returns:
            Dictionary with metadata
        """
        metadata = {}
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            
            if name and content:
                # Clean up the name (remove og: prefix, etc.)
                clean_name = name.lower().replace('og:', '').replace('twitter:', '')
                metadata[clean_name] = content
        
        # Extract publication date
        try:
            date_patterns = [
                ('meta', {'property': 'article:published_time'}),
                ('meta', {'name': 'publication_date'}),
                ('time', {}),
                ('span', {'class': re.compile(r'date|time|publish', re.I)})
            ]
            
            for tag, attrs in date_patterns:
                date_elem = soup.find(tag, attrs)
                if date_elem:
                    if tag == 'meta':
                        metadata['published_date'] = date_elem.get('content', '')
                    else:
                        metadata['published_date'] = date_elem.text.strip()
                    break
        except Exception as e:
            logger.debug(f"Error extracting publication date: {str(e)}")
        
        # Extract author
        try:
            author_patterns = [
                ('meta', {'property': 'article:author'}),
                ('meta', {'name': 'author'}),
                ('a', {'class': re.compile(r'author', re.I)}),
                ('span', {'class': re.compile(r'author', re.I)})
            ]
            
            for tag, attrs in author_patterns:
                author_elem = soup.find(tag, attrs)
                if author_elem:
                    if tag == 'meta':
                        metadata['author'] = author_elem.get('content', '')
                    else:
                        metadata['author'] = author_elem.text.strip()
                    break
        except Exception as e:
            logger.debug(f"Error extracting author: {str(e)}")
        
        return metadata
    
    @crawler_utils.retry_on_error(max_retries=2)
    def crawl(self, max_pages: int = 50) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Crawl the website and extract content with error handling.
        
        Args:
            max_pages: Maximum number of pages to crawl
            
        Returns:
            Tuple containing:
            - List of successfully extracted content dictionaries
            - List of error results
        """
        successful_results = []
        error_results = []
        to_visit = [self.base_url]
        visited_count = 0
        
        while to_visit and visited_count < max_pages:
            url = to_visit.pop(0)
            
            response = self._safe_request(url)
            if not response:
                error_results.append(
                    self._handle_network_error(url, None, Exception("Failed to fetch URL"))
                )
                continue
                
            visited_count += 1
            
            # Extract data with error handling
            result = self.extract_data(url, response.text)
            
            # Check if the result is an error result
            if self._is_error_result(result):
                error_results.append(cast(ErrorResult, result))
            else:
                successful_results.append(cast(DataResult, result))
            
            # Find more links to follow
            if 'links' in result and not self._is_error_result(result):
                for link in result['links']:
                    if link not in self.visited_urls and link not in to_visit:
                        to_visit.append(link)
        
        return successful_results, error_results


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    crawler = ImprovedCrawler("https://example.com")
    successful_results, error_results = crawler.crawl(max_pages=10)
    
    print(f"Successfully crawled {len(successful_results)} pages")
    print(f"Encountered errors on {len(error_results)} pages")
    
    # Example of how to handle the results
    if successful_results:
        print("\nExample of successful result:")
        example = successful_results[0]
        print(f"Title: {example.get('title', 'No title')}")
        print(f"URL: {example['url']}")
        
    if error_results:
        print("\nExample of error result:")
        example = error_results[0]
        print(f"Error type: {example['error_type']}")
        print(f"Error message: {example['error_message']}")
        print(f"URL: {example['url']}")
