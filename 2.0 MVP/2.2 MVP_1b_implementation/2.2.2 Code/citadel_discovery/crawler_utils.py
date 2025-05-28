"""Utility functions for web crawlers.

This module provides common utility functions for web crawlers to avoid code duplication
across different crawler implementations. It includes URL validation, HTML parsing,
rate limiting, error handling, and HTTP session management.
"""

import time
import requests
from typing import Optional, Dict, Any, List, Union, Set, Callable, TypeVar
from bs4 import BeautifulSoup
import re
import logging
import functools
from urllib.parse import urlparse, urljoin
from http import HTTPStatus

logger = logging.getLogger(__name__)

# Type variables for better type hinting
T = TypeVar('T')
R = TypeVar('R')


def validate_url(url: str, base_url: str, allowed_domains: Optional[List[str]] = None,
                 allowed_schemes: Optional[List[str]] = None,
                 url_patterns: Optional[List[str]] = None,
                 excluded_patterns: Optional[List[str]] = None,
                 callback: Optional[Callable[[str], None]] = None) -> bool:
    """Validate if a URL is valid and belongs to the allowed domains.
    
    This function checks if a URL is valid and belongs to the allowed domains.
    It also checks if the URL matches any of the allowed patterns and doesn't match
    any of the excluded patterns.
    
    Args:
        url: URL to validate
        base_url: Base URL for the crawler
        allowed_domains: List of allowed domains. If None, only the base domain is allowed
        allowed_schemes: List of allowed URL schemes. Defaults to ['http', 'https']
        url_patterns: List of URL patterns to include. URLs matching any of these patterns
                     will be considered valid
        excluded_patterns: List of URL patterns to exclude. URLs matching any of these patterns
                          will be considered invalid
        callback: Optional callback function to call with the URL if it's valid
        
    Returns:
        True if URL is valid, False otherwise
    """
    if not url:
        return False
        
    # Set defaults
    if allowed_schemes is None:
        allowed_schemes = ['http', 'https']
        
    # Parse the URL
    try:
        parsed_url = urlparse(url)
        
        # Check if scheme and netloc are present
        if not parsed_url.scheme or not parsed_url.netloc:
            return False
            
        # Check if URL uses allowed schemes
        if parsed_url.scheme not in allowed_schemes:
            return False
            
        # Check if URL belongs to the allowed domains
        base_domain = urlparse(base_url).netloc
        
        # If no additional allowed domains are specified, only allow the base domain
        domains_to_check = [base_domain]
        if allowed_domains:
            domains_to_check.extend(allowed_domains)
            
        domain_valid = False
        for domain in domains_to_check:
            if parsed_url.netloc == domain or parsed_url.netloc.endswith('.' + domain):
                domain_valid = True
                break
                
        if not domain_valid:
            return False
            
        # Check excluded patterns
        if excluded_patterns:
            if any(pattern in url for pattern in excluded_patterns):
                return False
                
        # Check URL patterns
        if url_patterns:
            # If URL patterns are specified, at least one must match
            # Exception: the base URL itself is always allowed
            if url == base_url:
                pass  # Allow the base URL
            elif not any(pattern in url for pattern in url_patterns):
                return False
                
        # If a callback is provided, call it with the URL
        if callback:
            callback(url)
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating URL {url}: {str(e)}")
        return False


def parse_html(html_content: str, parser: str = 'html.parser') -> BeautifulSoup:
    """Parse HTML content using BeautifulSoup.
    
    Args:
        html_content: Raw HTML content
        parser: HTML parser to use. Default is 'html.parser', but 'lxml' might be faster
                if installed
        
    Returns:
        BeautifulSoup object
    """
    return BeautifulSoup(html_content, parser)


def rate_limiter(min_interval: float = 1.0) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to implement rate limiting for functions.
    
    This decorator ensures that the decorated function is not called more frequently
    than the specified minimum interval.
    
    Args:
        min_interval: Minimum interval between function calls in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        last_call_time = 0.0
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            nonlocal last_call_time
            current_time = time.time()
            elapsed = current_time - last_call_time
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)
                
            last_call_time = time.time()
            return func(*args, **kwargs)
            
        return wrapper
        
    return decorator


def apply_rate_limiting(last_request_time: float, min_request_interval: float) -> float:
    """Apply rate limiting based on the last request time.
    
    Args:
        last_request_time: Timestamp of the last request
        min_request_interval: Minimum interval between requests in seconds
        
    Returns:
        Current timestamp after applying rate limiting
    """
    current_time = time.time()
    elapsed = current_time - last_request_time
    
    if elapsed < min_request_interval:
        sleep_time = min_request_interval - elapsed
        time.sleep(sleep_time)
        
    return time.time()


def get_session(user_agent: str = 'CitadelRevisions/1.0',
                accept: str = 'text/html,application/xhtml+xml,application/xml',
                accept_language: str = 'en-US,en;q=0.9',
                additional_headers: Optional[Dict[str, str]] = None) -> requests.Session:
    """Create and configure a requests session.
    
    Args:
        user_agent: User agent string
        accept: Accept header
        accept_language: Accept-Language header
        additional_headers: Additional headers to add to the session
        
    Returns:
        Configured requests session
    """
    session = requests.Session()
    
    headers = {
        'User-Agent': user_agent,
        'Accept': accept,
        'Accept-Language': accept_language,
    }
    
    if additional_headers:
        headers.update(additional_headers)
        
    session.headers.update(headers)
    return session


def safe_request(url: str, session: requests.Session, 
                 validate_func: Callable[[str], bool],
                 rate_limit_func: Optional[Callable[[], None]] = None,
                 visited_urls: Optional[Set[str]] = None,
                 timeout: int = 30, 
                 max_retries: int = 3,
                 method: str = 'GET',
                 backoff_factor: int = 2,
                 **kwargs) -> Optional[requests.Response]:
    """Make a safe HTTP request with error handling and retries.
    
    Args:
        url: URL to request
        session: Requests session to use
        validate_func: Function to validate the URL
        rate_limit_func: Optional function to apply rate limiting
        visited_urls: Optional set of visited URLs to avoid duplicates
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries for failed requests
        method: HTTP method (GET, POST, etc.)
        backoff_factor: Factor to use for exponential backoff between retries
        **kwargs: Additional arguments to pass to requests
        
    Returns:
        Response object or None if request failed
    """
    if not validate_func(url):
        logger.warning(f"Invalid URL: {url}")
        return None
        
    if visited_urls is not None and url in visited_urls:
        logger.debug(f"Skipping already visited URL: {url}")
        return None
        
    if rate_limit_func:
        rate_limit_func()
        
    for attempt in range(max_retries + 1):
        try:
            response = session.request(
                method=method,
                url=url,
                timeout=timeout,
                **kwargs
            )
            response.raise_for_status()
            
            if visited_urls is not None:
                visited_urls.add(url)
                
            return response
            
        except requests.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt+1}/{max_retries+1}): {url}, Error: {str(e)}")
            
            if attempt < max_retries:
                # Exponential backoff
                wait_time = backoff_factor ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Max retries exceeded for URL: {url}")
                return None


def extract_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Extract all links from a BeautifulSoup object.
    
    Args:
        soup: BeautifulSoup object
        base_url: Base URL to resolve relative links
        
    Returns:
        List of absolute URLs
    """
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        absolute_url = urljoin(base_url, href)
        links.append(absolute_url)
    return links


def is_success_status(status_code: int) -> bool:
    """Check if an HTTP status code indicates success.
    
    Args:
        status_code: HTTP status code
        
    Returns:
        True if status code indicates success, False otherwise
    """
    return 200 <= status_code < 300


def handle_http_error(response: requests.Response) -> Optional[str]:
    """Handle HTTP errors and return an appropriate error message.
    
    Args:
        response: HTTP response
        
    Returns:
        Error message or None if no error
    """
    if is_success_status(response.status_code):
        return None
        
    status_messages = {
        HTTPStatus.BAD_REQUEST: "Bad request",
        HTTPStatus.UNAUTHORIZED: "Unauthorized",
        HTTPStatus.FORBIDDEN: "Forbidden",
        HTTPStatus.NOT_FOUND: "Not found",
        HTTPStatus.TOO_MANY_REQUESTS: "Too many requests",
        HTTPStatus.INTERNAL_SERVER_ERROR: "Internal server error",
        HTTPStatus.SERVICE_UNAVAILABLE: "Service unavailable",
        HTTPStatus.GATEWAY_TIMEOUT: "Gateway timeout",
    }
    
    message = status_messages.get(response.status_code, f"HTTP error {response.status_code}")
    return f"{message}: {response.url}"


def retry_on_error(max_retries: int = 3, 
                  retry_codes: Optional[List[int]] = None,
                  backoff_factor: int = 2) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function on error.
    
    Args:
        max_retries: Maximum number of retries
        retry_codes: List of HTTP status codes to retry on. If None, retry on all errors
        backoff_factor: Factor to use for exponential backoff between retries
        
    Returns:
        Decorated function
    """
    if retry_codes is None:
        retry_codes = [429, 500, 502, 503, 504]  # Common retry codes
        
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Check if result is a Response object and has a status code
                    if isinstance(result, requests.Response):
                        if result.status_code in retry_codes:
                            if attempt < max_retries:
                                wait_time = backoff_factor ** attempt
                                logger.info(f"Retrying due to status code {result.status_code} in {wait_time} seconds...")
                                time.sleep(wait_time)
                                continue
                                
                    return result
                    
                except Exception as e:
                    if attempt < max_retries:
                        wait_time = backoff_factor ** attempt
                        logger.warning(f"Error: {str(e)}. Retrying in {wait_time} seconds... (attempt {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Max retries exceeded. Last error: {str(e)}")
                        raise
                        
            # This should never be reached due to the raise in the except block
            return func(*args, **kwargs)
            
        return wrapper
        
    return decorator
