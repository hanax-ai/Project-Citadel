"""News website crawler implementation."""

import time
import requests
from typing import Optional, Dict, Any, List, Set
from bs4 import BeautifulSoup
import re
import logging
from urllib.parse import urlparse, urljoin
from datetime import datetime

logger = logging.getLogger(__name__)

class NewsCrawler:
    """Crawler for news websites and articles."""
    
    def __init__(self, base_url: str, timeout: int = 20, max_retries: int = 3):
        """Initialize the news crawler.
        
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
        self.min_request_interval = 1.5  # seconds
        self.visited_urls = set()
        self.article_urls = set()
    
    def _get_session(self) -> requests.Session:
        """Create and configure a requests session.
        
        Returns:
            Configured requests session
        """
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
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
                
            # Skip URLs with query parameters (often pagination or search)
            if parsed_url.query:
                return False
                
            # Check for article patterns in URL
            article_patterns = ['/article/', '/news/', '/story/', '/post/']
            if any(pattern in parsed_url.path for pattern in article_patterns):
                self.article_urls.add(url)
                
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
    
    def _extract_article_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract content from news article pages.
        
        Args:
            soup: BeautifulSoup object of the article page
            
        Returns:
            Dictionary with article content
        """
        article = {}
        
        # Try to extract title
        title_elem = soup.find('h1') or soup.find('meta', {'property': 'og:title'})
        if title_elem:
            article['title'] = title_elem.text.strip() if title_elem.name == 'h1' else title_elem.get('content', '')
        
        # Try to extract publication date
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
                    article['published_date'] = date_elem.get('content', '')
                else:
                    article['published_date'] = date_elem.text.strip()
                break
        
        # Try to extract author
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
                    article['author'] = author_elem.get('content', '')
                else:
                    article['author'] = author_elem.text.strip()
                break
        
        # Try to extract main content
        content_patterns = [
            ('article', {}),
            ('div', {'class': re.compile(r'article|content|story', re.I)}),
            ('div', {'id': re.compile(r'article|content|story', re.I)})
        ]
        
        for tag, attrs in content_patterns:
            content_elem = soup.find(tag, attrs)
            if content_elem:
                # Remove script and style elements
                for script in content_elem(['script', 'style']):
                    script.decompose()
                
                article['content'] = content_elem.get_text(separator=' ', strip=True)
                break
        
        return article
    
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
            
        if url in self.visited_urls:
            logger.debug(f"Skipping already visited URL: {url}")
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
                self.visited_urls.add(url)
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
    
    def crawl(self, max_pages: int = 100) -> List[Dict[str, Any]]:
        """Crawl news website and extract article information.
        
        Args:
            max_pages: Maximum number of pages to crawl
            
        Returns:
            List of article dictionaries
        """
        articles = []
        to_visit = [self.base_url]
        visited_count = 0
        
        while to_visit and visited_count < max_pages:
            url = to_visit.pop(0)
            
            response = self._safe_request(url)
            if not response:
                continue
                
            visited_count += 1
            
            soup = self._parse_html(response.text)
            
            # Extract article content if this is an article page
            if url in self.article_urls:
                article = self._extract_article_content(soup)
                article['url'] = url
                article['crawled_at'] = datetime.now().isoformat()
                articles.append(article)
            
            # Find more links to follow
            for link in soup.find_all('a', href=True):
                next_url = urljoin(url, link['href'])
                if next_url not in self.visited_urls and next_url not in to_visit:
                    to_visit.append(next_url)
        
        return articles
