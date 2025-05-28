"""Academic website crawler implementation."""

import time
import requests
from typing import Optional, Dict, Any, List, Set
from bs4 import BeautifulSoup
import re
import logging
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)

class AcademicCrawler:
    """Crawler for academic websites and repositories."""
    
    def __init__(self, base_url: str, timeout: int = 30, max_retries: int = 3):
        """Initialize the academic crawler.
        
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
        self.min_request_interval = 2.0  # seconds - more conservative for academic sites
        self.visited_urls = set()
        self.paper_urls = set()
    
    def _get_session(self) -> requests.Session:
        """Create and configure a requests session.
        
        Returns:
            Configured requests session
        """
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'CitadelRevisions/1.0 (Academic Research)',
            'Accept': 'text/html,application/xhtml+xml,application/xml,application/pdf',
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
                
            # Check if URL belongs to the base domain or is an allowed external domain
            base_domain = urlparse(self.base_url).netloc
            allowed_domains = [base_domain, 'doi.org', 'arxiv.org', 'scholar.google.com']
            
            if not any(parsed_url.netloc.endswith(domain) for domain in allowed_domains):
                return False
                
            # Additional check for PDF files
            if parsed_url.path.lower().endswith('.pdf'):
                self.paper_urls.add(url)
                
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
    
    def _extract_paper_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from academic paper pages.
        
        Args:
            soup: BeautifulSoup object of the paper page
            
        Returns:
            Dictionary with paper metadata
        """
        metadata = {}
        
        # Try to extract title
        title_elem = soup.find('meta', {'name': 'citation_title'}) or soup.find('h1')
        if title_elem:
            metadata['title'] = title_elem.get('content', '') if title_elem.name == 'meta' else title_elem.text.strip()
        
        # Try to extract authors
        author_elems = soup.find_all('meta', {'name': 'citation_author'})
        if author_elems:
            metadata['authors'] = [elem.get('content', '').strip() for elem in author_elems]
        
        # Try to extract publication date
        date_elem = soup.find('meta', {'name': 'citation_publication_date'})
        if date_elem:
            metadata['publication_date'] = date_elem.get('content', '').strip()
        
        return metadata
    
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
        """Crawl academic website and extract paper information.
        
        Args:
            max_pages: Maximum number of pages to crawl
            
        Returns:
            List of paper metadata dictionaries
        """
        papers = []
        to_visit = [self.base_url]
        visited_count = 0
        
        while to_visit and visited_count < max_pages:
            url = to_visit.pop(0)
            
            response = self._safe_request(url)
            if not response:
                continue
                
            visited_count += 1
            
            if response.headers.get('Content-Type', '').startswith('application/pdf'):
                # Handle PDF directly
                papers.append({'url': url, 'type': 'pdf'})
                continue
                
            soup = self._parse_html(response.text)
            
            # Extract paper metadata if this looks like a paper page
            if soup.find('meta', {'name': 'citation_title'}):
                metadata = self._extract_paper_metadata(soup)
                metadata['url'] = url
                papers.append(metadata)
            
            # Find more links to follow
            for link in soup.find_all('a', href=True):
                next_url = urljoin(url, link['href'])
                if next_url not in self.visited_urls and next_url not in to_visit:
                    to_visit.append(next_url)
        
        return papers
