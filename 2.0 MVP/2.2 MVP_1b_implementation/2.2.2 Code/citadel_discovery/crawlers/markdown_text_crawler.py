
"""
Markdown/Text Crawler Implementation for Project Citadel.

This crawler is designed to process Markdown and text files from local or remote sources.
"""
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Union
import aiohttp
import aiofiles
from urllib.parse import urlparse
import re
import markdown
from bs4 import BeautifulSoup

from citadel.crawlers.base_crawler import BaseCrawler, CrawlResult
from citadel.utils.crawler_utils import validate_url


class MarkdownTextCrawler(BaseCrawler):
    """
    Crawler implementation for processing Markdown and text files.
    
    This crawler can handle both local files and remote URLs, extracting structured
    data from Markdown and plain text content with support for various formats.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MarkdownTextCrawler with configuration.
        
        Args:
            config: Dictionary containing crawler configuration parameters including:
                - user_agent: Custom user agent string
                - timeout: Request timeout in seconds
                - max_retries: Maximum number of retry attempts
                - retry_delay: Delay between retries in seconds
                - headers: Additional HTTP headers
                - extraction_patterns: Dictionary of regex patterns for text extraction
                - validate_ssl: Whether to validate SSL certificates
                - markdown_extensions: List of markdown extensions to use
                - max_file_size: Maximum file size to process in bytes
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.extraction_patterns = config.get('extraction_patterns', {})
        self.validate_ssl = config.get('validate_ssl', True)
        self.markdown_extensions = config.get('markdown_extensions', ['tables', 'fenced_code'])
        self.max_file_size = config.get('max_file_size', 10 * 1024 * 1024)  # Default: 10MB
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
    
    async def crawl(self, source: Union[str, List[str]], **kwargs) -> Union[CrawlResult, List[CrawlResult]]:
        """
        Process Markdown or text content from a file path or URL.
        
        Args:
            source: File path, URL, or list of paths/URLs to process
            **kwargs: Additional parameters:
                - is_local: Boolean indicating if source is a local file path
                - custom_headers: Dict of headers to merge with default headers
                - extraction_override: Dict of extraction patterns to override defaults
                
        Returns:
            CrawlResult or List[CrawlResult] containing the processed data
        """
        # Handle list of sources
        if isinstance(source, list):
            results = []
            for single_source in source:
                result = await self._process_single_source(single_source, **kwargs)
                results.append(result)
            return results
            
        # Handle single source
        return await self._process_single_source(source, **kwargs)
    
    async def _process_single_source(self, source: str, **kwargs) -> CrawlResult:
        """
        Process a single Markdown or text source.
        
        Args:
            source: File path or URL to process
            **kwargs: Additional parameters
            
        Returns:
            CrawlResult containing the processed data
        """
        is_local = kwargs.get('is_local', self._is_local_path(source))
        
        if not await self.validate_url(source) and not is_local:
            return await self.handle_error(source, ValueError(f"Invalid source: {source}"))
            
        try:
            # Process local file
            if is_local:
                content = await self._read_local_file(source)
            # Process remote URL
            else:
                # Create session if not exists
                if self.session is None:
                    self.session = aiohttp.ClientSession(headers=self.headers)
                    
                # Apply rate limiting for remote requests
                await self.handle_rate_limits()
                
                # Track request time for rate limiting
                self.last_request_time = asyncio.get_event_loop().time()
                
                content = await self._fetch_remote_content(source, **kwargs)
                
            # Determine content type
            content_type = self._detect_content_type(source, content)
            
            # Extract data based on content type
            extraction_override = kwargs.get('extraction_override', {})
            extraction_patterns = {**self.extraction_patterns, **extraction_override}
            
            extracted_data = await self.extract_data(content, source, extraction_patterns, content_type)
            
            return CrawlResult(
                url=source,
                content=content,
                metadata=extracted_data,
                status_code=200,
                success=True
            )
        except Exception as e:
            return await self.handle_error(source, e)
    
    async def _read_local_file(self, file_path: str) -> str:
        """
        Read content from a local file.
        
        Args:
            file_path: Path to the local file
            
        Returns:
            String containing the file content
        """
        # Validate file size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            raise ValueError(f"File size ({file_size} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)")
            
        # Read file content
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            
        return content
    
    async def _fetch_remote_content(self, url: str, **kwargs) -> str:
        """
        Fetch content from a remote URL.
        
        Args:
            url: URL to fetch content from
            **kwargs: Additional parameters
            
        Returns:
            String containing the fetched content
        """
        custom_headers = kwargs.get('custom_headers', {})
        merged_headers = {**self.headers, **custom_headers}
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(
                    url, 
                    timeout=self.timeout,
                    headers=merged_headers,
                    ssl=None if not self.validate_ssl else True
                ) as response:
                    if response.status >= 400:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                            continue
                        raise ValueError(f"Failed to fetch content: HTTP {response.status}")
                        
                    # Check content length
                    content_length = response.headers.get('Content-Length')
                    if content_length and int(content_length) > self.max_file_size:
                        raise ValueError(f"Content size ({content_length} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)")
                        
                    content = await response.text()
                    return content
            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise asyncio.TimeoutError(f"Request timed out after {self.timeout} seconds")
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise e
    
    def _detect_content_type(self, source: str, content: str) -> str:
        """
        Detect the type of content based on file extension and content.
        
        Args:
            source: File path or URL
            content: The content to analyze
            
        Returns:
            String indicating the detected content type
        """
        # Check file extension
        if source.lower().endswith(('.md', '.markdown')):
            return 'markdown'
        elif source.lower().endswith(('.txt', '.text')):
            return 'text'
        elif source.lower().endswith(('.rst', '.rest')):
            return 'rst'
            
        # Check content for Markdown indicators
        if re.search(r'^#+ ', content, re.MULTILINE) or '```' in content or '*' in content:
            return 'markdown'
            
        # Default to plain text
        return 'text'
    
    async def extract_data(self, content: str, source: str, extraction_patterns: Optional[Dict[str, str]] = None, content_type: str = 'text') -> Dict[str, Any]:
        """
        Extract structured data from the text content.
        
        Args:
            content: The text content to parse
            source: The source (file path or URL)
            extraction_patterns: Optional dictionary of regex patterns
            content_type: Type of content ('markdown', 'text', 'rst')
            
        Returns:
            Dictionary containing extracted data with standardized error handling
        """
        from citadel.utils.crawler_utils import ParsingError, get_soup

        if extraction_patterns is None:
            extraction_patterns = self.extraction_patterns
            
        result = {
            "source": source,
            "content_type": content_type,
            "timestamp": asyncio.get_event_loop().time(),
            "success": True,
            "partial_success": False,
            "errors": []
        }
        
        try:
            # Extract basic metadata with error handling
            try:
                lines = content.split('\n')
                result["line_count"] = len(lines)
                result["word_count"] = len(re.findall(r'\b\w+\b', content))
                result["char_count"] = len(content)
            except Exception as e:
                result["errors"].append({"type": "basic_metadata_extraction", "message": str(e)})
                result["partial_success"] = True
                # Set default values for basic metadata
                result["line_count"] = 0
                result["word_count"] = 0
                result["char_count"] = 0
            
            # Process based on content type
            if content_type == 'markdown':
                try:
                    # Convert Markdown to HTML for structured extraction
                    html = markdown.markdown(content, extensions=self.markdown_extensions)
                    soup = get_soup(html, 'html.parser')
                    
                    # Extract title (first heading) with error handling
                    try:
                        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                        if headings:
                            result["title"] = headings[0].get_text(strip=True)
                    except Exception as e:
                        result["errors"].append({"type": "title_extraction", "message": str(e)})
                        result["partial_success"] = True
                    
                    # Extract all headings with hierarchy with error handling
                    try:
                        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                        result["headings"] = []
                        for heading in headings:
                            level = int(heading.name[1])
                            result["headings"].append({
                                "level": level,
                                "text": heading.get_text(strip=True)
                            })
                    except Exception as e:
                        result["errors"].append({"type": "headings_extraction", "message": str(e)})
                        result["partial_success"] = True
                        result["headings"] = []
                    
                    # Extract code blocks with error handling
                    try:
                        code_blocks = []
                        for pre in soup.find_all('pre'):
                            code = pre.find('code')
                            if code:
                                code_blocks.append(code.get_text())
                            else:
                                code_blocks.append(pre.get_text())
                        result["code_blocks"] = code_blocks
                    except Exception as e:
                        result["errors"].append({"type": "code_blocks_extraction", "message": str(e)})
                        result["partial_success"] = True
                        result["code_blocks"] = []
                    
                    # Extract links with error handling
                    try:
                        links = []
                        for a in soup.find_all('a', href=True):
                            links.append({
                                "text": a.get_text(strip=True),
                                "href": a['href']
                            })
                        result["links"] = links
                    except Exception as e:
                        result["errors"].append({"type": "links_extraction", "message": str(e)})
                        result["partial_success"] = True
                        result["links"] = []
                except Exception as e:
                    result["errors"].append({"type": "markdown_processing", "message": str(e)})
                    result["partial_success"] = True
            else:  # Plain text or RST
                # Try to extract title (first non-empty line) with error handling
                try:
                    for line in lines:
                        if line.strip():
                            result["title"] = line.strip()
                            break
                except Exception as e:
                    result["errors"].append({"type": "title_extraction", "message": str(e)})
                    result["partial_success"] = True
            
            # Apply custom extraction patterns with error handling
            extraction_errors = []
            for key, pattern in extraction_patterns.items():
                try:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    if matches:
                        if len(matches) == 1:
                            result[key] = matches[0]
                        else:
                            result[key] = matches
                except Exception as e:
                    extraction_errors.append({"pattern": key, "message": str(e)})
                    result["partial_success"] = True
            
            if extraction_errors:
                result["errors"].append({"type": "pattern_extraction", "details": extraction_errors})
            
            return result
        except ParsingError as e:
            # Handle specific parsing errors from the utility function
            self.logger.error(f"Parsing error extracting data from {source}: {str(e)}")
            return {
                "source": source,
                "content_type": content_type,
                "timestamp": asyncio.get_event_loop().time(),
                "success": False,
                "error_type": "parsing_error",
                "error_message": str(e),
                "errors": [{"type": "html_parsing", "message": str(e)}]
            }
        except Exception as e:
            # Handle any other unexpected errors
            self.logger.error(f"Error extracting data from {source}: {str(e)}")
            return {
                "source": source,
                "content_type": content_type,
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
        # Only apply rate limiting for remote URLs, not local files
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

    def _is_local_path(self, path: str) -> bool:
        """
        Check if a path is a local file path.
        
        Args:
            path: The path to check
            
        Returns:
            Boolean indicating if the path is a local file path
        """
        # Check if it's an absolute path
        if os.path.isabs(path):
            return True
            
        # Check if it's a relative path
        if not urlparse(path).scheme:
            return True
            
        return False
