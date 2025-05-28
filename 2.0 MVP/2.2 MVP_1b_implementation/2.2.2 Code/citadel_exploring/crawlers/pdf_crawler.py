
"""
PDF Crawler for Project Citadel.

This module provides a specialized crawler for PDF documents that extends
the BaseCrawler class and uses the PDF processing module.
"""

import os
import tempfile
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import io
import requests
from urllib.parse import urlparse

# Import the base crawler
import sys
import os
from pathlib import Path

# Add the legacy directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'legacy' / 'src'))

# Import the base crawler
from citadel_revisions.crawlers.base_crawler import BaseCrawler

# Import the PDF processing module
from citadel_core.pdf_processing import PDFProcessor

logger = logging.getLogger(__name__)

class PDFCrawler(BaseCrawler):
    """
    Specialized crawler for PDF documents that extends BaseCrawler.
    """
    
    def __init__(self, base_url: str, 
                 timeout: int = 30, 
                 max_retries: int = 3,
                 ocr_enabled: bool = True,
                 language: str = 'eng',
                 chunk_size: int = 1000,
                 chunk_overlap: int = 100):
        """
        Initialize the PDF crawler.
        
        Args:
            base_url: The base URL to crawl
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            ocr_enabled: Whether to enable OCR for images in PDFs
            language: Language for OCR (default: 'eng')
            chunk_size: Maximum size of each text chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        super().__init__(base_url, timeout, max_retries)
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(ocr_enabled=ocr_enabled, language=language)
        
        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Results storage
        self.results = []
    
    def crawl(self, urls: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Crawl PDF documents from the specified URLs.
        
        Args:
            urls: List of URLs to PDF documents. If None, will use the base_url.
            
        Returns:
            List of dictionaries containing processed PDF data
        """
        if urls is None:
            urls = [self.base_url]
        
        self.results = []
        
        for url in urls:
            try:
                logger.info(f"Crawling PDF from URL: {url}")
                
                # Download the PDF
                pdf_data = self._download_pdf(url)
                if pdf_data:
                    # Process the PDF
                    result = self._process_pdf(pdf_data, url)
                    if result:
                        self.results.append(result)
                
            except Exception as e:
                logger.error(f"Error crawling PDF from {url}: {str(e)}")
        
        return self.results
    
    def _download_pdf(self, url: str) -> Optional[bytes]:
        """
        Download a PDF from a URL.
        
        Args:
            url: URL to the PDF document
            
        Returns:
            PDF data as bytes or None if download failed
        """
        response = self._safe_request(url)
        if response is None:
            logger.error(f"Failed to download PDF from {url}")
            return None
        
        # Check if the response is a PDF
        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
            logger.warning(f"URL {url} does not appear to be a PDF (Content-Type: {content_type})")
            # We'll try to process it anyway, as sometimes Content-Type is incorrect
        
        return response.content
    
    def _process_pdf(self, pdf_data: bytes, source_url: str) -> Optional[Dict[str, Any]]:
        """
        Process a PDF document.
        
        Args:
            pdf_data: PDF data as bytes
            source_url: Source URL of the PDF
            
        Returns:
            Dictionary containing processed PDF data or None if processing failed
        """
        try:
            # Create a BytesIO object from the PDF data
            pdf_file = io.BytesIO(pdf_data)
            
            # Process the PDF
            result = self.pdf_processor.process_pdf(pdf_file)
            
            # Add source information
            result['source_url'] = source_url
            result['filename'] = os.path.basename(urlparse(source_url).path) or 'unknown.pdf'
            
            # Generate chunks for LLM processing
            result['chunks'] = self.pdf_processor.chunk_content(
                result['text'], 
                chunk_size=self.chunk_size, 
                overlap=self.chunk_overlap
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF from {source_url}: {str(e)}")
            return None
    
    def save_pdf(self, url: str, output_path: Union[str, Path]) -> bool:
        """
        Download and save a PDF to disk.
        
        Args:
            url: URL to the PDF document
            output_path: Path where to save the PDF
            
        Returns:
            True if successful, False otherwise
        """
        pdf_data = self._download_pdf(url)
        if pdf_data is None:
            return False
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(pdf_data)
            
            logger.info(f"PDF saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving PDF to {output_path}: {str(e)}")
            return False
    
    def process_local_pdf(self, pdf_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Process a local PDF file.
        
        Args:
            pdf_path: Path to the local PDF file
            
        Returns:
            Dictionary containing processed PDF data or None if processing failed
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return None
            
            # Process the PDF
            result = self.pdf_processor.process_pdf(pdf_path)
            
            # Add source information
            result['source'] = 'local'
            result['filename'] = pdf_path.name
            result['filepath'] = str(pdf_path.absolute())
            
            # Generate chunks for LLM processing
            result['chunks'] = self.pdf_processor.chunk_content(
                result['text'], 
                chunk_size=self.chunk_size, 
                overlap=self.chunk_overlap
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing local PDF {pdf_path}: {str(e)}")
            return None
