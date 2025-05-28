
"""
PDF Processing Module for Project Citadel.

This module provides comprehensive PDF processing capabilities including:
- Text extraction
- Metadata extraction
- Image handling with OCR
- Document structure preservation
- Content chunking for LLM integration
"""

import io
import os
import re
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path

# PDF processing
import PyPDF2
from PyPDF2 import PdfReader

# Image processing and OCR
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    A comprehensive PDF processing class that handles text extraction,
    metadata extraction, image processing with OCR, and document structure
    preservation.
    """
    
    def __init__(self, ocr_enabled: bool = True, language: str = 'eng'):
        """
        Initialize the PDF processor.
        
        Args:
            ocr_enabled: Whether to enable OCR for images
            language: Language for OCR (default: 'eng')
        """
        self.ocr_enabled = ocr_enabled and HAS_OCR
        self.language = language
        
        if ocr_enabled and not HAS_OCR:
            logger.warning("OCR dependencies not installed. OCR will be disabled.")
    
    def process_pdf(self, pdf_path: Union[str, Path, io.BytesIO]) -> Dict[str, Any]:
        """
        Process a PDF file and extract all relevant information.
        
        Args:
            pdf_path: Path to the PDF file or a file-like object
            
        Returns:
            Dictionary containing extracted text, metadata, and structure
        """
        try:
            reader = PdfReader(pdf_path)
            
            # Extract basic information
            result = {
                'metadata': self.extract_metadata(reader),
                'pages': [],
                'text': '',
                'images': [],
                'structure': {
                    'headings': [],
                    'paragraphs': [],
                    'tables': []
                }
            }
            
            # Process each page
            for i, page in enumerate(reader.pages):
                page_result = self.process_page(page, page_number=i+1)
                result['pages'].append(page_result)
                result['text'] += page_result['text'] + '\n\n'
                result['images'].extend(page_result['images'])
                
                # Update document structure
                for heading in page_result['structure']['headings']:
                    result['structure']['headings'].append({
                        'page': i+1,
                        'text': heading['text'],
                        'level': heading['level']
                    })
                
                for paragraph in page_result['structure']['paragraphs']:
                    result['structure']['paragraphs'].append({
                        'page': i+1,
                        'text': paragraph
                    })
                
                for table in page_result['structure']['tables']:
                    result['structure']['tables'].append({
                        'page': i+1,
                        'data': table
                    })
            
            # Generate chunks for LLM processing
            result['chunks'] = self.chunk_content(result['text'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def extract_metadata(self, reader: PdfReader) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file.
        
        Args:
            reader: PyPDF2 PdfReader object
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {}
        
        # Extract standard metadata
        info = reader.metadata
        if info:
            for key, value in info.items():
                # Convert from PyPDF2's private types to standard Python types
                if isinstance(value, (str, int, float, bool, list, dict)):
                    metadata[key] = value
                else:
                    # Convert other types to string
                    metadata[key] = str(value)
        
        # Add additional metadata
        metadata['page_count'] = len(reader.pages)
        
        return metadata
    
    def process_page(self, page: Any, page_number: int) -> Dict[str, Any]:
        """
        Process a single page from a PDF.
        
        Args:
            page: PyPDF2 page object
            page_number: Page number (1-based)
            
        Returns:
            Dictionary containing page information
        """
        result = {
            'page_number': page_number,
            'text': '',
            'images': [],
            'structure': {
                'headings': [],
                'paragraphs': [],
                'tables': []
            }
        }
        
        # Extract text
        text = page.extract_text()
        if text:
            result['text'] = text
            
            # Identify document structure
            result['structure'] = self._identify_structure(text)
        
        # Extract and process images if OCR is enabled
        if self.ocr_enabled:
            try:
                # This is a simplified approach - in a real implementation,
                # you would use PyMuPDF or a similar library to extract images
                # PyPDF2 has limited image extraction capabilities
                
                # For demonstration purposes, we'll assume we have image data
                # In a real implementation, you would extract actual images
                
                # Placeholder for image extraction
                # In a real implementation, you would extract actual images from the PDF
                # and process them with OCR
                pass
                
            except Exception as e:
                logger.warning(f"Error extracting images from page {page_number}: {str(e)}")
        
        return result
    
    def _identify_structure(self, text: str) -> Dict[str, List]:
        """
        Identify document structure from text.
        
        Args:
            text: Extracted text from a page
            
        Returns:
            Dictionary containing identified structure elements
        """
        structure = {
            'headings': [],
            'paragraphs': [],
            'tables': []
        }
        
        # Split text into lines
        lines = text.split('\n')
        
        # Process lines to identify structure
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                # Empty line indicates paragraph break
                if current_paragraph:
                    structure['paragraphs'].append(' '.join(current_paragraph))
                    current_paragraph = []
                continue
            
            # Check if line is a heading
            if self._is_heading(line):
                # Determine heading level based on characteristics
                level = self._determine_heading_level(line)
                structure['headings'].append({
                    'text': line,
                    'level': level
                })
            else:
                # Add to current paragraph
                current_paragraph.append(line)
        
        # Add final paragraph if exists
        if current_paragraph:
            structure['paragraphs'].append(' '.join(current_paragraph))
        
        # Table detection would require more sophisticated analysis
        # This is a placeholder for table detection
        
        return structure
    
    def _is_heading(self, line: str) -> bool:
        """
        Determine if a line is likely a heading.
        
        Args:
            line: Text line
            
        Returns:
            True if line is likely a heading
        """
        # Simple heuristics for heading detection
        # In a real implementation, you would use more sophisticated methods
        
        # Check if line is short
        if len(line) < 100 and len(line) > 0:
            # Check if line ends with colon or is all uppercase
            if line.endswith(':') or line.isupper():
                return True
            
            # Check if line starts with common heading patterns
            heading_patterns = [
                r'^[0-9]+\.[0-9]*\s',  # Numbered headings like "1.2 Title"
                r'^[A-Z][a-z]+\s[0-9]+:',  # "Section 1:"
                r'^Chapter\s[0-9]+',  # "Chapter 1"
                r'^[A-Z][A-Z\s]+$'  # ALL CAPS or Title Case
            ]
            
            for pattern in heading_patterns:
                if re.match(pattern, line):
                    return True
        
        return False
    
    def _determine_heading_level(self, heading: str) -> int:
        """
        Determine the level of a heading.
        
        Args:
            heading: Heading text
            
        Returns:
            Heading level (1-6)
        """
        # Simple heuristics for heading level detection
        # In a real implementation, you would use more sophisticated methods
        
        # Check for numbered headings
        if re.match(r'^[0-9]+\.[0-9]+\.[0-9]+', heading):
            return 3  # Like 1.2.3
        elif re.match(r'^[0-9]+\.[0-9]+', heading):
            return 2  # Like 1.2
        elif re.match(r'^[0-9]+\.', heading):
            return 1  # Like 1.
        
        # Check based on text characteristics
        if heading.isupper():
            return 1
        elif heading.startswith('Chapter') or heading.startswith('CHAPTER'):
            return 1
        elif heading.startswith('Section') or heading.startswith('SECTION'):
            return 2
        
        # Default level
        return 3
    
    def perform_ocr(self, image_data: bytes) -> str:
        """
        Perform OCR on an image.
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            Extracted text from the image
        """
        if not self.ocr_enabled:
            logger.warning("OCR is disabled or dependencies not installed")
            return ""
        
        try:
            image = Image.open(io.BytesIO(image_data))
            text = pytesseract.image_to_string(image, lang=self.language)
            return text
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            return ""
    
    def chunk_content(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Split content into chunks suitable for LLM processing.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # Simple chunking by character count
        # In a real implementation, you would use more sophisticated methods
        # that respect sentence and paragraph boundaries
        
        if len(text) <= chunk_size:
            chunks.append(text)
        else:
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                
                # Try to find a good breaking point (end of sentence)
                if end < len(text):
                    # Look for sentence endings within the last 20% of the chunk
                    search_start = max(start + int(chunk_size * 0.8), start)
                    search_text = text[search_start:end]
                    
                    # Find the last sentence ending
                    sentence_endings = [
                        search_text.rfind('. '),
                        search_text.rfind('? '),
                        search_text.rfind('! '),
                        search_text.rfind('.\n'),
                        search_text.rfind('?\n'),
                        search_text.rfind('!\n')
                    ]
                    
                    best_ending = max(sentence_endings)
                    if best_ending != -1:
                        # Adjust the end position
                        end = search_start + best_ending + 2  # +2 to include the period and space
                
                # Add the chunk
                chunks.append(text[start:end])
                
                # Move to next chunk with overlap
                start = end - overlap
        
        return chunks

def extract_text_from_pdf(pdf_path: Union[str, Path, io.BytesIO]) -> str:
    """
    Simple utility function to extract text from a PDF.
    
    Args:
        pdf_path: Path to the PDF file or a file-like object
        
    Returns:
        Extracted text as a string
    """
    processor = PDFProcessor(ocr_enabled=False)
    result = processor.process_pdf(pdf_path)
    return result['text']

def extract_metadata_from_pdf(pdf_path: Union[str, Path, io.BytesIO]) -> Dict[str, Any]:
    """
    Simple utility function to extract metadata from a PDF.
    
    Args:
        pdf_path: Path to the PDF file or a file-like object
        
    Returns:
        Dictionary containing metadata
    """
    reader = PdfReader(pdf_path)
    processor = PDFProcessor(ocr_enabled=False)
    return processor.extract_metadata(reader)

def chunk_pdf_content(pdf_path: Union[str, Path, io.BytesIO], 
                     chunk_size: int = 1000, 
                     overlap: int = 100) -> List[str]:
    """
    Simple utility function to extract and chunk text from a PDF.
    
    Args:
        pdf_path: Path to the PDF file or a file-like object
        chunk_size: Maximum size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    processor = PDFProcessor(ocr_enabled=False)
    result = processor.process_pdf(pdf_path)
    return processor.chunk_content(result['text'], chunk_size, overlap)
