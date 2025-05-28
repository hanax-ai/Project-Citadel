
"""
Text chunker for the Citadel LLM package.
"""

import re
from typing import Any, Dict, List, Optional, Union, Tuple

from .base import BaseProcessor


class TextChunker(BaseProcessor):
    """
    A processor for chunking text into smaller pieces for LLM processing.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        respect_paragraphs: bool = True,
        respect_sentences: bool = True,
        **kwargs
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Overlap between chunks in characters.
            respect_paragraphs: Whether to avoid breaking paragraphs.
            respect_sentences: Whether to avoid breaking sentences.
            **kwargs: Additional parameters for the base class.
        """
        super().__init__(**kwargs)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_paragraphs = respect_paragraphs
        self.respect_sentences = respect_sentences
    
    async def process(
        self, 
        text: str, 
        return_metadata: bool = False,
        **kwargs
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """
        Chunk the input text.
        
        Args:
            text: Input text to chunk.
            return_metadata: Whether to return metadata with each chunk.
            **kwargs: Additional processing parameters.
            
        Returns:
            List of text chunks or list of dictionaries with chunks and metadata.
        """
        if not text:
            return []
        
        chunks = []
        
        # Simple case: text is shorter than chunk size
        if len(text) <= self.chunk_size:
            if return_metadata:
                chunks.append({
                    "text": text,
                    "start": 0,
                    "end": len(text),
                    "index": 0
                })
            else:
                chunks.append(text)
            return chunks
        
        # Split text into paragraphs if respecting paragraphs
        if self.respect_paragraphs:
            paragraphs = re.split(r'\n\s*\n', text)
        else:
            paragraphs = [text]
        
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            # If respecting sentences, split paragraph into sentences
            if self.respect_sentences:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            else:
                sentences = [paragraph]
            
            for sentence in sentences:
                # If adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) + 1 > self.chunk_size and current_chunk:
                    # Add current chunk to chunks
                    if return_metadata:
                        chunks.append({
                            "text": current_chunk,
                            "start": current_start,
                            "end": current_start + len(current_chunk),
                            "index": chunk_index
                        })
                    else:
                        chunks.append(current_chunk)
                    
                    # Start new chunk with overlap
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:]
                    current_start += overlap_start
                    chunk_index += 1
                
                # Add sentence to current chunk
                if current_chunk and not current_chunk.endswith(" "):
                    current_chunk += " "
                current_chunk += sentence
            
            # Add paragraph break
            if self.respect_paragraphs and not paragraph.endswith("\n"):
                current_chunk += "\n\n"
        
        # Add final chunk
        if current_chunk:
            if return_metadata:
                chunks.append({
                    "text": current_chunk,
                    "start": current_start,
                    "end": current_start + len(current_chunk),
                    "index": chunk_index
                })
            else:
                chunks.append(current_chunk)
        
        return chunks
    
    async def chunk_pdf_content(
        self,
        pdf_content: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chunk the content of a PDF document.
        
        Args:
            pdf_content: PDF content dictionary from PDFProcessor.
            **kwargs: Additional processing parameters.
            
        Returns:
            Updated PDF content dictionary with chunks.
        """
        if not pdf_content or "text" not in pdf_content:
            return pdf_content
        
        # Create a copy of the PDF content
        result = pdf_content.copy()
        
        # Chunk the text
        chunks = await self.process(pdf_content["text"], return_metadata=True, **kwargs)
        
        # Update the chunks in the result
        result["chunks"] = chunks
        
        return result
