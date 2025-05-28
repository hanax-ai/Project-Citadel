
"""
File document loader for Project Citadel LangChain integration.
"""

import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document

from citadel_core.logging import get_logger

from .base import BaseLoader
from .pdf import PDFLoader


class FileLoader(BaseLoader):
    """Loader for various file types."""
    
    def __init__(
        self,
        pdf_loader: Optional[PDFLoader] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the file loader.
        
        Args:
            pdf_loader: PDFLoader instance to use for PDF files.
            logger: Logger instance.
        """
        super().__init__(logger)
        
        self.pdf_loader = pdf_loader or PDFLoader()
        
        # Map of file extensions to handler methods
        self.handlers = {
            ".pdf": self._load_pdf,
            ".txt": self._load_text,
            ".md": self._load_text,
            ".html": self._load_text,
            ".htm": self._load_text,
            ".csv": self._load_text,
            ".json": self._load_text,
        }
    
    def load(
        self, 
        source: Union[str, Path, List[Union[str, Path]]],
        **kwargs
    ) -> List[Document]:
        """
        Load documents from files.
        
        Args:
            source: Path to file or list of paths.
            **kwargs: Additional loading parameters.
            
        Returns:
            List of loaded documents.
        """
        if isinstance(source, (str, Path)):
            sources = [source]
        else:
            sources = source
        
        documents = []
        
        for src in sources:
            try:
                path = Path(src)
                
                # Check if file exists
                if not path.exists():
                    self.logger.warning(f"File not found: {path}")
                    continue
                
                # Get file extension
                ext = path.suffix.lower()
                
                # Use the appropriate handler
                if ext in self.handlers:
                    docs = self.handlers[ext](path, **kwargs)
                    documents.extend(docs)
                else:
                    self.logger.warning(f"Unsupported file type: {ext}")
                
            except Exception as e:
                self.logger.error(f"Error loading file {src}: {str(e)}")
        
        return documents
    
    def _load_pdf(self, path: Path, **kwargs) -> List[Document]:
        """
        Load documents from a PDF file.
        
        Args:
            path: Path to the PDF file.
            **kwargs: Additional loading parameters.
            
        Returns:
            List of loaded documents.
        """
        return self.pdf_loader.load(path, **kwargs)
    
    def _load_text(self, path: Path, **kwargs) -> List[Document]:
        """
        Load documents from a text file.
        
        Args:
            path: Path to the text file.
            **kwargs: Additional loading parameters.
            
        Returns:
            List of loaded documents.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            document = self._create_document(
                text=text,
                metadata={
                    "source": str(path),
                    "file_type": path.suffix.lower()[1:],
                    "file_name": path.name,
                    "file_size": path.stat().st_size,
                }
            )
            
            return [document]
            
        except Exception as e:
            self.logger.error(f"Error loading text file {path}: {str(e)}")
            return []
