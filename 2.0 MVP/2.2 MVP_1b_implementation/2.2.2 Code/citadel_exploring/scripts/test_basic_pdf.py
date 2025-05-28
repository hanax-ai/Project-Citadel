#!/usr/bin/env python3
"""
Basic PDF Processing Test Script for Project Citadel.

This script tests the basic functionality of the PDF processing module
without running the full test suite.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the PDF processing module
from citadel_core.pdf_processing import extract_text_from_pdf, extract_metadata_from_pdf
from citadel_core.crawlers.pdf_crawler import PDFCrawler

def main():
    """Run the basic PDF processing test."""
    # Path to the sample PDF
    sample_pdf = Path(__file__).parent.parent / 'tests' / 'fixtures' / 'sample.pdf'
    
    if not sample_pdf.exists():
        print(f"Error: Sample PDF not found at {sample_pdf}")
        return 1
    
    print(f"Testing with sample PDF: {sample_pdf}")
    
    # Test text extraction
    print("\nTesting text extraction...")
    try:
        text = extract_text_from_pdf(sample_pdf)
        print(f"Text extracted successfully ({len(text)} characters)")
        print(f"Sample: {text[:100]}...")
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
    
    # Test metadata extraction
    print("\nTesting metadata extraction...")
    try:
        metadata = extract_metadata_from_pdf(sample_pdf)
        print("Metadata extracted successfully:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
    
    # Test PDFCrawler with local file
    print("\nTesting PDFCrawler with local file...")
    try:
        crawler = PDFCrawler(base_url="https://example.com", ocr_enabled=False)
        result = crawler.process_local_pdf(sample_pdf)
        print("PDF processed successfully by PDFCrawler")
        print(f"Extracted text length: {len(result['text'])} characters")
        print(f"Number of pages: {result['metadata']['page_count']}")
        print(f"Number of chunks: {len(result['chunks'])}")
    except Exception as e:
        print(f"Error processing PDF with PDFCrawler: {str(e)}")
    
    print("\nBasic PDF processing tests completed")
    return 0

if __name__ == '__main__':
    sys.exit(main())
