#!/usr/bin/env python3
"""
PDF Processing Demo Script for Project Citadel.

This script demonstrates the PDF processing capabilities implemented
for Project Citadel, including text extraction, metadata extraction,
and content chunking.
"""

import os
import sys
from pathlib import Path
import json
import argparse

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the PDF processing module
from citadel_core.pdf_processing import PDFProcessor, extract_text_from_pdf, extract_metadata_from_pdf
from citadel_core.crawlers.pdf_crawler import PDFCrawler

def main():
    """Run the PDF processing demo."""
    parser = argparse.ArgumentParser(description='PDF Processing Demo for Project Citadel')
    parser.add_argument('--pdf', type=str, help='Path to a PDF file to process')
    parser.add_argument('--url', type=str, help='URL to a PDF file to process')
    parser.add_argument('--output', type=str, help='Path to save the output JSON')
    parser.add_argument('--ocr', action='store_true', help='Enable OCR for images')
    args = parser.parse_args()
    
    # Use the sample PDF if no PDF file or URL is provided
    if not args.pdf and not args.url:
        sample_pdf = Path(__file__).parent.parent / 'tests' / 'fixtures' / 'sample.pdf'
        if sample_pdf.exists():
            args.pdf = str(sample_pdf)
            print(f"Using sample PDF: {args.pdf}")
        else:
            print("Error: No PDF file or URL provided, and sample PDF not found.")
            return 1
    
    # Process the PDF
    result = None
    
    if args.pdf:
        print(f"Processing local PDF: {args.pdf}")
        pdf_path = Path(args.pdf)
        
        # Create a PDF processor
        processor = PDFProcessor(ocr_enabled=args.ocr)
        result = processor.process_pdf(pdf_path)
        
        # Add source information
        result['source'] = 'local'
        result['filename'] = pdf_path.name
        result['filepath'] = str(pdf_path.absolute())
        
    elif args.url:
        print(f"Processing PDF from URL: {args.url}")
        
        # Create a PDF crawler
        crawler = PDFCrawler(base_url=args.url, ocr_enabled=args.ocr)
        
        # Crawl the PDF
        results = crawler.crawl([args.url])
        if results:
            result = results[0]
    
    if result:
        # Print a summary of the processed PDF
        print("\nPDF Processing Summary:")
        print(f"Filename: {result.get('filename', 'Unknown')}")
        print(f"Pages: {result['metadata'].get('page_count', 0)}")
        
        # Print metadata
        print("\nMetadata:")
        for key, value in result['metadata'].items():
            print(f"  {key}: {value}")
        
        # Print text sample
        text_sample = result['text'][:500] + "..." if len(result['text']) > 500 else result['text']
        print(f"\nText Sample:\n{text_sample}")
        
        # Print structure information
        print("\nDocument Structure:")
        print(f"  Headings: {len(result['structure']['headings'])}")
        print(f"  Paragraphs: {len(result['structure']['paragraphs'])}")
        print(f"  Tables: {len(result['structure']['tables'])}")
        
        # Print chunking information
        print(f"\nChunks for LLM processing: {len(result['chunks'])}")
        
        # Save the result to a JSON file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"\nOutput saved to: {output_path}")
        
        return 0
    else:
        print("Error: Failed to process the PDF.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
