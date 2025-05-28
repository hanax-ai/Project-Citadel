"""Main entry point for Citadel Revisions package.

This module provides a simple command-line interface to run the example crawler.
"""

import argparse
import logging
import sys
from typing import List

from citadel_revisions.crawlers.example_crawler import ExampleCrawler


def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Citadel Revisions Crawler")
    parser.add_argument("url", help="URL to crawl")
    parser.add_argument(
        "--max-pages", 
        type=int, 
        default=10, 
        help="Maximum number of pages to crawl (default: 10)"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=30, 
        help="Request timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--max-retries", 
        type=int, 
        default=3, 
        help="Maximum number of retries for failed requests (default: 3)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    return parser.parse_args(args)


def main(args: List[str] = None) -> int:
    """Main entry point.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    if args is None:
        args = sys.argv[1:]
        
    parsed_args = parse_args(args)
    
    # Configure logging
    log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create and run crawler
    try:
        crawler = ExampleCrawler(
            base_url=parsed_args.url,
            timeout=parsed_args.timeout,
            max_retries=parsed_args.max_retries
        )
        
        results = crawler.crawl(max_pages=parsed_args.max_pages)
        
        print(f"\nFound {len(results)} content pages:")
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result['url']
            print(f"{i}. {title} ({url})")
            
        return 0
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
