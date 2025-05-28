# Citadel Revisions Crawler Utilities

This module provides common utility functions for web crawlers to avoid code duplication across different crawler implementations.

## Overview

The `crawler_utils.py` module contains the following utility functions:

1. **URL Validation**
   - `validate_url()`: Validates URLs against allowed domains, schemes, and patterns

2. **HTML Parsing**
   - `parse_html()`: Parses HTML content using BeautifulSoup
   - `extract_links()`: Extracts all links from a BeautifulSoup object

3. **Rate Limiting**
   - `rate_limiter()`: Decorator to implement rate limiting for functions
   - `apply_rate_limiting()`: Function to apply rate limiting based on the last request time

4. **Error Handling**
   - `safe_request()`: Makes HTTP requests with error handling and retries
   - `is_success_status()`: Checks if an HTTP status code indicates success
   - `handle_http_error()`: Handles HTTP errors and returns appropriate error messages
   - `retry_on_error()`: Decorator to retry a function on error

5. **HTTP Session Management**
   - `get_session()`: Creates and configures a requests session

## Usage

Here's a simple example of how to use the crawler utilities:

```python
from citadel_revisions import crawler_utils

# Create a session
session = crawler_utils.get_session(user_agent="MyCrawler/1.0")

# Define a validation function
def validate_my_url(url):
    return crawler_utils.validate_url(
        url=url,
        base_url="https://example.com",
        allowed_domains=["example.org", "example.net"],
        url_patterns=["/article/", "/blog/"]
    )

# Make a safe request
response = crawler_utils.safe_request(
    url="https://example.com/article/123",
    session=session,
    validate_func=validate_my_url,
    timeout=30,
    max_retries=3
)

if response:
    # Parse the HTML
    soup = crawler_utils.parse_html(response.text)
    
    # Extract links
    links = crawler_utils.extract_links(soup, "https://example.com/article/123")
    
    # Process the links
    for link in links:
        print(link)
```

For a more complete example, see the `example_crawler.py` file.

## Key Improvements

The `crawler_utils.py` module addresses several issues in the original crawler implementations:

1. **Fixed Unreachable Code**: The URL validation function has been rewritten to eliminate the unreachable code issue.

2. **Reduced Code Duplication**: Common functionality has been extracted into utility functions to avoid duplication across crawler implementations.

3. **Improved Flexibility**: The utility functions are designed to be flexible and configurable to accommodate different crawler requirements.

4. **Better Error Handling**: The module includes comprehensive error handling and retry mechanisms.

5. **Type Hints**: All functions include proper type hints for better IDE support and code quality.

## Best Practices

When using the crawler utilities, consider the following best practices:

1. **Respect Robots.txt**: Always check and respect the robots.txt file of the websites you crawl.

2. **Use Appropriate Rate Limiting**: Adjust the rate limiting parameters based on the website's requirements.

3. **Handle Errors Gracefully**: Use the error handling utilities to handle errors gracefully and avoid crashing.

4. **Customize URL Validation**: Customize the URL validation function to match your specific requirements.

5. **Monitor Crawler Performance**: Monitor the performance of your crawler and adjust parameters as needed.
