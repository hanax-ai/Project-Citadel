"""Tests for crawler_utils module."""

import unittest
from unittest import mock
import time
import requests
from bs4 import BeautifulSoup

from citadel_revisions import crawler_utils


class TestUrlValidation(unittest.TestCase):
    """Tests for URL validation functions."""
    
    def test_validate_url_basic(self):
        """Test basic URL validation."""
        # Valid URLs
        self.assertTrue(crawler_utils.validate_url(
            "https://example.com/page",
            base_url="https://example.com"
        ))
        
        self.assertTrue(crawler_utils.validate_url(
            "https://subdomain.example.com/page",
            base_url="https://example.com"
        ))
        
        # Invalid URLs
        self.assertFalse(crawler_utils.validate_url(
            "https://malicious.com/page",
            base_url="https://example.com"
        ))
        
        self.assertFalse(crawler_utils.validate_url(
            "ftp://example.com/file",
            base_url="https://example.com"
        ))
        
        self.assertFalse(crawler_utils.validate_url(
            "invalid-url",
            base_url="https://example.com"
        ))
    
    def test_validate_url_with_allowed_domains(self):
        """Test URL validation with allowed domains."""
        self.assertTrue(crawler_utils.validate_url(
            "https://allowed.com/page",
            base_url="https://example.com",
            allowed_domains=["allowed.com", "also-allowed.com"]
        ))
        
        self.assertFalse(crawler_utils.validate_url(
            "https://disallowed.com/page",
            base_url="https://example.com",
            allowed_domains=["allowed.com", "also-allowed.com"]
        ))
    
    def test_validate_url_with_patterns(self):
        """Test URL validation with patterns."""
        self.assertTrue(crawler_utils.validate_url(
            "https://example.com/article/123",
            base_url="https://example.com",
            url_patterns=["/article/", "/blog/"]
        ))
        
        self.assertFalse(crawler_utils.validate_url(
            "https://example.com/product/123",
            base_url="https://example.com",
            url_patterns=["/article/", "/blog/"]
        ))
        
        # Base URL should always be valid
        self.assertTrue(crawler_utils.validate_url(
            "https://example.com",
            base_url="https://example.com",
            url_patterns=["/article/", "/blog/"]
        ))
    
    def test_validate_url_with_excluded_patterns(self):
        """Test URL validation with excluded patterns."""
        self.assertFalse(crawler_utils.validate_url(
            "https://example.com/login",
            base_url="https://example.com",
            excluded_patterns=["/login", "/admin"]
        ))
        
        self.assertTrue(crawler_utils.validate_url(
            "https://example.com/article/123",
            base_url="https://example.com",
            excluded_patterns=["/login", "/admin"]
        ))
    
    def test_validate_url_with_callback(self):
        """Test URL validation with callback."""
        urls = []
        
        def callback(url):
            urls.append(url)
        
        crawler_utils.validate_url(
            "https://example.com/article/123",
            base_url="https://example.com",
            callback=callback
        )
        
        self.assertEqual(urls, ["https://example.com/article/123"])


class TestHtmlParsing(unittest.TestCase):
    """Tests for HTML parsing functions."""
    
    def test_parse_html(self):
        """Test HTML parsing."""
        html = "<html><body><h1>Title</h1><p>Paragraph</p></body></html>"
        soup = crawler_utils.parse_html(html)
        
        self.assertIsInstance(soup, BeautifulSoup)
        self.assertEqual(soup.h1.text, "Title")
        self.assertEqual(soup.p.text, "Paragraph")
    
    def test_extract_links(self):
        """Test link extraction."""
        html = """
        <html>
            <body>
                <a href="/page1">Page 1</a>
                <a href="page2">Page 2</a>
                <a href="https://example.org/page3">Page 3</a>
            </body>
        </html>
        """
        soup = crawler_utils.parse_html(html)
        links = crawler_utils.extract_links(soup, "https://example.com")
        
        self.assertEqual(len(links), 3)
        self.assertEqual(links[0], "https://example.com/page1")
        self.assertEqual(links[1], "https://example.com/page2")
        self.assertEqual(links[2], "https://example.org/page3")


class TestRateLimiting(unittest.TestCase):
    """Tests for rate limiting functions."""
    
    def test_apply_rate_limiting(self):
        """Test rate limiting application."""
        start_time = time.time()
        
        # This should not sleep
        new_time = crawler_utils.apply_rate_limiting(
            start_time - 2.0,  # Last request was 2 seconds ago
            min_request_interval=1.0
        )
        
        # Verify that it didn't sleep for long
        self.assertLess(time.time() - start_time, 0.1)
        
        # Reset start time
        start_time = time.time()
        
        # This should sleep for about 0.5 seconds
        new_time = crawler_utils.apply_rate_limiting(
            start_time - 0.5,  # Last request was 0.5 seconds ago
            min_request_interval=1.0
        )
        
        # Verify that it slept for about 0.5 seconds
        elapsed = time.time() - start_time
        self.assertGreater(elapsed, 0.4)
        self.assertLess(elapsed, 0.7)  # Allow some margin for error
    
    @mock.patch('time.sleep')
    def test_rate_limiter_decorator(self, mock_sleep):
        """Test rate limiter decorator."""
        
        @crawler_utils.rate_limiter(min_interval=1.0)
        def test_function():
            return "result"
        
        # First call should not sleep
        result = test_function()
        self.assertEqual(result, "result")
        mock_sleep.assert_not_called()
        
        # Second call should sleep
        result = test_function()
        self.assertEqual(result, "result")
        mock_sleep.assert_called_once()


class TestHttpSessionManagement(unittest.TestCase):
    """Tests for HTTP session management functions."""
    
    def test_get_session(self):
        """Test session creation."""
        session = crawler_utils.get_session(
            user_agent="TestCrawler/1.0",
            accept="text/html",
            accept_language="en-US",
            additional_headers={"X-Test": "test"}
        )
        
        self.assertIsInstance(session, requests.Session)
        self.assertEqual(session.headers["User-Agent"], "TestCrawler/1.0")
        self.assertEqual(session.headers["Accept"], "text/html")
        self.assertEqual(session.headers["Accept-Language"], "en-US")
        self.assertEqual(session.headers["X-Test"], "test")


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling functions."""
    
    def test_is_success_status(self):
        """Test success status check."""
        self.assertTrue(crawler_utils.is_success_status(200))
        self.assertTrue(crawler_utils.is_success_status(201))
        self.assertTrue(crawler_utils.is_success_status(299))
        
        self.assertFalse(crawler_utils.is_success_status(199))
        self.assertFalse(crawler_utils.is_success_status(300))
        self.assertFalse(crawler_utils.is_success_status(404))
        self.assertFalse(crawler_utils.is_success_status(500))
    
    def test_handle_http_error(self):
        """Test HTTP error handling."""
        # Create mock responses
        ok_response = mock.Mock()
        ok_response.status_code = 200
        
        not_found_response = mock.Mock()
        not_found_response.status_code = 404
        not_found_response.url = "https://example.com/not-found"
        
        server_error_response = mock.Mock()
        server_error_response.status_code = 500
        server_error_response.url = "https://example.com/error"
        
        # Test responses
        self.assertIsNone(crawler_utils.handle_http_error(ok_response))
        
        error_message = crawler_utils.handle_http_error(not_found_response)
        self.assertIn("Not found", error_message)
        self.assertIn("https://example.com/not-found", error_message)
        
        error_message = crawler_utils.handle_http_error(server_error_response)
        self.assertIn("Internal server error", error_message)
        self.assertIn("https://example.com/error", error_message)
    
    @mock.patch('time.sleep')
    @mock.patch('requests.Session.request')
    def test_safe_request(self, mock_request, mock_sleep):
        """Test safe request function."""
        # Mock session and validate function
        session = requests.Session()
        validate_func = lambda url: url.startswith("https://example.com")
        
        # Mock successful response
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response
        
        # Test successful request
        response = crawler_utils.safe_request(
            url="https://example.com/page",
            session=session,
            validate_func=validate_func,
            timeout=10,
            max_retries=3
        )
        
        self.assertEqual(response, mock_response)
        mock_request.assert_called_once()
        
        # Reset mocks
        mock_request.reset_mock()
        
        # Test invalid URL
        response = crawler_utils.safe_request(
            url="https://malicious.com/page",
            session=session,
            validate_func=validate_func,
            timeout=10,
            max_retries=3
        )
        
        self.assertIsNone(response)
        mock_request.assert_not_called()
        
        # Test request with retry
        mock_request.side_effect = [
            requests.RequestException("Error"),
            mock_response
        ]
        
        response = crawler_utils.safe_request(
            url="https://example.com/page",
            session=session,
            validate_func=validate_func,
            timeout=10,
            max_retries=3
        )
        
        self.assertEqual(response, mock_response)
        self.assertEqual(mock_request.call_count, 2)
        mock_sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
