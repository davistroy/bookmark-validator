"""
Unit tests for URL validator module.

Tests the URLValidator for validating bookmark URLs with retry logic.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
from datetime import datetime

from bookmark_processor.core.url_validator import (
    URLValidator,
    ValidationResult,
    ValidationError
)
from bookmark_processor.utils.intelligent_rate_limiter import IntelligentRateLimiter
from bookmark_processor.utils.browser_simulator import BrowserSimulator
from bookmark_processor.utils.retry_handler import RetryHandler


class TestValidationResult:
    """Test ValidationResult class."""
    
    def test_successful_result(self):
        """Test creating a successful validation result."""
        result = ValidationResult(
            url="https://example.com",
            is_valid=True,
            final_url="https://example.com",
            status_code=200,
            response_time=0.5,
            attempts=1
        )
        
        assert result.url == "https://example.com"
        assert result.is_valid is True
        assert result.final_url == "https://example.com"
        assert result.status_code == 200
        assert result.response_time == 0.5
        assert result.attempts == 1
        assert result.error is None
        assert result.redirect_chain == []
    
    def test_failed_result(self):
        """Test creating a failed validation result."""
        error = ValidationError("Connection timeout")
        result = ValidationResult(
            url="https://example.com",
            is_valid=False,
            error=error,
            attempts=3
        )
        
        assert result.url == "https://example.com"
        assert result.is_valid is False
        assert result.error == error
        assert result.attempts == 3
        assert result.status_code is None
        assert result.final_url is None
    
    def test_result_with_redirects(self):
        """Test result with redirect chain."""
        redirects = ["https://example.com", "https://www.example.com"]
        result = ValidationResult(
            url="https://example.com",
            is_valid=True,
            final_url="https://www.example.com",
            status_code=200,
            redirect_chain=redirects
        )
        
        assert result.redirect_chain == redirects
        assert result.final_url == "https://www.example.com"
    
    def test_string_representation(self):
        """Test string representation of ValidationResult."""
        result = ValidationResult(
            url="https://example.com",
            is_valid=True,
            status_code=200
        )
        
        str_repr = str(result)
        assert "https://example.com" in str_repr
        assert "200" in str_repr
        assert "valid" in str_repr.lower()


class TestURLValidator:
    """Test URLValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a URLValidator instance with mocked dependencies."""
        with patch('bookmark_processor.core.url_validator.IntelligentRateLimiter'), \
             patch('bookmark_processor.core.url_validator.BrowserSimulator'), \
             patch('bookmark_processor.core.url_validator.RetryHandler'):
            return URLValidator(timeout=5, max_retries=2)
    
    def test_initialization(self, validator):
        """Test URLValidator initialization."""
        assert validator.timeout == 5
        assert validator.max_retries == 2
        assert validator.session is not None
        assert hasattr(validator, 'rate_limiter')
        assert hasattr(validator, 'browser_simulator')
        assert hasattr(validator, 'retry_handler')
    
    def test_validate_single_url_success(self, validator):
        """Test successful validation of a single URL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.elapsed.total_seconds.return_value = 0.5
        mock_response.history = []
        
        with patch.object(validator.session, 'get', return_value=mock_response):
            result = validator.validate_single_url("https://example.com")
        
        assert result.is_valid is True
        assert result.status_code == 200
        assert result.final_url == "https://example.com"
        assert result.response_time == 0.5
        assert result.attempts == 1
    
    def test_validate_single_url_with_redirect(self, validator):
        """Test validation with redirects."""
        # Mock redirect history
        redirect_resp = Mock()
        redirect_resp.url = "https://example.com"
        
        final_resp = Mock()
        final_resp.status_code = 200
        final_resp.url = "https://www.example.com"
        final_resp.elapsed.total_seconds.return_value = 0.8
        final_resp.history = [redirect_resp]
        
        with patch.object(validator.session, 'get', return_value=final_resp):
            result = validator.validate_single_url("https://example.com")
        
        assert result.is_valid is True
        assert result.status_code == 200
        assert result.final_url == "https://www.example.com"
        assert len(result.redirect_chain) == 2
        assert result.redirect_chain[0] == "https://example.com"
        assert result.redirect_chain[1] == "https://www.example.com"
    
    def test_validate_single_url_timeout(self, validator):
        """Test validation with timeout error."""
        with patch.object(validator.session, 'get', side_effect=Timeout("Request timeout")):
            result = validator.validate_single_url("https://example.com")
        
        assert result.is_valid is False
        assert "timeout" in result.error.message.lower()
        assert result.status_code is None
    
    def test_validate_single_url_connection_error(self, validator):
        """Test validation with connection error."""
        with patch.object(validator.session, 'get', side_effect=ConnectionError("Connection failed")):
            result = validator.validate_single_url("https://example.com")
        
        assert result.is_valid is False
        assert "connection" in result.error.message.lower()
    
    def test_validate_single_url_invalid_status(self, validator):
        """Test validation with invalid status code."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.url = "https://example.com/notfound"
        mock_response.elapsed.total_seconds.return_value = 0.3
        mock_response.history = []
        
        with patch.object(validator.session, 'get', return_value=mock_response):
            result = validator.validate_single_url("https://example.com/notfound")
        
        assert result.is_valid is False
        assert result.status_code == 404
        assert "404" in result.error.message
    
    def test_validate_single_url_with_retries(self, validator):
        """Test validation with retry logic."""
        # First attempt fails, second succeeds
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.elapsed.total_seconds.return_value = 0.5
        mock_response.history = []
        
        side_effects = [Timeout("First attempt timeout"), mock_response]
        
        with patch.object(validator.session, 'get', side_effect=side_effects):
            result = validator.validate_single_url("https://example.com")
        
        assert result.is_valid is True
        assert result.status_code == 200
        assert result.attempts > 1
    
    def test_validate_single_url_max_retries_exceeded(self, validator):
        """Test validation when max retries are exceeded."""
        with patch.object(validator.session, 'get', side_effect=Timeout("Persistent timeout")):
            result = validator.validate_single_url("https://example.com")
        
        assert result.is_valid is False
        assert result.attempts > 1
        assert "timeout" in result.error.message.lower()
    
    def test_validate_single_url_rate_limiting(self, validator):
        """Test that rate limiting is applied."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.elapsed.total_seconds.return_value = 0.5
        mock_response.history = []
        
        with patch.object(validator.session, 'get', return_value=mock_response), \
             patch.object(validator.rate_limiter, 'wait_if_needed') as mock_wait:
            
            result = validator.validate_single_url("https://example.com")
            
            # Rate limiter should have been called
            mock_wait.assert_called_once()
    
    def test_validate_single_url_browser_simulation(self, validator):
        """Test that browser headers are applied."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.elapsed.total_seconds.return_value = 0.5
        mock_response.history = []
        
        mock_headers = {"User-Agent": "Mozilla/5.0 Test Browser"}
        validator.browser_simulator.get_headers.return_value = mock_headers
        
        with patch.object(validator.session, 'get', return_value=mock_response) as mock_get:
            result = validator.validate_single_url("https://example.com")
            
            # Should have called get with browser headers
            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            assert 'headers' in call_kwargs
    
    def test_batch_validate(self, validator):
        """Test batch validation of multiple URLs."""
        urls = [
            "https://example.com",
            "https://test.com", 
            "https://invalid.com"
        ]
        
        # Mock responses for each URL
        responses = []
        for i, url in enumerate(urls):
            mock_resp = Mock()
            mock_resp.status_code = 200 if i < 2 else 404
            mock_resp.url = url
            mock_resp.elapsed.total_seconds.return_value = 0.5
            mock_resp.history = []
            responses.append(mock_resp)
        
        with patch.object(validator.session, 'get', side_effect=responses):
            results = validator.batch_validate(urls)
        
        assert len(results) == 3
        assert results[0].is_valid is True
        assert results[1].is_valid is True
        assert results[2].is_valid is False
    
    def test_batch_validate_with_progress_callback(self, validator):
        """Test batch validation with progress callback."""
        urls = ["https://example.com", "https://test.com"]
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.url = "https://example.com"
        mock_response.elapsed.total_seconds.return_value = 0.5
        mock_response.history = []
        
        progress_calls = []
        def progress_callback(url):
            progress_calls.append(url)
        
        with patch.object(validator.session, 'get', return_value=mock_response):
            results = validator.batch_validate(urls, progress_callback=progress_callback)
        
        assert len(progress_calls) == 2
        assert progress_calls[0] == urls[0]
        assert progress_calls[1] == urls[1]
    
    def test_batch_validate_without_retries(self, validator):
        """Test batch validation with retries disabled."""
        urls = ["https://example.com"]
        
        with patch.object(validator.session, 'get', side_effect=Timeout("Timeout")):
            results = validator.batch_validate(urls, enable_retries=False)
        
        assert len(results) == 1
        assert results[0].is_valid is False
        assert results[0].attempts == 1  # No retries
    
    def test_is_valid_url_scheme(self, validator):
        """Test URL scheme validation."""
        valid_urls = [
            "https://example.com",
            "http://example.com",
            "https://www.example.com/path?query=1"
        ]
        
        invalid_urls = [
            "",
            "not-a-url",
            "ftp://example.com",
            "javascript:void(0)",
            "mailto:test@example.com"
        ]
        
        for url in valid_urls:
            assert validator._is_valid_url_scheme(url) is True, f"Should be valid: {url}"
        
        for url in invalid_urls:
            assert validator._is_valid_url_scheme(url) is False, f"Should be invalid: {url}"
    
    def test_normalize_url(self, validator):
        """Test URL normalization."""
        test_cases = [
            ("https://example.com", "https://example.com"),
            ("https://example.com/", "https://example.com"),
            ("https://EXAMPLE.COM/Path", "https://example.com/Path"),
            ("https://example.com/path?b=2&a=1", "https://example.com/path?a=1&b=2"),
            ("example.com", "https://example.com"),
            ("www.example.com", "https://www.example.com"),
        ]
        
        for input_url, expected in test_cases:
            result = validator._normalize_url(input_url)
            assert result == expected, f"Failed for {input_url}: got {result}, expected {expected}"
    
    def test_get_validation_statistics(self, validator):
        """Test getting validation statistics."""
        # Simulate some validation results
        validator.total_validated = 100
        validator.successful_validations = 85
        validator.failed_validations = 15
        validator.total_response_time = 50.0
        
        stats = validator.get_validation_statistics()
        
        assert stats['total_validated'] == 100
        assert stats['successful_validations'] == 85
        assert stats['failed_validations'] == 15
        assert stats['success_rate'] == 85.0
        assert stats['average_response_time'] == 0.5
    
    def test_get_validation_statistics_empty(self, validator):
        """Test getting statistics when no validations performed."""
        stats = validator.get_validation_statistics()
        
        assert stats['total_validated'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['average_response_time'] == 0.0
    
    def test_session_configuration(self, validator):
        """Test that session is properly configured."""
        session = validator.session
        
        # Should have appropriate timeout
        assert hasattr(session, 'timeout') or validator.timeout is not None
        
        # Should have retry configuration
        assert hasattr(session, 'adapters')
    
    def test_cleanup(self, validator):
        """Test cleanup method."""
        validator.cleanup()
        # Should not raise any errors
    
    def test_concurrent_validation_safety(self, validator):
        """Test that validator is thread-safe for concurrent use."""
        import threading
        import time
        
        urls = [f"https://example{i}.com" for i in range(10)]
        results = []
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.elapsed.total_seconds.return_value = 0.1
        mock_response.history = []
        
        def validate_url(url):
            mock_response.url = url
            with patch.object(validator.session, 'get', return_value=mock_response):
                result = validator.validate_single_url(url)
                results.append(result)
        
        threads = []
        for url in urls:
            thread = threading.Thread(target=validate_url, args=(url,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == len(urls)
        assert all(r.is_valid for r in results)


if __name__ == "__main__":
    pytest.main([__file__])