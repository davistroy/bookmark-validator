"""
URL Validation Module

Validates bookmark URLs with intelligent retry logic, rate limiting, and browser simulation.
Handles large datasets efficiently with concurrent processing and comprehensive error reporting.
"""

import requests
import logging
import time
import threading
from typing import List, Dict, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urljoin, quote
from pathlib import Path
import ssl
import socket
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..utils.intelligent_rate_limiter import IntelligentRateLimiter
from ..utils.browser_simulator import BrowserSimulator
from ..utils.retry_handler import RetryHandler, ErrorType
from ..utils.security_validator import SecurityValidator, SecurityValidationResult
from ..utils.secure_logging import secure_logger


@dataclass
class ValidationResult:
    """Result of URL validation"""
    url: str
    is_valid: bool
    status_code: Optional[int] = None
    final_url: Optional[str] = None  # After redirects
    response_time: float = 0.0
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    security_validation: Optional[SecurityValidationResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'url': self.url,
            'is_valid': self.is_valid,
            'status_code': self.status_code,
            'final_url': self.final_url,
            'response_time': self.response_time,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'content_type': self.content_type,
            'content_length': self.content_length,
            'timestamp': self.timestamp.isoformat(),
            'security_validation': self.security_validation.to_dict() if self.security_validation else None
        }


@dataclass
class ValidationStats:
    """Statistics for validation process"""
    total_urls: int = 0
    valid_urls: int = 0
    invalid_urls: int = 0
    timeout_urls: int = 0
    error_urls: int = 0
    redirected_urls: int = 0
    total_time: float = 0.0
    average_response_time: float = 0.0
    error_distribution: Dict[str, int] = field(default_factory=dict)
    
    def update_from_result(self, result: ValidationResult) -> None:
        """Update statistics from a validation result"""
        self.total_urls += 1
        self.total_time += result.response_time
        
        if result.is_valid:
            self.valid_urls += 1
            if result.final_url and result.final_url != result.url:
                self.redirected_urls += 1
        else:
            self.invalid_urls += 1
            
            if result.error_type:
                self.error_distribution[result.error_type] = (
                    self.error_distribution.get(result.error_type, 0) + 1
                )
                
                if 'timeout' in result.error_type.lower():
                    self.timeout_urls += 1
                else:
                    self.error_urls += 1
        
        # Update average response time
        if self.total_urls > 0:
            self.average_response_time = self.total_time / self.total_urls


class URLValidator:
    """Enhanced URL validation with retry logic and intelligent rate limiting"""
    
    # URLs to skip validation (known problematic patterns)
    SKIP_PATTERNS = [
        'javascript:',
        'mailto:',
        'tel:',
        'ftp:',
        'file:',
        'data:',
        '#',  # Fragment-only URLs
        'about:blank'
    ]
    
    # Valid HTTP status codes for success
    SUCCESS_CODES = {200, 201, 202, 203, 204, 205, 206, 300, 301, 302, 303, 304, 307, 308}
    
    def __init__(self, 
                 timeout: float = 30.0,
                 max_redirects: int = 5,
                 max_concurrent: int = 10,
                 user_agent_rotation: bool = True,
                 verify_ssl: bool = True,
                 rate_limiter: Optional[IntelligentRateLimiter] = None):
        """
        Initialize URL validator.
        
        Args:
            timeout: Request timeout in seconds
            max_redirects: Maximum number of redirects to follow
            max_concurrent: Maximum concurrent requests
            user_agent_rotation: Whether to rotate user agents
            verify_ssl: Whether to verify SSL certificates
            rate_limiter: Optional rate limiter instance
        """
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.max_concurrent = max_concurrent
        self.verify_ssl = verify_ssl
        
        # Initialize components
        self.rate_limiter = rate_limiter or IntelligentRateLimiter(max_concurrent=max_concurrent)
        self.browser_simulator = BrowserSimulator(rotate_agents=user_agent_rotation)
        self.retry_handler = RetryHandler()
        self.security_validator = SecurityValidator()
        
        # Create session
        self.session = self._create_session()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = ValidationStats()
        
        logging.info(f"Initialized URL validator (timeout={timeout}s, "
                    f"max_concurrent={max_concurrent}, ssl_verify={verify_ssl})")
    
    def batch_validate_optimized(self,
                                urls: List[str],
                                progress_callback: Optional[Callable[[str], None]] = None,
                                batch_size: int = 100,
                                max_workers: Optional[int] = None) -> List[ValidationResult]:
        """
        Optimized batch validation with memory-efficient processing
        
        Args:
            urls: List of URLs to validate
            progress_callback: Optional progress callback function
            batch_size: Number of URLs to process in each batch
            max_workers: Number of concurrent workers
            
        Returns:
            List of ValidationResult objects
        """
        logging.info(f"Starting optimized batch validation of {len(urls)} URLs in batches of {batch_size}")
        start_time = time.time()
        
        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(urls))
        total_urls = len(unique_urls)
        
        if max_workers is None:
            max_workers = min(self.max_concurrent, 20)  # Cap at 20 for optimization
        
        all_results = []
        processed_count = 0
        
        # Process in batches to manage memory
        for batch_start in range(0, total_urls, batch_size):
            batch_end = min(batch_start + batch_size, total_urls)
            batch_urls = unique_urls[batch_start:batch_end]
            
            logging.info(f"Processing batch {batch_start//batch_size + 1}: {len(batch_urls)} URLs")
            
            # Validate batch with threading
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all validations
                future_to_url = {
                    executor.submit(self.validate_url, url): url 
                    for url in batch_urls
                }
                
                batch_results = []
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        processed_count += 1
                        
                        # Progress callback
                        if progress_callback:
                            progress_callback(f"Validated {processed_count}/{total_urls}: {url}")
                            
                    except Exception as e:
                        logging.error(f"Error validating {url}: {e}")
                        # Create error result
                        error_result = ValidationResult(
                            url=url,
                            is_valid=False,
                            error_message=f"Validation error: {str(e)}",
                            error_type="validation_error"
                        )
                        batch_results.append(error_result)
                        processed_count += 1
            
            # Add batch results to total
            all_results.extend(batch_results)
            
            # Force garbage collection between batches
            import gc
            gc.collect()
            
            # Brief pause between batches to allow rate limiting
            if batch_end < total_urls:
                time.sleep(0.1)
        
        total_time = time.time() - start_time
        valid_count = sum(1 for r in all_results if r.is_valid)
        
        logging.info(f"Batch validation completed: {valid_count}/{len(all_results)} valid URLs in {total_time:.2f}s")
        
        return all_results
    
    def validate_url(self, url: str) -> ValidationResult:
        """
        Validate a single URL.
        
        Args:
            url: URL to validate
            
        Returns:
            ValidationResult object
        """
        start_time = time.time()
        
        # Security validation first
        security_result = self.security_validator.validate_url_security(url)
        if not security_result.is_safe:
            # Log security event
            if security_result.risk_level in ['high', 'critical']:
                secure_logger.log_url_validation_failure(
                    url, 
                    security_result.blocked_reason or "Security validation failed",
                    security_result.issues
                )
            
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=security_result.blocked_reason or "Security validation failed",
                error_type="security_error",
                response_time=time.time() - start_time,
                security_validation=security_result
            )
        
        # Basic URL validation
        if not self._is_valid_url_format(url):
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message="Invalid URL format",
                error_type="format_error",
                response_time=time.time() - start_time,
                security_validation=security_result
            )
        
        # Check skip patterns
        if self._should_skip_url(url):
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message="URL type not supported for validation",
                error_type="unsupported_scheme",
                response_time=time.time() - start_time,
                security_validation=security_result
            )
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed(url)
        
        try:
            # Get headers for this request
            headers = self.browser_simulator.get_headers(url)
            
            # Make request
            response = self.session.head(
                url,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=True,
                verify=self.verify_ssl
            )
            
            response_time = time.time() - start_time
            
            # Check if successful
            is_valid = response.status_code in self.SUCCESS_CODES
            
            result = ValidationResult(
                url=url,
                is_valid=is_valid,
                status_code=response.status_code,
                final_url=response.url if response.url != url else None,
                response_time=response_time,
                content_type=response.headers.get('content-type'),
                content_length=self._parse_content_length(response.headers.get('content-length')),
                security_validation=security_result
            )
            
            if not is_valid:
                result.error_message = f"HTTP {response.status_code}"
                result.error_type = self._classify_http_error(response.status_code)
            
            # Record success for rate limiter
            self.rate_limiter.record_success(url)
            
            return result
            
        except requests.exceptions.SSLError as e:
            # Try without SSL verification if allowed
            if self.verify_ssl:
                logging.debug(f"SSL error for {url}, trying without verification: {e}")
                try:
                    response = self.session.head(
                        url,
                        headers=headers,
                        timeout=self.timeout,
                        allow_redirects=True,
                        verify=False
                    )
                    
                    response_time = time.time() - start_time
                    is_valid = response.status_code in self.SUCCESS_CODES
                    
                    result = ValidationResult(
                        url=url,
                        is_valid=is_valid,
                        status_code=response.status_code,
                        final_url=response.url if response.url != url else None,
                        response_time=response_time,
                        content_type=response.headers.get('content-type'),
                        content_length=self._parse_content_length(response.headers.get('content-length')),
                        error_message="SSL certificate warning" if is_valid else f"HTTP {response.status_code}",
                        error_type="ssl_warning" if is_valid else self._classify_http_error(response.status_code),
                        security_validation=security_result
                    )
                    
                    self.rate_limiter.record_success(url)
                    return result
                    
                except Exception:
                    pass  # Fall through to error handling
            
            self.rate_limiter.record_error(url)
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=f"SSL Error: {str(e)}",
                error_type="ssl_error",
                response_time=time.time() - start_time,
                security_validation=security_result
            )
            
        except requests.exceptions.Timeout as e:
            self.rate_limiter.record_error(url)
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=f"Timeout after {self.timeout}s",
                error_type="timeout",
                response_time=time.time() - start_time,
                security_validation=security_result
            )
            
        except requests.exceptions.ConnectionError as e:
            self.rate_limiter.record_error(url)
            error_msg = str(e)
            error_type = "connection_error"
            
            if "dns" in error_msg.lower() or "name resolution" in error_msg.lower():
                error_type = "dns_error"
            elif "refused" in error_msg.lower():
                error_type = "connection_refused"
            
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=error_msg,
                error_type=error_type,
                response_time=time.time() - start_time,
                security_validation=security_result
            )
            
        except Exception as e:
            self.rate_limiter.record_error(url)
            return ValidationResult(
                url=url,
                is_valid=False,
                error_message=f"Unexpected error: {str(e)}",
                error_type="unknown_error",
                response_time=time.time() - start_time,
                security_validation=security_result
            )
    
    def batch_validate(self, 
                      urls: List[str], 
                      progress_callback: Optional[Callable[[str], None]] = None,
                      enable_retries: bool = True) -> List[ValidationResult]:
        """
        Validate multiple URLs with concurrent processing and retry logic.
        
        Args:
            urls: List of URLs to validate
            progress_callback: Optional progress callback function
            enable_retries: Whether to enable retry logic for failed URLs
            
        Returns:
            List of ValidationResult objects
        """
        logging.info(f"Starting batch validation of {len(urls)} URLs")
        start_time = time.time()
        
        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(urls))
        if len(unique_urls) != len(urls):
            logging.info(f"Removed {len(urls) - len(unique_urls)} duplicate URLs")
        
        results = []
        
        # First pass: concurrent validation
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all validation tasks
            future_to_url = {
                executor.submit(self.validate_url, url): url 
                for url in unique_urls
            }
            
            completed = 0
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update statistics
                    with self.lock:
                        self.stats.update_from_result(result)
                    
                    # Add failed URLs to retry queue
                    if enable_retries and not result.is_valid:
                        if result.error_type not in ['format_error', 'unsupported_scheme']:
                            error = Exception(result.error_message or "Validation failed")
                            self.retry_handler.add_failed_url(url, error)
                    
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(f"Validated {completed}/{len(unique_urls)}: {url}")
                    
                except Exception as e:
                    logging.error(f"Error validating {url}: {e}")
                    error_result = ValidationResult(
                        url=url,
                        is_valid=False,
                        error_message=str(e),
                        error_type="validation_error"
                    )
                    results.append(error_result)
                    
                    if enable_retries:
                        self.retry_handler.add_failed_url(url, e)
        
        # Second pass: retry failed URLs
        if enable_retries and self.retry_handler.retry_queue:
            logging.info(f"Retrying {len(self.retry_handler.retry_queue)} failed URLs")
            
            retry_results = self.retry_handler.retry_failed_items(
                self.validate_url,
                progress_callback
            )
            
            # Update results with successful retries
            retry_urls = {r.url for r in retry_results}
            results = [r for r in results if r.url not in retry_urls]
            results.extend(retry_results)
        
        total_time = time.time() - start_time
        
        # Final statistics
        valid_count = sum(1 for r in results if r.is_valid)
        logging.info(f"Batch validation completed in {total_time:.2f}s: "
                    f"{valid_count}/{len(results)} URLs valid "
                    f"({valid_count/len(results)*100:.1f}%)")
        
        return results
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics"""
        with self.lock:
            stats_dict = {
                'total_urls': self.stats.total_urls,
                'valid_urls': self.stats.valid_urls,
                'invalid_urls': self.stats.invalid_urls,
                'success_rate': self.stats.valid_urls / max(self.stats.total_urls, 1),
                'average_response_time': self.stats.average_response_time,
                'total_processing_time': self.stats.total_time,
                'redirected_urls': self.stats.redirected_urls,
                'error_distribution': dict(self.stats.error_distribution),
                'rate_limiter_stats': self.rate_limiter.get_all_stats(),
            }
            
            if self.retry_handler:
                stats_dict['retry_stats'] = self.retry_handler.get_retry_statistics()
            
            return stats_dict
    
    def _create_session(self) -> requests.Session:
        """Create configured requests session"""
        session = requests.Session()
        
        # Configure adapters for retries at the HTTP level
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=0,  # We handle retries at application level
            connect=1,  # Allow one connection retry
            read=1,     # Allow one read retry
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        
        # Enhanced adapter with larger connection pool
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,    # Number of connection pools
            pool_maxsize=50,        # Max connections per pool
            pool_block=False        # Don't block when pool is full
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default timeout
        session.timeout = self.timeout
        
        return session
    
    def _is_valid_url_format(self, url: str) -> bool:
        """Check if URL has valid format"""
        if not url or not isinstance(url, str):
            return False
        
        url = url.strip()
        if not url:
            return False
        
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False
    
    def _should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped"""
        url_lower = url.lower().strip()
        
        for pattern in self.SKIP_PATTERNS:
            if url_lower.startswith(pattern):
                return True
        
        return False
    
    def _classify_http_error(self, status_code: int) -> str:
        """Classify HTTP error by status code"""
        if 400 <= status_code < 500:
            error_types = {
                400: "bad_request",
                401: "unauthorized",
                403: "forbidden",
                404: "not_found",
                405: "method_not_allowed",
                408: "request_timeout",
                429: "rate_limited",
            }
            return error_types.get(status_code, "client_error")
        elif 500 <= status_code < 600:
            error_types = {
                500: "internal_server_error",
                501: "not_implemented",
                502: "bad_gateway",
                503: "service_unavailable",
                504: "gateway_timeout",
            }
            return error_types.get(status_code, "server_error")
        else:
            return "http_error"
    
    def _parse_content_length(self, content_length_header: Optional[str]) -> Optional[int]:
        """Parse content-length header"""
        if not content_length_header:
            return None
        
        try:
            return int(content_length_header)
        except (ValueError, TypeError):
            return None
    
    def reset_statistics(self) -> None:
        """Reset validation statistics"""
        with self.lock:
            self.stats = ValidationStats()
            self.rate_limiter.reset_domain_stats()
            if self.retry_handler:
                self.retry_handler.clear_completed()
        
        logging.info("Reset validation statistics")
    
    def close(self) -> None:
        """Clean up resources"""
        if hasattr(self, 'session'):
            self.session.close()
        
        logging.info("URL validator closed")