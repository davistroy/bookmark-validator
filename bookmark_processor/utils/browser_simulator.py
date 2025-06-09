"""
Browser Simulator Module

Provides realistic browser headers and user agent strings to avoid detection
as automated scraping while remaining ethical and respectful.
"""

import logging
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class BrowserProfile:
    """Browser profile with headers and capabilities"""

    user_agent: str
    accept: str
    accept_language: str
    accept_encoding: str
    sec_fetch_dest: str = "document"
    sec_fetch_mode: str = "navigate"
    sec_fetch_site: str = "none"
    sec_fetch_user: str = "?1"


class BrowserSimulator:
    """Simulate realistic browser behavior and headers"""

    # Current popular user agents (updated periodically)
    USER_AGENTS = [
        # Chrome on Windows
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
         "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
         "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"),
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
         "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"),
        # Firefox on Windows
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) "
         "Gecko/20100101 Firefox/120.0"),
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) "
         "Gecko/20100101 Firefox/119.0"),
        # Edge on Windows
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
         "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"),
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
         "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"),
        # Chrome on macOS
        ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
        ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"),
        # Safari on macOS
        ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
         "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"),
        ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
         "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"),
    ]

    # Accept headers for different content types
    ACCEPT_HEADERS = [
        ("text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
         "image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"),
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    ]

    # Language preferences
    ACCEPT_LANGUAGES = [
        "en-US,en;q=0.9",
        "en-US,en;q=0.8",
        "en-US,en;q=0.5",
        "en-GB,en;q=0.9,en-US;q=0.8",
    ]

    # Encoding preferences
    ACCEPT_ENCODINGS = [
        "gzip, deflate, br",
        "gzip, deflate",
        "identity",
    ]

    def __init__(self, rotate_agents: bool = True, session_persistence: bool = True):
        """
        Initialize browser simulator.

        Args:
            rotate_agents: Whether to rotate user agents
            session_persistence: Whether to maintain same headers per session
        """
        self.rotate_agents = rotate_agents
        self.session_persistence = session_persistence
        self.current_profile: Optional[BrowserProfile] = None
        self.session_start_time = time.time()

        # Load user agents from file if available
        self._load_user_agents_from_file()

        logging.info(
            f"Initialized browser simulator with {len(self.USER_AGENTS)} user agents"
        )

    def get_headers(self, url: str = None) -> Dict[str, str]:
        """
        Get realistic browser headers.

        Args:
            url: URL being requested (for context-specific headers)

        Returns:
            Dictionary of HTTP headers
        """
        profile = self._get_current_profile()

        headers = {
            "User-Agent": profile.user_agent,
            "Accept": profile.accept,
            "Accept-Language": profile.accept_language,
            "Accept-Encoding": profile.accept_encoding,
            "DNT": "1",  # Do Not Track
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": profile.sec_fetch_dest,
            "Sec-Fetch-Mode": profile.sec_fetch_mode,
            "Sec-Fetch-Site": profile.sec_fetch_site,
            "Sec-Fetch-User": profile.sec_fetch_user,
            "Cache-Control": "max-age=0",
        }

        # Add referer for non-initial requests
        if url and random.random() < 0.3:  # 30% chance of having referer
            headers["Referer"] = self._generate_realistic_referer(url)

        return headers

    def get_random_user_agent(self) -> str:
        """Get a random user agent string"""
        return random.choice(self.USER_AGENTS)

    def rotate_profile(self) -> None:
        """Force rotation to a new browser profile"""
        self.current_profile = None
        self.session_start_time = time.time()

    def _get_current_profile(self) -> BrowserProfile:
        """Get current browser profile, creating new one if needed"""
        if not self.session_persistence or self.current_profile is None:
            self.current_profile = self._create_browser_profile()
        elif self.rotate_agents and self._should_rotate_profile():
            self.current_profile = self._create_browser_profile()

        return self.current_profile

    def _create_browser_profile(self) -> BrowserProfile:
        """Create a new realistic browser profile"""
        user_agent = random.choice(self.USER_AGENTS)

        # Select headers that match the browser type
        if "Firefox" in user_agent:
            accept = ("text/html,application/xhtml+xml,application/xml;q=0.9,"
                      "image/avif,image/webp,*/*;q=0.8")
            sec_fetch_dest = "document"
        elif "Safari" in user_agent and "Chrome" not in user_agent:
            accept = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            sec_fetch_dest = "document"
        else:  # Chrome-based browsers
            accept = random.choice(self.ACCEPT_HEADERS)
            sec_fetch_dest = "document"

        profile = BrowserProfile(
            user_agent=user_agent,
            accept=accept,
            accept_language=random.choice(self.ACCEPT_LANGUAGES),
            accept_encoding=random.choice(self.ACCEPT_ENCODINGS),
            sec_fetch_dest=sec_fetch_dest,
            sec_fetch_mode="navigate",
            sec_fetch_site="none",
            sec_fetch_user="?1",
        )

        logging.debug(f"Created new browser profile: {user_agent[:50]}...")
        return profile

    def _should_rotate_profile(self) -> bool:
        """Determine if profile should be rotated"""
        # Rotate every 10-30 minutes
        session_duration = time.time() - self.session_start_time
        rotation_interval = random.uniform(600, 1800)  # 10-30 minutes

        return session_duration > rotation_interval

    def _generate_realistic_referer(self, url: str) -> str:
        """Generate a realistic referer URL"""
        common_referers = [
            "https://www.google.com/",
            "https://duckduckgo.com/",
            "https://www.bing.com/",
            "https://github.com/",
            "https://stackoverflow.com/",
            "https://news.ycombinator.com/",
        ]

        # Sometimes use actual domain as referer
        if random.random() < 0.7:  # 70% chance
            return random.choice(common_referers)
        else:
            # Use same domain
            from urllib.parse import urlparse

            try:
                parsed = urlparse(url)
                return f"{parsed.scheme}://{parsed.netloc}/"
            except Exception:
                return "https://www.google.com/"

    def _load_user_agents_from_file(self) -> None:
        """Load additional user agents from file if available"""
        try:
            # Try to load from data directory
            import sys
            from pathlib import Path

            if getattr(sys, "frozen", False):
                # Running as executable
                base_path = Path(sys.executable).parent
            else:
                # Running as script
                base_path = Path(__file__).parent.parent

            user_agents_file = base_path / "data" / "user_agents.txt"

            if user_agents_file.exists():
                with open(user_agents_file, "r", encoding="utf-8") as f:
                    additional_agents = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ]

                if additional_agents:
                    self.USER_AGENTS.extend(additional_agents)
                    logging.info(
                        f"Loaded {len(additional_agents)} additional user agents "
                        f"from file"
                    )

        except Exception as e:
            logging.debug(f"Could not load user agents from file: {e}")

    def get_profile_info(self) -> Dict[str, str]:
        """Get current profile information for debugging"""
        if self.current_profile:
            return {
                "user_agent": self.current_profile.user_agent,
                "accept": self.current_profile.accept,
                "accept_language": self.current_profile.accept_language,
                "session_age": str(time.time() - self.session_start_time),
            }
        return {"status": "No profile created yet"}


# Convenience function for getting headers
def get_random_headers(url: str = None) -> Dict[str, str]:
    """Get random browser headers - convenience function"""
    simulator = BrowserSimulator(rotate_agents=True, session_persistence=False)
    return simulator.get_headers(url)
