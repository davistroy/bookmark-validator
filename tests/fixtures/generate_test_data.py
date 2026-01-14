"""
Generate realistic test data for performance testing.

This module creates test CSV files with varying sizes to simulate
different performance scenarios for the bookmark processor.
"""

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


class TestDataGenerator:
    """Generates realistic raindrop.io export data for testing."""

    # Sample domains and URL patterns
    DOMAINS = [
        "github.com", "stackoverflow.com", "medium.com", "dev.to",
        "python.org", "docs.python.org", "npmjs.com", "go.dev",
        "rust-lang.org", "example.com", "wikipedia.org", "reddit.com",
        "news.ycombinator.com", "aws.amazon.com", "cloud.google.com",
        "microsoft.com", "mozilla.org", "w3.org", "ietf.org", "arxiv.org"
    ]

    # Sample folders
    FOLDERS = [
        "Programming/Python",
        "Programming/JavaScript",
        "Programming/Go",
        "Programming/Rust",
        "Development/Tools",
        "Development/DevOps",
        "Documentation",
        "Tutorials",
        "Articles",
        "Resources",
        "Web Development",
        "Data Science",
        "Machine Learning",
        "Cloud/AWS",
        "Cloud/GCP",
        "Cloud/Azure",
        "Security",
        "Performance",
        "Testing",
        "Uncategorized"
    ]

    # Sample tags
    TAG_SETS = [
        ["python", "programming"],
        ["javascript", "web", "frontend"],
        ["backend", "api", "rest"],
        ["devops", "docker", "kubernetes"],
        ["aws", "cloud", "infrastructure"],
        ["tutorial", "learning"],
        ["documentation", "reference"],
        ["article", "blog"],
        ["tool", "utility"],
        ["security", "authentication"],
        ["database", "sql", "nosql"],
        ["testing", "qa", "automation"],
        ["performance", "optimization"],
        ["machine-learning", "ai", "data-science"],
        ["git", "version-control"]
    ]

    # Sample titles and notes
    TITLE_TEMPLATES = [
        "Introduction to {topic}",
        "Advanced {topic} Techniques",
        "Best Practices for {topic}",
        "Understanding {topic}",
        "Complete Guide to {topic}",
        "{topic} Tutorial",
        "{topic} Documentation",
        "Getting Started with {topic}",
        "{topic} - Tips and Tricks",
        "Mastering {topic}"
    ]

    TOPICS = [
        "Python", "JavaScript", "Docker", "Kubernetes", "AWS", "React",
        "Vue.js", "Django", "FastAPI", "Node.js", "PostgreSQL", "MongoDB",
        "Redis", "GraphQL", "REST APIs", "Microservices", "CI/CD",
        "Testing", "Security", "Performance Optimization"
    ]

    NOTE_TEMPLATES = [
        "Comprehensive guide covering {aspect}",
        "Essential resource for understanding {aspect}",
        "Step-by-step tutorial on {aspect}",
        "Technical documentation about {aspect}",
        "Best practices and patterns for {aspect}",
        "Quick reference for {aspect}",
        "In-depth article discussing {aspect}",
        "",  # Some bookmarks have no notes
        "",
        "Useful resource for {aspect}"
    ]

    ASPECTS = [
        "core concepts", "implementation details", "common patterns",
        "performance tuning", "security considerations", "deployment strategies",
        "debugging techniques", "integration approaches", "architecture design",
        "testing methodologies"
    ]

    EXCERPT_TEMPLATES = [
        "Learn about {topic} with practical examples and real-world applications.",
        "Comprehensive guide to {topic} covering best practices and common pitfalls.",
        "{topic} tutorial with step-by-step instructions and code examples.",
        "Official documentation for {topic} including API reference and guides.",
        "In-depth article exploring {topic} and its ecosystem.",
        "Practical tips and techniques for working with {topic}.",
        "Everything you need to know about {topic} in one place.",
        "",  # Some bookmarks have no excerpts
        "Quick reference guide for {topic}."
    ]

    def __init__(self, seed: int = 42):
        """Initialize the test data generator with a random seed."""
        random.seed(seed)
        self.base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def generate_url(self, index: int) -> str:
        """Generate a realistic URL."""
        domain = random.choice(self.DOMAINS)

        # Vary the URL patterns
        patterns = [
            f"https://{domain}/article/{index}",
            f"https://{domain}/docs/page-{index}",
            f"https://{domain}/blog/post-{index}",
            f"https://{domain}/guide/{index}",
            f"https://{domain}/tutorial/{index}",
            f"https://{domain}/resource/{index}",
            f"https://{domain}/page/{index}",
        ]

        return random.choice(patterns)

    def generate_title(self) -> str:
        """Generate a realistic bookmark title."""
        template = random.choice(self.TITLE_TEMPLATES)
        topic = random.choice(self.TOPICS)
        return template.format(topic=topic)

    def generate_note(self) -> str:
        """Generate a realistic note."""
        template = random.choice(self.NOTE_TEMPLATES)
        aspect = random.choice(self.ASPECTS)
        return template.format(aspect=aspect)

    def generate_excerpt(self) -> str:
        """Generate a realistic excerpt."""
        template = random.choice(self.EXCERPT_TEMPLATES)
        topic = random.choice(self.TOPICS)
        return template.format(topic=topic)

    def generate_tags(self) -> str:
        """Generate a realistic tag set."""
        tags = random.choice(self.TAG_SETS)
        # Sometimes add an extra random tag
        if random.random() > 0.7:
            all_tags = [tag for tag_set in self.TAG_SETS for tag in tag_set]
            extra_tag = random.choice(all_tags)
            if extra_tag not in tags:
                tags.append(extra_tag)
        return ", ".join(tags)

    def generate_folder(self) -> str:
        """Generate a realistic folder path."""
        return random.choice(self.FOLDERS)

    def generate_created_date(self, index: int, total: int) -> str:
        """Generate a created date (spread over time)."""
        # Spread bookmarks over the past year
        days_offset = (index / total) * 365
        date = self.base_date - timedelta(days=days_offset)
        # Add some random variation (within a day)
        hours_variation = random.randint(0, 23)
        minutes_variation = random.randint(0, 59)
        date = date + timedelta(hours=hours_variation, minutes=minutes_variation)
        return date.isoformat().replace("+00:00", "Z")

    def generate_bookmark(self, index: int, total: int, include_invalid: bool = True) -> Dict[str, str]:
        """Generate a single bookmark entry."""
        # Occasionally create invalid bookmarks (2% of the time if enabled)
        if include_invalid and random.random() < 0.02:
            return self._generate_invalid_bookmark(index)

        return {
            "id": str(index),
            "title": self.generate_title(),
            "note": self.generate_note(),
            "excerpt": self.generate_excerpt(),
            "url": self.generate_url(index),
            "folder": self.generate_folder(),
            "tags": self.generate_tags(),
            "created": self.generate_created_date(index, total),
            "cover": "",
            "highlights": "",
            "favorite": "true" if random.random() > 0.9 else "false"
        }

    def _generate_invalid_bookmark(self, index: int) -> Dict[str, str]:
        """Generate an intentionally invalid bookmark for testing error handling."""
        invalid_types = [
            {"url": "not-a-valid-url", "title": "Invalid URL Example"},
            {"url": "", "title": "Empty URL Example"},
            {"url": "javascript:void(0)", "title": "JavaScript URL"},
            {"url": "mailto:test@example.com", "title": "Email Link"},
        ]

        invalid = random.choice(invalid_types)

        return {
            "id": str(index),
            "title": invalid["title"],
            "note": "This bookmark is intentionally invalid for testing",
            "excerpt": "",
            "url": invalid["url"],
            "folder": "Test/Invalid",
            "tags": "test, invalid",
            "created": self.generate_created_date(index, 100),
            "cover": "",
            "highlights": "",
            "favorite": "false"
        }

    def generate_dataset(
        self,
        size: int,
        include_invalid: bool = True,
        duplicate_rate: float = 0.0
    ) -> List[Dict[str, str]]:
        """
        Generate a dataset of bookmarks.

        Args:
            size: Number of bookmarks to generate
            include_invalid: Whether to include invalid bookmarks
            duplicate_rate: Percentage of duplicates (0.0 to 1.0)

        Returns:
            List of bookmark dictionaries
        """
        bookmarks = []

        for i in range(1, size + 1):
            # Occasionally create duplicates
            if duplicate_rate > 0 and random.random() < duplicate_rate and bookmarks:
                # Duplicate a random existing bookmark
                duplicate = bookmarks[random.randint(0, len(bookmarks) - 1)].copy()
                duplicate["id"] = str(i)
                bookmarks.append(duplicate)
            else:
                bookmarks.append(self.generate_bookmark(i, size, include_invalid))

        return bookmarks

    def save_to_csv(
        self,
        bookmarks: List[Dict[str, str]],
        output_path: Path
    ) -> None:
        """Save bookmarks to a CSV file in raindrop.io export format."""
        df = pd.DataFrame(bookmarks)

        # Ensure correct column order (11-column raindrop export format)
        columns = [
            "id", "title", "note", "excerpt", "url", "folder",
            "tags", "created", "cover", "highlights", "favorite"
        ]
        df = df[columns]

        # Save to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated {len(bookmarks)} bookmarks to {output_path}")

    def generate_performance_suite(self, output_dir: Path) -> Dict[str, Path]:
        """
        Generate a complete suite of test files for performance testing.

        Returns:
            Dictionary mapping test names to file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        test_files = {}

        # Small test (100 bookmarks) - quick sanity check
        small_bookmarks = self.generate_dataset(100, include_invalid=True)
        small_path = output_dir / "performance_test_100.csv"
        self.save_to_csv(small_bookmarks, small_path)
        test_files["small"] = small_path

        # Medium test (1000 bookmarks) - medium load
        medium_bookmarks = self.generate_dataset(1000, include_invalid=True)
        medium_path = output_dir / "performance_test_1000.csv"
        self.save_to_csv(medium_bookmarks, medium_path)
        test_files["medium"] = medium_path

        # Large test (3500 bookmarks) - full load simulation
        large_bookmarks = self.generate_dataset(3500, include_invalid=True)
        large_path = output_dir / "performance_test_3500.csv"
        self.save_to_csv(large_bookmarks, large_path)
        test_files["large"] = large_path

        # Duplicate test (500 bookmarks with 10% duplicates)
        duplicate_bookmarks = self.generate_dataset(
            500, include_invalid=True, duplicate_rate=0.1
        )
        duplicate_path = output_dir / "performance_test_duplicates.csv"
        self.save_to_csv(duplicate_bookmarks, duplicate_path)
        test_files["duplicates"] = duplicate_path

        # Clean test (100 bookmarks, all valid)
        clean_bookmarks = self.generate_dataset(100, include_invalid=False)
        clean_path = output_dir / "performance_test_clean.csv"
        self.save_to_csv(clean_bookmarks, clean_path)
        test_files["clean"] = clean_path

        return test_files


def main():
    """Generate test data when run as a script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate test data for bookmark processor performance testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "performance_data",
        help="Output directory for test files"
    )
    parser.add_argument(
        "--size",
        type=int,
        help="Generate a single file with specified size"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (for single file generation)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Generate complete performance test suite"
    )

    args = parser.parse_args()

    generator = TestDataGenerator(seed=args.seed)

    if args.suite:
        # Generate complete suite
        test_files = generator.generate_performance_suite(args.output_dir)
        print("\nGenerated performance test suite:")
        for name, path in test_files.items():
            print(f"  {name}: {path}")
    elif args.size and args.output:
        # Generate single file
        bookmarks = generator.generate_dataset(args.size)
        generator.save_to_csv(bookmarks, args.output)
    else:
        # Default: generate suite
        test_files = generator.generate_performance_suite(args.output_dir)
        print("\nGenerated performance test suite:")
        for name, path in test_files.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
