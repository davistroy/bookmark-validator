"""
Multi-file processing module for batch bookmark operations.

This module provides functionality to automatically detect and process
multiple bookmark files in a single operation.
"""

import glob
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .data_models import Bookmark
from .import_module import MultiFormatImporter
from ..utils.error_handler import BookmarkImportError, UnsupportedFormatError


class MultiFileProcessor:
    """
    Processor for handling multiple bookmark files in batch operations.

    Supports auto-detection of bookmark files in a directory and
    processing them in a unified workflow.
    """

    def __init__(self):
        """Initialize the multi-file processor."""
        self.logger = logging.getLogger(__name__)
        self.importer = MultiFormatImporter()

    def auto_detect_files(self, directory: Optional[Union[str, Path]] = None) -> List[Path]:
        """
        Auto-detect bookmark files in a directory.

        Args:
            directory: Directory to search. Defaults to current directory.

        Returns:
            List of detected bookmark file paths

        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        if directory is None:
            directory = Path.cwd()
        else:
            directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        self.logger.info(f"Auto-detecting bookmark files in: {directory}")

        # Find all potential bookmark files
        potential_files = []

        # Search for CSV files
        csv_files = glob.glob(str(directory / "*.csv"))
        potential_files.extend(csv_files)

        # Search for HTML files
        html_files = glob.glob(str(directory / "*.html"))
        html_files.extend(glob.glob(str(directory / "*.htm")))
        potential_files.extend(html_files)

        # Validate each file
        valid_files = []
        for file_path in potential_files:
            try:
                file_info = self.importer.get_file_info(file_path)
                if file_info["is_supported"] and file_info["estimated_bookmarks"] > 0:
                    valid_files.append(Path(file_path))
                    self.logger.info(
                        f"Detected valid bookmark file: {file_path} "
                        f"({file_info['format']}, "
                        f"~{file_info['estimated_bookmarks']} bookmarks)"
                    )
                else:
                    self.logger.debug(f"Skipping unsupported file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Error validating file {file_path}: {e}")

        self.logger.info(f"Found {len(valid_files)} valid bookmark files")
        return sorted(valid_files)  # Sort for consistent processing order

    def process_multiple_files(
        self, file_paths: List[Union[str, Path]], merge_strategy: str = "combine"
    ) -> Tuple[List[Bookmark], Dict[str, Any]]:
        """
        Process multiple bookmark files.

        Args:
            file_paths: List of file paths to process
            merge_strategy: How to handle multiple files ("combine", "separate")

        Returns:
            Tuple of (combined bookmarks, processing statistics)

        Raises:
            BookmarkImportError: If any critical errors occur during processing
        """
        if not file_paths:
            raise ValueError("No files provided for processing")

        self.logger.info(f"Processing {len(file_paths)} bookmark files")

        all_bookmarks = []
        processing_stats = {
            "total_files": len(file_paths),
            "successful_files": 0,
            "failed_files": 0,
            "total_bookmarks": 0,
            "file_results": {},
            "errors": [],
        }

        for file_path in file_paths:
            file_path = Path(file_path)

            try:
                self.logger.info(f"Processing file: {file_path}")

                # Import bookmarks from this file
                bookmarks = self.importer.import_bookmarks(file_path)

                # Add source file information to bookmarks
                for bookmark in bookmarks:
                    if not hasattr(bookmark, "source_file"):
                        bookmark.source_file = str(file_path)

                all_bookmarks.extend(bookmarks)

                # Record success
                processing_stats["successful_files"] += 1
                processing_stats["file_results"][str(file_path)] = {
                    "status": "success",
                    "bookmark_count": len(bookmarks),
                    "file_format": self.importer.detect_format(file_path),
                }

                self.logger.info(
                    f"Successfully processed {len(bookmarks)} bookmarks "
                    f"from {file_path}"
                )

            except (BookmarkImportError, UnsupportedFormatError) as e:
                error_msg = f"Failed to process {file_path}: {str(e)}"
                self.logger.error(error_msg)

                processing_stats["failed_files"] += 1
                processing_stats["errors"].append(error_msg)
                processing_stats["file_results"][str(file_path)] = {
                    "status": "failed",
                    "error": str(e),
                    "bookmark_count": 0,
                }

                # Continue processing other files instead of failing completely
                continue

            except Exception as e:
                error_msg = f"Unexpected error processing {file_path}: {str(e)}"
                self.logger.exception(error_msg)

                processing_stats["failed_files"] += 1
                processing_stats["errors"].append(error_msg)
                processing_stats["file_results"][str(file_path)] = {
                    "status": "failed",
                    "error": str(e),
                    "bookmark_count": 0,
                }

                # Continue processing other files
                continue

        processing_stats["total_bookmarks"] = len(all_bookmarks)

        if processing_stats["successful_files"] == 0:
            raise BookmarkImportError("Failed to process any files successfully")

        self.logger.info(
            f"Multi-file processing completed: "
            f"{processing_stats['successful_files']}/"
            f"{processing_stats['total_files']} files successful, "
            f"{processing_stats['total_bookmarks']} total bookmarks"
        )

        return all_bookmarks, processing_stats

    def generate_timestamped_output_paths(
        self, base_name: str = "bookmarks"
    ) -> Dict[str, Path]:
        """
        Generate timestamped output file paths for both CSV and HTML formats.

        Args:
            base_name: Base name for output files

        Returns:
            Dictionary with 'csv' and 'html' keys containing Path objects
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return {
            "csv": Path(f"{base_name}_{timestamp}.csv"),
            "html": Path(f"{base_name}_{timestamp}.html"),
        }

    def get_processing_summary(self, stats: Dict[str, Any]) -> str:
        """
        Generate a human-readable processing summary.

        Args:
            stats: Processing statistics from process_multiple_files

        Returns:
            Formatted summary string
        """
        summary_lines = [
            "Multi-file Processing Summary:",
            f"  Files processed: {stats['successful_files']}/{stats['total_files']}",
            f"  Total bookmarks: {stats['total_bookmarks']}",
        ]

        if stats["errors"]:
            summary_lines.append(f"  Errors: {len(stats['errors'])}")

        summary_lines.append("\nFile Details:")
        for file_path, result in stats["file_results"].items():
            status_icon = "✓" if result["status"] == "success" else "✗"
            if result["status"] == "success":
                summary_lines.append(
                    f"  {status_icon} {Path(file_path).name}: "
                    f"{result['bookmark_count']} bookmarks "
                    f"({result['file_format']})"
                )
            else:
                summary_lines.append(
                    f"  {status_icon} {Path(file_path).name}: {result['error']}"
                )

        return "\n".join(summary_lines)

    def validate_directory_for_auto_detection(
        self, directory: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Validate a directory for auto-detection capabilities.

        Args:
            directory: Directory to validate. Defaults to current directory.

        Returns:
            Validation report dictionary
        """
        if directory is None:
            directory = Path.cwd()
        else:
            directory = Path(directory)

        report = {
            "directory": str(directory),
            "exists": directory.exists(),
            "is_directory": directory.is_dir() if directory.exists() else False,
            "readable": os.access(directory, os.R_OK) if directory.exists() else False,
            "potential_files": [],
            "valid_files": [],
            "invalid_files": [],
            "total_estimated_bookmarks": 0,
            "can_auto_detect": False,
        }

        if not report["exists"]:
            report["error"] = "Directory does not exist"
            return report

        if not report["is_directory"]:
            report["error"] = "Path is not a directory"
            return report

        if not report["readable"]:
            report["error"] = "Directory is not readable"
            return report

        try:
            # Find potential files
            potential_files = []
            potential_files.extend(glob.glob(str(directory / "*.csv")))
            potential_files.extend(glob.glob(str(directory / "*.html")))
            potential_files.extend(glob.glob(str(directory / "*.htm")))

            report["potential_files"] = [str(Path(f).name) for f in potential_files]

            # Validate each file
            for file_path in potential_files:
                try:
                    file_info = self.importer.get_file_info(file_path)
                    if file_info["is_supported"]:
                        report["valid_files"].append(
                            {
                                "name": Path(file_path).name,
                                "format": file_info["format"],
                                "estimated_bookmarks": file_info["estimated_bookmarks"],
                                "size_mb": file_info["size_bytes"] / 1024 / 1024,
                            }
                        )
                        report["total_estimated_bookmarks"] += file_info[
                            "estimated_bookmarks"
                        ]
                    else:
                        report["invalid_files"].append(
                            {
                                "name": Path(file_path).name,
                                "reason": "Unsupported format or invalid content",
                            }
                        )
                except Exception as e:
                    report["invalid_files"].append(
                        {"name": Path(file_path).name, "reason": str(e)}
                    )

            report["can_auto_detect"] = len(report["valid_files"]) > 0

        except Exception as e:
            report["error"] = f"Error scanning directory: {str(e)}"

        return report
