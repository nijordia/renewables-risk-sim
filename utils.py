"""
Renewables Risk Simulator - Utility Functions

Shared utilities for API requests, date handling, and data processing.
"""

import json
import time
from datetime import datetime, timedelta

import requests


# REData API has a limit on date ranges, so we chunk requests
MAX_DAYS_PER_REQUEST = 5


def fetch_with_retry(url: str, max_retries: int = 3, timeout: int = 30) -> dict:
    """
    Fetch JSON from URL with retry logic and error handling.

    Args:
        url: API endpoint URL
        max_retries: Number of retry attempts
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response as dict

    Raises:
        requests.RequestException: If all retries fail
    """
    headers = {
        "Accept": "application/json",
        "User-Agent": "renewables-risk-sim/1.0"
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"  HTTP error (attempt {attempt + 1}/{max_retries}): {e}")
            last_error = e
        except requests.exceptions.ConnectionError as e:
            print(f"  Connection error (attempt {attempt + 1}/{max_retries}): {e}")
            last_error = e
        except requests.exceptions.Timeout as e:
            print(f"  Timeout (attempt {attempt + 1}/{max_retries}): {e}")
            last_error = e
        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt + 1}/{max_retries}): {e}")
            last_error = e

        # Wait before retry (exponential backoff)
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            time.sleep(wait_time)

    raise requests.RequestException(f"Failed after {max_retries} attempts: {last_error}")


def date_range_chunks(start_date: str, end_date: str, chunk_days: int = MAX_DAYS_PER_REQUEST):
    """
    Split a date range into smaller chunks for API requests.

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        chunk_days: Maximum days per chunk

    Yields:
        Tuples of (chunk_start, chunk_end) as datetime strings
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        # Format with timezone for API compatibility
        yield (
            current.strftime("%Y-%m-%dT00:00"),
            chunk_end.strftime("%Y-%m-%dT23:59")
        )
        current = chunk_end + timedelta(days=1)


def get_season(month: int) -> str:
    """Return season name for a given month (Northern Hemisphere)."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


def validate_dates(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    """
    Validate date strings and return datetime objects.

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        Tuple of (start_datetime, end_datetime)

    Raises:
        ValueError: If dates are invalid or start >= end
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    if start >= end:
        raise ValueError("Start date must be before end date")
    return start, end


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
