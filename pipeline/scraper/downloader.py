import logging
import time
from pathlib import Path
import hashlib
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from pipeline.config import settings

logger = logging.getLogger(__name__)

class DownloadError(Exception):
    """Base exception for download errors."""
    pass

class HTTPStatusError(DownloadError):
    """Exception raised when an HTTP request returns an error status code."""
    def __init__(self, status_code: int, message: str):
        super().__init__(f"HTTP status {status_code}: {message}")
        self.status_code = status_code

class PDFDownloader:
    """Handles downloading PDF files with retry logic and validation."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": settings.scraper_user_agent
        })
        self.timeout = settings.scraper_timeout

    def _calculate_sha256(self, file_path: Path) -> str:
        """Calculates the SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.scraper_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, HTTPStatusError)),
    )
    def _download_with_retry(self, url: str, dest_path: Path) -> tuple[int, str]:
        """Performs the actual download with retry rules."""
        logger.debug(f"Requesting URL: {url}")
        
        # We need to make a request and download the file stream.
        # Note: download_file.php might return 302 or direct download. requests handles redirects automatically.
        try:
            response = self.session.get(url, stream=True, timeout=self.timeout)
            
            # Raise exception on HTTP errors
            if response.status_code == 429:
                logger.warning("HTTP 429 Rate Limit hit. Backing off...")
                raise HTTPStatusError(response.status_code, "Rate limit hit")
            elif 500 <= response.status_code < 600:
                logger.warning(f"HTTP {response.status_code} server error. Retrying...")
                raise HTTPStatusError(response.status_code, "Server error")
            elif response.status_code != 200:
                # E.g. 404. Don't retry on 404 because it's not going to change.
                # However, tenacity will retry on HTTPStatusError by default because of retry_if_exception_type.
                # Let's check: should we retry on 404? No. So let's handle 404 differently.
                if response.status_code == 404:
                    raise DownloadError(f"HTTP 404: File not found at {url}")
                response.raise_for_status()

            # Create destination parent directories if they don't exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Temporary file to download content to prevent incomplete files on disk
            temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
            
            total_bytes = 0
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_bytes += len(chunk)

            # Check content length if available
            content_length = response.headers.get("Content-Length")
            if content_length:
                expected_bytes = int(content_length)
                if total_bytes != expected_bytes:
                    temp_path.unlink(missing_ok=True)
                    raise DownloadError(
                        f"Size mismatch: downloaded {total_bytes} bytes, expected {expected_bytes} bytes."
                    )
            
            # Rename temp file to final destination
            temp_path.replace(dest_path)
            
            # Calculate SHA-256
            sha256_hash = self._calculate_sha256(dest_path)
            return total_bytes, sha256_hash

        except requests.RequestException as e:
            logger.warning(f"Network error during download: {e}. Retrying...")
            raise

    def download(self, url: str, dest_path: Path, force: bool = False) -> tuple[Path, int, str]:
        """
        Downloads a PDF from url to dest_path.
        Returns a tuple of (Path, size_bytes, sha256).
        """
        if dest_path.exists() and not force:
            logger.info(f"File already exists, skipping: {dest_path.name}")
            size_bytes = dest_path.stat().st_size
            sha256_hash = self._calculate_sha256(dest_path)
            return dest_path, size_bytes, sha256_hash

        # Implement scraper rate limit delay
        if settings.scraper_rate_limit_delay > 0:
            time.sleep(settings.scraper_rate_limit_delay)

        logger.info(f"Downloading: {url} -> {dest_path}")
        size_bytes, sha256_hash = self._download_with_retry(url, dest_path)
        return dest_path, size_bytes, sha256_hash
