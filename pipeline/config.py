"""
pipeline/config.py

Pipeline configuration — shared settings for all pipeline stages.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class PipelineSettings(BaseSettings):
    """
    Global pipeline settings.
    Loaded from environment variables with EDGENIE_ prefix.
    """

    # ── Storage ──
    storage_dir: Path = Path("pipeline/storage")
    syllabi_dir: Path = Path("syllabi")

    # ── Scraper ──
    scraper_rate_limit_delay: float = 2.0  # seconds between HTTP requests
    scraper_max_retries: int = 5
    scraper_timeout: int = 30  # seconds per request
    scraper_user_agent: str = "EdGenie/1.0 (Educational Research Tool)"
    scraper_watermark_min_year: int = 2026  # Auto-remove PapaCambridge watermarks for papers >= this year

    # ── Logging ──
    log_level: str = "INFO"

    # ── Unified 3-in-1 Pipeline (Unstructured.io Transform API) ──
    # Combines Text Extraction, Visual Extraction, and Syllabus Classification into a single cloud parse job.
    unstructured_api_key: str | None = None
    unstructured_server_url: str = "https://transform.unstructured.io"
    unstructured_profile: str = "best"
    unstructured_poll_interval: float = 5.0
    unstructured_max_poll_time: float = 600.0
    unstructured_timeout_seconds: float = 600.0

    # ── Visual Cropping & Resolution (PyMuPDF) ──
    visual_dpi: int = 300
    visual_pad: float = 2.0
    extract_tables: bool = True

    # ── Database & Cloud Asset Hosting ──
    postgres_uri: str = "postgresql://localhost:5432/edgenie"
    cdn_domain_prefix: str = "https://assets.edgenie.io"
    firebase_storage_bucket: str | None = None

    model_config = {
        "env_prefix": "EDGENIE_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton instance — import this across the pipeline
settings = PipelineSettings()
