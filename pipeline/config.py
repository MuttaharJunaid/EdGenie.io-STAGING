"""
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

    # ── Scraper ──
    scraper_rate_limit_delay: float = 2.0  # seconds between HTTP requests
    scraper_max_retries: int = 5
    scraper_timeout: int = 30  # seconds per request
    scraper_user_agent: str = "EdGenie/1.0 (Educational Research Tool)"

    # ── Logging ──
    log_level: str = "INFO"

    # ── AI Agents ──
    llm_provider: str = "google"
    llm_model: str = ""

    model_config = {
        "env_prefix": "EDGENIE_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton instance — import this across the pipeline
settings = PipelineSettings()
