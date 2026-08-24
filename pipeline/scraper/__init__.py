"""
EdGenie Scraper — Standalone module for downloading CAIE past papers from PapaCambridge.

Usage (standalone):
    python -m pipeline.scraper scrape --level o-level --subject accounting --code 7707 --year 2023 --session jun
    python -m pipeline.scraper list --code 7707
"""

from pipeline.scraper.models import ScrapeTarget, PaperFileInfo, ScrapeManifest, PaperDocumentType
from pipeline.scraper.url_builder import URLBuilder
from pipeline.scraper.downloader import PDFDownloader
from pipeline.scraper.scraper import PapaCambridgeScraper

__all__ = [
    "ScrapeTarget",
    "PaperFileInfo",
    "ScrapeManifest",
    "PaperDocumentType",
    "URLBuilder",
    "PDFDownloader",
    "PapaCambridgeScraper",
]
