import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TextBlock(BaseModel):
    text: str
    bbox: List[float]  # [x0, y0, x1, y1]
    font_name: str
    font_size: float
    is_bold: bool
    is_italic: bool
    page_num: int

class ExtractedPage(BaseModel):
    page_num: int
    width: float
    height: float
    blocks: List[TextBlock] = []
    raw_text: str

class TextExtractor:
    """Extracts structured text blocks with font and coordinate metadata from PDF files."""

    def extract(self, pdf_path: Path) -> List[ExtractedPage]:
        """
        Extracts pages and text blocks from a PDF using PyMuPDF (fitz)
        supplemented by pdfplumber for coordinate precision if needed.
        """
        logger.info(f"Extracting structured text from: {pdf_path}")
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        extracted_pages = []
        doc = fitz.open(pdf_path)

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            width = page.rect.width
            height = page.rect.height
            
            # Using PyMuPDF's page.get_text("dict") to get precise block/span formatting
            text_dict = page.get_text("dict")
            blocks = []
            
            for b in text_dict.get("blocks", []):
                # We are only interested in text blocks (type 0)
                if b.get("type") == 0:
                    block_text_parts = []
                    x0, y0, x1, y1 = b.get("bbox", [0.0, 0.0, 0.0, 0.0])
                    
                    # Accumulate font details for the block based on the first span
                    first_span = None
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            if not first_span:
                                first_span = span
                            block_text_parts.append(span.get("text", ""))
                    
                    if first_span:
                        font_name = first_span.get("font", "unknown")
                        font_size = first_span.get("size", 10.0)
                        
                        # Heuristic check for bold/italic from font flags or font name
                        flags = first_span.get("flags", 0)
                        is_bold = bool(flags & 2) or "bold" in font_name.lower() or "black" in font_name.lower()
                        is_italic = bool(flags & 1) or "italic" in font_name.lower() or "oblique" in font_name.lower()
                        
                        text = " ".join(block_text_parts).strip()
                        if text:
                            blocks.append(TextBlock(
                                text=text,
                                bbox=[x0, y0, x1, y1],
                                font_name=font_name,
                                font_size=font_size,
                                is_bold=is_bold,
                                is_italic=is_italic,
                                page_num=page_num
                            ))

            raw_text = page.get_text()
            extracted_pages.append(ExtractedPage(
                page_num=page_num,
                width=width,
                height=height,
                blocks=blocks,
                raw_text=raw_text
            ))

        logger.info(f"Extracted {len(extracted_pages)} pages from {pdf_path.name}")
        return extracted_pages
