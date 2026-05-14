"""PDF text extraction using PyMuPDF."""

from importlib import import_module
from pathlib import Path
from typing import Any

from dockar.models import DocumentPage


class PDFTextExtractor:
    """Extract embedded text and metadata from PDF files."""

    def extract(self, path: Path) -> list[DocumentPage]:
        """Return page-level embedded text for a PDF."""

        pdf_path = Path(path)
        if pdf_path.suffix.lower() != ".pdf":
            message = f"PDFTextExtractor only supports .pdf files: {pdf_path}"
            raise ValueError(message)

        fitz: Any = import_module("fitz")
        pages: list[DocumentPage] = []
        with fitz.open(pdf_path) as document:
            for index, page in enumerate(document, start=1):
                raw_text = page.get_text("text").strip()
                pages.append(
                    DocumentPage(
                        page_number=index,
                        raw_text=raw_text,
                        extraction_method="text",
                        metadata={
                            "width": float(page.rect.width),
                            "height": float(page.rect.height),
                        },
                    )
                )
        return pages
