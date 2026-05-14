from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import Any

from dockar.models import DocumentPage


class OCRProcessor:
    """Run Tesseract OCR over selected PDF pages."""

    def __init__(self, pdf_path: Path, dpi: int = 200, language: str = "eng") -> None:
        self.pdf_path = Path(pdf_path)
        self.dpi = dpi
        self.language = language

    def extract_pages(self, pages: list[DocumentPage]) -> list[DocumentPage]:
        """Return OCR text for the provided page list."""

        if not pages:
            return []

        fitz: Any = import_module("fitz")
        image_module: Any = import_module("PIL.Image")
        pytesseract: Any = import_module("pytesseract")

        requested_pages = {page.page_number for page in pages}
        ocr_pages: list[DocumentPage] = []
        zoom = self.dpi / 72

        with fitz.open(self.pdf_path) as document:
            for page_number in sorted(requested_pages):
                page = document[page_number - 1]
                matrix = fitz.Matrix(zoom, zoom)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image = image_module.open(BytesIO(pixmap.tobytes("png")))
                raw_text = pytesseract.image_to_string(image, lang=self.language).strip()
                ocr_pages.append(
                    DocumentPage(
                        page_number=page_number,
                        raw_text=raw_text,
                        extraction_method="ocr",
                        metadata={
                            "dpi": self.dpi,
                            "language": self.language,
                        },
                    )
                )

        return ocr_pages
