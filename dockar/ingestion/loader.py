"""High-level document loading pipeline."""

from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

from dockar.ingestion.interfaces import TextExtractor
from dockar.ingestion.pdf import PDFTextExtractor
from dockar.models import Document, DocumentPage
from dockar.ocr.interfaces import OcrEngine
from dockar.ocr.tesseract import OCRProcessor


class DocumentLoader:
    """Load PDFs using embedded text first and OCR when text density is low."""

    def __init__(
        self,
        text_extractor: TextExtractor | None = None,
        min_chars_per_page: int = 40,
        ocr_enabled: bool = True,
        max_doc_pages: int | None = None,
    ) -> None:
        self.text_extractor = text_extractor or PDFTextExtractor()
        self.min_chars_per_page = min_chars_per_page
        self.ocr_enabled = ocr_enabled
        self.max_doc_pages = max_doc_pages

    def load(self, path: Path) -> Document:
        """Load one PDF into a domain document."""

        pdf_path = Path(path)
        self._validate_pdf_path(pdf_path)

        text_pages = self.text_extractor.extract(pdf_path)
        limited_pages = self._limit_pages(text_pages)
        text_density = self._text_density(limited_pages)
        should_ocr = self.ocr_enabled and self._is_low_text_density(limited_pages)
        pages = limited_pages

        if should_ocr:
            ocr_processor = self._build_ocr_processor(pdf_path)
            ocr_pages = ocr_processor.extract_pages(limited_pages)
            pages = self._merge_pages(limited_pages, ocr_pages)

        raw_text = self._join_pages(pages)
        return Document(
            id=self._document_id(pdf_path),
            source_path=pdf_path,
            raw_text=raw_text,
            pages=pages,
            metadata={
                "file_name": pdf_path.name,
                "file_type": "pdf",
                "page_count": len(text_pages),
                "loaded_page_count": len(limited_pages),
                "text_density": text_density,
                "is_scanned": should_ocr,
                "ocr_applied": should_ocr,
            },
        )

    def ingest(self, path: Path) -> list[Document]:
        """Load one PDF file or all PDFs in a directory."""

        source_path = Path(path)
        if source_path.is_dir():
            return [self.load(pdf_path) for pdf_path in sorted(source_path.glob("*.pdf"))]
        return [self.load(source_path)]

    def _build_ocr_processor(self, path: Path) -> OcrEngine:
        return OCRProcessor(path)

    def _validate_pdf_path(self, path: Path) -> None:
        if not path.exists():
            message = f"Document path does not exist: {path}"
            raise FileNotFoundError(message)
        if not path.is_file():
            message = f"Document path must be a file: {path}"
            raise ValueError(message)
        if path.suffix.lower() != ".pdf":
            message = f"DocumentLoader only supports PDF files: {path}"
            raise ValueError(message)

    def _limit_pages(self, pages: list[DocumentPage]) -> list[DocumentPage]:
        if self.max_doc_pages is None:
            return pages
        return pages[: self.max_doc_pages]

    def _is_low_text_density(self, pages: list[DocumentPage]) -> bool:
        return self._text_density(pages) < self.min_chars_per_page

    def _text_density(self, pages: list[DocumentPage]) -> float:
        if not pages:
            return 0.0
        return sum(len(page.raw_text.strip()) for page in pages) / len(pages)

    def _merge_pages(
        self,
        text_pages: list[DocumentPage],
        ocr_pages: Iterable[DocumentPage],
    ) -> list[DocumentPage]:
        ocr_by_page = {page.page_number: page for page in ocr_pages}
        merged_pages: list[DocumentPage] = []
        for page in text_pages:
            ocr_page = ocr_by_page.get(page.page_number)
            if ocr_page is None:
                merged_pages.append(page)
                continue
            merged_pages.append(
                DocumentPage(
                    page_number=page.page_number,
                    raw_text=ocr_page.raw_text,
                    extraction_method="ocr",
                    metadata={
                        **page.metadata,
                        **ocr_page.metadata,
                        "embedded_text": page.raw_text,
                    },
                )
            )
        return merged_pages

    def _join_pages(self, pages: list[DocumentPage]) -> str:
        return "\n\n".join(page.raw_text for page in pages if page.raw_text)

    def _document_id(self, path: Path) -> str:
        digest = sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
        return f"{path.stem}-{digest}"
