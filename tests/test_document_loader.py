from pathlib import Path

import pytest

from dockar.ingestion.loader import DocumentLoader
from dockar.models import DocumentPage
from dockar.ocr.interfaces import OcrEngine


class FakeTextExtractor:
    def __init__(self, pages: list[DocumentPage]) -> None:
        self.pages = pages

    def extract(self, path: Path) -> list[DocumentPage]:
        return self.pages


class FakeOcrProcessor:
    def __init__(self, pages: list[DocumentPage]) -> None:
        self.pages = pages

    def extract_pages(self, pages: list[DocumentPage]) -> list[DocumentPage]:
        return self.pages


class FakeDocumentLoader(DocumentLoader):
    def __init__(
        self,
        text_pages: list[DocumentPage],
        ocr_pages: list[DocumentPage],
        **kwargs: object,
    ) -> None:
        super().__init__(text_extractor=FakeTextExtractor(text_pages), **kwargs)
        self.ocr_pages = ocr_pages

    def _build_ocr_processor(self, path: Path) -> OcrEngine:
        return FakeOcrProcessor(self.ocr_pages)


def test_document_loader_uses_text_when_density_is_high(tmp_path: Path) -> None:
    pdf_path = tmp_path / "invoice.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    text_pages = [
        DocumentPage(page_number=1, raw_text="A text-heavy invoice page.", extraction_method="text")
    ]
    loader = FakeDocumentLoader(
        text_pages=text_pages,
        ocr_pages=[],
        min_chars_per_page=10,
    )

    document = loader.load(pdf_path)

    assert document.raw_text == "A text-heavy invoice page."
    assert document.pages[0].extraction_method == "text"
    assert document.metadata["is_scanned"] is False
    assert document.metadata["ocr_applied"] is False


def test_document_loader_runs_ocr_when_density_is_low(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    text_pages = [DocumentPage(page_number=1, raw_text="", extraction_method="text")]
    ocr_pages = [DocumentPage(page_number=1, raw_text="OCR invoice text", extraction_method="ocr")]
    loader = FakeDocumentLoader(
        text_pages=text_pages,
        ocr_pages=ocr_pages,
        min_chars_per_page=10,
    )

    document = loader.load(pdf_path)

    assert document.raw_text == "OCR invoice text"
    assert document.pages[0].extraction_method == "ocr"
    assert document.pages[0].metadata["embedded_text"] == ""
    assert document.metadata["is_scanned"] is True
    assert document.metadata["ocr_applied"] is True


def test_document_loader_can_disable_ocr(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    loader = FakeDocumentLoader(
        text_pages=[DocumentPage(page_number=1, raw_text="", extraction_method="text")],
        ocr_pages=[DocumentPage(page_number=1, raw_text="OCR text", extraction_method="ocr")],
        min_chars_per_page=10,
        ocr_enabled=False,
    )

    document = loader.load(pdf_path)

    assert document.raw_text == ""
    assert document.pages[0].extraction_method == "text"
    assert document.metadata["ocr_applied"] is False


def test_document_loader_rejects_non_pdf(tmp_path: Path) -> None:
    text_path = tmp_path / "notes.txt"
    text_path.write_text("hello", encoding="utf-8")
    loader = DocumentLoader()

    with pytest.raises(ValueError, match="only supports PDF"):
        loader.load(text_path)
