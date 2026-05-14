"""OCR boundary."""

from dockar.ocr.interfaces import OcrEngine
from dockar.ocr.tesseract import OCRProcessor

__all__ = ["OCRProcessor", "OcrEngine"]
