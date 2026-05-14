"""Fixed-size document chunking with page provenance."""

from dataclasses import dataclass

from dockar.models import Chunk, Document


@dataclass(frozen=True)
class TextSegment:
    """Internal text segment mapped to a source page."""

    text: str
    page_number: int | None


class FixedSizeChunker:
    """Split documents into fixed-size chunks, preferring sentence boundaries."""

    def __init__(self, chunk_size: int = 4000, sentence_window: int = 200) -> None:
        if chunk_size <= 0:
            message = "chunk_size must be greater than 0"
            raise ValueError(message)
        if sentence_window < 0:
            message = "sentence_window must be greater than or equal to 0"
            raise ValueError(message)

        self.chunk_size = chunk_size
        self.sentence_window = min(sentence_window, chunk_size)

    def chunk(self, document: Document) -> list[Chunk]:
        """Return fixed-size chunks for a document."""

        segments = self._document_segments(document)
        if not segments:
            return []

        chunks: list[Chunk] = []
        current_text = ""
        current_pages: list[int] = []

        for segment in segments:
            remaining = segment.text
            while remaining:
                available = self._available_chars(current_text)
                if available <= 0:
                    chunks.append(
                        self._build_chunk(document.id, len(chunks), current_text, current_pages)
                    )
                    current_text = ""
                    current_pages = []
                    available = self._available_chars(current_text)

                if len(remaining) <= available:
                    current_text = self._append_text(current_text, remaining)
                    if segment.page_number is not None:
                        current_pages.append(segment.page_number)
                    remaining = ""
                    continue

                split_at = self._split_index(remaining, available)
                piece = remaining[:split_at].strip()
                if not piece:
                    piece = remaining[:available].strip()
                    split_at = available

                current_text = self._append_text(current_text, piece)
                if segment.page_number is not None:
                    current_pages.append(segment.page_number)
                chunks.append(
                    self._build_chunk(document.id, len(chunks), current_text, current_pages)
                )
                current_text = ""
                current_pages = []
                remaining = remaining[split_at:].strip()

        if current_text:
            chunks.append(self._build_chunk(document.id, len(chunks), current_text, current_pages))

        return chunks

    def _available_chars(self, current: str) -> int:
        separator_length = 2 if current else 0
        return self.chunk_size - len(current) - separator_length

    def _document_segments(self, document: Document) -> list[TextSegment]:
        if document.pages:
            return [
                TextSegment(page.raw_text.strip(), page.page_number)
                for page in document.pages
                if page.raw_text.strip()
            ]
        if document.raw_text.strip():
            return [TextSegment(document.raw_text.strip(), None)]
        return []

    def _split_index(self, text: str, max_length: int) -> int:
        window_start = max(0, max_length - self.sentence_window)
        candidate_region = text[window_start:max_length]

        sentence_offsets = [
            candidate_region.rfind(". "),
            candidate_region.rfind("? "),
            candidate_region.rfind("! "),
            candidate_region.rfind("\n"),
        ]
        best_offset = max(sentence_offsets)
        if best_offset >= 0:
            return window_start + best_offset + 1

        whitespace_index = text.rfind(" ", 0, max_length)
        if whitespace_index > 0:
            return whitespace_index

        return max_length

    def _append_text(self, current: str, addition: str) -> str:
        addition = addition.strip()
        if not current:
            return addition
        return f"{current}\n\n{addition}"

    def _build_chunk(
        self,
        document_id: str,
        index: int,
        text: str,
        pages: list[int],
    ) -> Chunk:
        unique_pages = sorted(set(pages))
        page_range = (unique_pages[0], unique_pages[-1]) if unique_pages else None
        return Chunk(
            chunk_id=f"{document_id}::chunk-{index + 1:04d}",
            text=text,
            document_id=document_id,
            page_range=page_range,
        )


class Chunker(FixedSizeChunker):
    """Default fixed-size chunker."""
