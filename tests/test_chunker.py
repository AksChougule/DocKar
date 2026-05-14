from pathlib import Path

from dockar.chunking import Chunker
from dockar.models import Document, DocumentPage


def test_chunker_splits_document_text_by_configured_size() -> None:
    document = Document(
        id="doc-1",
        source_path=Path("sample.pdf"),
        raw_text="alpha beta gamma delta",
    )
    chunker = Chunker(chunk_size=11)

    chunks = chunker.chunk(document)

    assert [chunk.text for chunk in chunks] == ["alpha beta", "gamma delta"]
    assert [chunk.chunk_id for chunk in chunks] == [
        "doc-1::chunk-0001",
        "doc-1::chunk-0002",
    ]
    assert all(chunk.document_id == "doc-1" for chunk in chunks)
    assert all(len(chunk.text) <= 11 for chunk in chunks)


def test_chunker_prefers_sentence_boundary_near_limit() -> None:
    document = Document(
        id="doc-2",
        source_path=Path("sample.pdf"),
        raw_text="First sentence. Second sentence continues for a while.",
    )
    chunker = Chunker(chunk_size=30, sentence_window=20)

    chunks = chunker.chunk(document)

    assert chunks[0].text == "First sentence."
    assert chunks[1].text == "Second sentence continues for"
    assert chunks[2].text == "a while."


def test_chunker_preserves_page_ranges() -> None:
    document = Document(
        id="doc-3",
        source_path=Path("sample.pdf"),
        pages=[
            DocumentPage(page_number=1, raw_text="Page one has text."),
            DocumentPage(page_number=2, raw_text="Page two also has text."),
            DocumentPage(page_number=3, raw_text="Page three closes."),
        ],
    )
    chunker = Chunker(chunk_size=45)

    chunks = chunker.chunk(document)

    assert chunks[0].page_range == (1, 2)
    assert chunks[1].page_range == (3, 3)


def test_chunker_returns_empty_list_for_empty_document() -> None:
    document = Document(id="doc-4", source_path=Path("empty.pdf"))

    assert Chunker().chunk(document) == []
