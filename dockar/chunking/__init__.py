"""Document chunking boundary."""

from dockar.chunking.fixed import Chunker, FixedSizeChunker
from dockar.chunking.interfaces import ChunkingStrategy

__all__ = ["Chunker", "ChunkingStrategy", "FixedSizeChunker"]
