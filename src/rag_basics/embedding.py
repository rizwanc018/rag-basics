from typing import cast

import lancedb
import tiktoken
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.doc_chunk import DocChunk
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

MAX_TOKENS = 8192

tokenizer = OpenAITokenizer(
    tokenizer=tiktoken.get_encoding("cl100k_base"),
    max_tokens=MAX_TOKENS,
)

converter = DocumentConverter()
result = converter.convert("https://arxiv.org/pdf/2206.01062")

document = result.document
# md_output = document.export_to_markdown()
# json_output = document.export_to_dict()

chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

# chunk_iter = chunker.chunk(dl_doc=document)
# chunks = list(chunk_iter)
chunks = cast(list[DocChunk], list(chunker.chunk(dl_doc=document)))

# for chunk in chunks:
#     print(chunk.model_dump_json(indent=2))

# print(len(chunks))

db = lancedb.connect("db/lancedb")

func = get_registry().get("openai").create(name="text-embedding-3-large")


class ChunkMetadata(LanceModel):
    filename: str | None
    page_numbers: list[int] | None
    title: str | None


class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: ChunkMetadata


table = db.create_table("docling", schema=Chunks, mode="overwrite")


processed_text = [
    {
        "text": chunk.text,
        "metadata": {
            "filename": chunk.meta.origin.filename if chunk.meta.origin else None,
            "page_numbers": sorted(
                {prov.page_no for item in chunk.meta.doc_items for prov in item.prov}
            )
            or None,
            "title": chunk.meta.headings[0] if chunk.meta.headings else None,
        },
    }
    for chunk in chunks
]

table.add(processed_text)

table.to_pandas()
table.count_rows()
