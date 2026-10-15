import tiktoken
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

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

chunk_iter = chunker.chunk(dl_doc=document)
chunks = list(chunk_iter)
print(len(chunks))

for chunk in chunks:
    print(chunk)
