# rag-basics

A small learning project for the basics of RAG (Retrieval-Augmented Generation):

1. Convert a PDF to a structured document with [Docling](https://github.com/docling-project/docling)
2. Split it into chunks
3. Embed the chunks with OpenAI and store them in [LanceDB](https://lancedb.com)
4. Search the stored chunks with a query

## Setup


```bash
uv sync
```

Create a `.env` file with your OpenAI key:

```
OPENAI_API_KEY=sk-...
```

## Usage

**1. Build the database.** This downloads the [DocLayNet paper](https://arxiv.org/pdf/2206.01062), chunks it, embeds it with `text-embedding-3-large`, and saves it to `db/lancedb`:

```bash
uv run --env-file .env src/rag_basics/embedding.py
```

**2. Search it.** This runs a vector search for "DocLayNet" and prints the top 3 chunks:

```bash
uv run --env-file .env src/rag_basics/main.py
```

## Files

| File | What it does |
| --- | --- |
| [src/rag_basics/embedding.py](src/rag_basics/embedding.py) | Converts, chunks, embeds, and stores the document |
| [src/rag_basics/main.py](src/rag_basics/main.py) | Searches the stored chunks |
