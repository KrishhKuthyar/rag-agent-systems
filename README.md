# RAG Agent Systems

A Python project for answering questions using documents and citing the sources
of its evidence. Built incrementally from the concepts in the
[AI Data Engineer Bootcamp RAG module](https://github.com/owshq-academy/ai-data-engineer-bootcamp/tree/main/src/mod-2-rag-agent).

## Current implementation

Local UTF-8 text loading with source metadata. The loader returns a dictionary
containing the document's text and absolute source path. File and decoding
errors propagate to the caller.

## Run

Requires Python 3.10 or newer. No third-party dependencies are needed yet.

```bash
python3 main.py
```

The example reads `data/sample.txt` and prints its source path and contents.

## Project structure

```text
rag-agent-systems/
├── main.py                 # Runnable entry point
├── rag/
│   ├── __init__.py
│   └── documents.py        # Document loading
├── data/
│   └── sample.txt          # Sample input
└── README.md              # Implementation status and usage
```

## Planned work

- Text cleaning and chunking with source metadata.
- Embeddings and similarity search.
- Answers grounded in retrieved passages, with source references.
- Vector-store persistence and DOCX/MinIO integrations.
- Framework integrations, evaluation, reliability, and agent tools.

## Development progress

- Implemented local UTF-8 document loading with an absolute source path.
- Checked Unicode and paragraph preservation, empty files, missing files,
  invalid UTF-8, and running the example from another working directory.

The reference module mainly implements ingestion and low-level vector queries.
This project will also build the question-to-answer path and agent behavior,
with explicit failure handling and checks for each feature.
