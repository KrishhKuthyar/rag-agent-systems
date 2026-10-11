# RAG Agent Systems

A Python project for answering questions using documents and citing the sources
of its evidence. Built incrementally from the concepts in the
[AI Data Engineer Bootcamp RAG module](https://github.com/owshq-academy/ai-data-engineer-bootcamp/tree/main/src/mod-2-rag-agent).

## Current implementation

Local UTF-8 document loading, whitespace cleanup, and overlapping chunks with
source metadata.
The loader returns a dictionary containing the original text and absolute
source path. A separate function produces cleaned text without modifying that
record. The chunker splits the cleaned text into character-based passages,
retaining their source path, sequence number, and text offsets. File and decoding
errors propagate to the caller.

## Run

Requires Python 3.10 or newer. No third-party dependencies are needed yet.

```bash
python3 main.py
```

The example reads `data/sample.txt` and prints its source path, original text,
cleaned text, and chunks. The sample includes extra spaces, repeated blank lines,
and Unicode characters so the transformation is visible. Chunk output displays
newlines as `\n` so boundaries can be inspected.

## Processing behavior

`load_document(path)` reads a file using UTF-8 and keeps its source path beside
the text. `clean_text(text)` normalizes CRLF/CR line endings to newlines, collapses
spaces and tabs within each line, removes spaces at line edges, and reduces
repeated blank lines to one blank line between paragraphs.

Cleaning works line by line: `split("\n")` separates lines, a `for` loop cleans
each line, and `"\n".join(lines)` puts them back together. The regex `[ \t]+`
matches spaces and tabs; it preserves the newlines that identify paragraphs.
Words, punctuation, and other Unicode characters are retained.

This cleanup is intended for prose. It removes indentation and repeated spacing,
which can carry meaning in source code or preformatted tables.

`chunk_document(document, chunk_size=120, overlap=20)` takes a document record
and returns a list of chunk dictionaries. Each contains:

| Field | Meaning |
| --- | --- |
| `text` | The passage selected from the supplied text. |
| `source` | The document's source path. |
| `index` | The chunk's zero-based sequence number within this document. |
| `start` | The first character's position in the supplied text. |
| `end` | The exclusive end position, matching Python slice conventions. |

The runner passes cleaned text, so these offsets refer to that text, not the
original file's bytes or uncleaned contents. An index is a sequence number, not
a globally unique chunk ID. Empty text returns no chunks. Invalid sizes or
overlap raise `ValueError` before processing, including for empty input.

Chunk sizes and overlap are Python string character counts. Fixed boundaries
can split words, sentences, and paragraphs; they retain the selected characters
without trimming them. Token-based and structure-aware splitting can follow
once this baseline is understood.

## Why these components matter

The project is currently building document ingestion: loading, cleaning, and
chunking work. Embeddings, retrieval, generation, and agents remain pending.

Loading converts a file into text we can process. Keeping its source beside it
makes later passages traceable to the original file and enables source references
in answers. Cleaning removes distracting whitespace while preserving the
paragraph boundaries that later chunking can use.

Chunking gives retrieval smaller passages to rank instead of treating an entire
document as one result. Overlap repeats context around boundaries, reducing the
chance that useful surrounding text is separated completely. More overlap also
means more repeated text to store and process, so it is a parameter to evaluate.

Think of preparing notes from a book. First copy the passage and record which
book it came from. Then tidy its spacing while keeping distinct paragraphs
separate. Searching and answering questions will build on that prepared text.

## Flow of control

1. `python3 main.py` executes the imports and defines the `main` function.
2. The `if __name__ == "__main__"` guard calls `main` when the file is run directly.
3. `main` builds a path to `data/sample.txt` relative to its own file location.
4. `load_document` reads UTF-8 text and returns its original contents and absolute
   source path in a dictionary. Missing files and decoding errors stop execution.
5. `clean_text` receives the dictionary's text, normalizes line endings, cleans
   spaces and tabs within each line, and joins those lines with newlines. It
   limits blank-line runs and returns the result as a new string.
6. `main` makes a document record containing the cleaned text and the same source,
   then passes it to `chunk_document` with a size of 120 and overlap of 20.
7. The chunker checks its parameters, slices a passage, records its metadata, and
   advances its starting position while retaining the requested overlap. It
   stops as soon as the last passage reaches the end of the text.
8. Control returns to `main`, which prints the source, original text, cleaned
   text, and chunks. The original document dictionary is not modified.

```text
main.py → load_document(path) → {text, source}
        → clean_text(document["text"]) → cleaned string
        → chunk_document({text: cleaned string, source}) → list of chunks
        → print original text, cleaned text, and chunks
```

## Python syntax used

| Syntax | Role in this implementation |
| --- | --- |
| `from pathlib import Path` | Import a standard-library type for filesystem paths. |
| `from rag.documents import load_document` | Import a function from our own package. |
| `def load_document(path: Path) -> dict[str, str]:` | Define a function with an input parameter and type hints. The hints describe types; they do not enforce them at runtime. |
| Indentation | Put statements inside a function or loop. |
| `text = ...` | Assign an expression's result to a variable. |
| `path.read_text(encoding="utf-8")` | Call a method with a named argument to decode the file. |
| `{"text": text, "source": ...}` | Construct a dictionary with named values. |
| `document["text"]` | Retrieve a value by its dictionary key. |
| `Path(__file__).resolve().parent / "data" / "sample.txt"` | Build a path anchored to the runner's directory; `/` joins components for `Path` objects. |
| `for line in text.split("\n"):` | Loop over the individual lines in a string. |
| `r"[ \t]+"` | A raw regex string: match one or more spaces or tabs. |
| `re.sub(pattern, replacement, text)` | Replace matching text. |
| `lines.append(cleaned_line)` | Add a cleaned line to a list. |
| `"\n".join(lines)` | Combine the lines with newline separators. |
| `r"\n{3,}"` | A regex matching runs of at least three newlines. |
| `return` | Send a result to the function's caller. |
| `f"Source: {document['source']}"` | Insert a value into a formatted string. |
| `chunk_size: int = 120` | Describe an integer parameter and supply a default. |
| `list[dict[str, str \| int]]` | Describe a list of dictionaries whose values can be strings or integers. |
| `if ...: raise ValueError(...)` | Reject a parameter value with an explicit error. |
| `while start < len(text):` | Repeat while there is more text to process. |
| `min(start + chunk_size, len(text))` | Clamp a passage's end to the text length. |
| `text[start:end]` | Slice from `start` up to, but excluding, `end`. |
| `start = end - overlap` | Begin the next passage before the previous passage ends. |
| `break` | Leave the loop once the end of the text has been reached. |
| `repr(chunk["text"])` | Display text with escapes such as `\n` visible. |

For example, `First   paragraph.\n\n\nSecond paragraph.` becomes
`First paragraph.\n\nSecond paragraph.` The paragraph boundary survives. Cleaning
the result again returns the same text; this property is called idempotence.

For chunking, `abcdefghij` with `chunk_size=6` and `overlap=2` produces
`abcdef` at `[0:6]` and `efghij` at `[4:10]`. The shared `ef` provides overlap.
Requiring `0 <= overlap < chunk_size` makes each new start move forward.

## Checks

```bash
python3 -m unittest discover -s tests -v
```

Checks cover Unicode and paragraph preservation, individual line breaks,
punctuation and nonbreaking spaces, empty input, and idempotence: cleaning
already-cleaned text gives the same result.

Chunking checks cover exact overlap and offsets, source metadata, a shorter
final passage, empty documents, invalid parameters, Unicode reconstruction,
maximum valid overlap, and retaining the original document unchanged.

## Project structure

```text
rag-agent-systems/
├── main.py                 # Runnable entry point
├── rag/
│   ├── __init__.py
│   ├── chunking.py         # Overlapping character-based chunks
│   ├── documents.py        # Document loading
│   └── preprocessing.py   # Prose whitespace cleanup
├── data/
│   └── sample.txt          # Sample input
├── tests/
│   ├── test_chunking.py
│   └── test_preprocessing.py
└── README.md              # Implementation status and usage
```

## Planned work

- Stable chunk IDs that distinguish sources and repeated ingestion.
- Embeddings and similarity search.
- Answers grounded in retrieved passages, with source references.
- Vector-store persistence and DOCX/MinIO integrations.
- Framework integrations, evaluation, reliability, and agent tools.

## Development progress

- Implemented local UTF-8 document loading with an absolute source path.
- Checked Unicode and paragraph preservation, empty files, missing files,
  invalid UTF-8, and running the example from another working directory.
- Added whitespace cleanup that retains paragraph boundaries, a before-and-after
  example, and checks for preserved content and repeatable results.
- Added overlapping character-based chunks with source paths, indexes, and
  offsets in cleaned text; checked coverage and parameter validation.

The reference module mainly implements ingestion and low-level vector queries.
This project will also build the question-to-answer path and agent behavior,
with explicit failure handling and checks for each feature.
