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
errors propagate to the caller. Each chunk also has a repeatable SHA-256 ID
derived from its source, offsets, and complete passage text.

## Run

Requires Python 3.9 or newer. No third-party dependencies are needed yet.

```bash
python3 main.py
```

Check the interpreter selected by your terminal with `python3 --version`.
On this Mac, `/usr/bin/python3` is Python 3.9.6, while the development shell's
`python3` resolves to Homebrew Python 3.12.4. The example and test suite are
checked on both interpreters. An IDE or another terminal can select a different
interpreter even when the command is spelled `python3` in each.

The return annotation uses `Union[str, int]` from the standard-library `typing`
module to mean a string or an integer. The shorter `str | int` syntax requires
Python 3.10 or newer and raises a `TypeError` during import on Python 3.9.
Type hints do not enforce values, but their expressions can still be evaluated
while Python defines a function; that is why an annotation can stop startup.

The example reads `data/sample.txt` and prints its source path, original text,
cleaned text, and chunks. The sample includes extra spaces, repeated blank lines,
and Unicode characters so the transformation is visible. Chunk output displays
newlines as `\n` so boundaries can be inspected. Run it twice to compare the
chunk IDs: unchanged input produces the same IDs on both runs.

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
| `id` | A 64-character SHA-256 fingerprint of the source, positions, text, and identity format version. |
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
7. The chunker checks its parameters and slices a passage. `create_chunk_id`
   serializes the source, offsets, complete passage text, and format version to
   JSON, encodes that record as UTF-8, and calculates its SHA-256 fingerprint.
   The chunker adds the ID and metadata to its result list, then advances its
   starting position with overlap. It stops when a passage reaches the text end.
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
| `list[dict[str, Union[str, int]]]` | Describe a list of dictionaries whose values can be strings or integers, using the Python 3.9-compatible `typing.Union`. |
| `if ...: raise ValueError(...)` | Reject a parameter value with an explicit error. |
| `while start < len(text):` | Repeat while there is more text to process. |
| `min(start + chunk_size, len(text))` | Clamp a passage's end to the text length. |
| `text[start:end]` | Slice from `start` up to, but excluding, `end`. |
| `start = end - overlap` | Begin the next passage before the previous passage ends. |
| `break` | Leave the loop once the end of the text has been reached. |
| `repr(chunk["text"])` | Display text with escapes such as `\n` visible. |
| `json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))` | Serialize fields with sorted keys, Unicode characters, and fixed JSON separators. |
| `payload.encode("utf-8")` | Convert the serialized string into the bytes required by a hash function. |
| `hashlib.sha256(...).hexdigest()` | Calculate a deterministic SHA-256 fingerprint and return it as hexadecimal text. |

For example, `First   paragraph.\n\n\nSecond paragraph.` becomes
`First paragraph.\n\nSecond paragraph.` The paragraph boundary survives. Cleaning
the result again returns the same text; this property is called idempotence.

For chunking, `abcdefghij` with `chunk_size=6` and `overlap=2` produces
`abcdef` at `[0:6]` and `efghij` at `[4:10]`. The shared `ef` provides overlap.
Requiring `0 <= overlap < chunk_size` makes each new start move forward.

## Chunking parameters and executed examples

`chunk_size=120` and `overlap=20` are defaults, not requirements or measured
optimal values. The runner explicitly passes these demonstration values.
Other callers can supply different integers:

```python
document = {"text": "abcdefghij", "source": "example.txt"}
chunks = chunk_document(document, chunk_size=6, overlap=2)
```

For a single call, we use fixed-size sliding windows. Starting at zero, the
chunker selects `text[start:end]`, records the passage and source metadata, and
advances to `end - overlap`. It stops as soon as a passage reaches the end.
The final passage can be shorter than `chunk_size`.

The following inputs were run against the implementation:

| Input text | Size | Overlap | Output text and exclusive offsets |
| --- | --- | --- | --- |
| `abcdefghij` | 6 | 2 | `abcdef` at `[0:6]`; `efghij` at `[4:10]` |
| `abcdefghij` | 6 | 0 | `abcdef` at `[0:6]`; `ghij` at `[6:10]` |
| `abcdefghij` | 4 | 1 | `abcd` at `[0:4]`; `defg` at `[3:7]`; `ghij` at `[6:10]` |
| `A red fox sleeps.` | 10 | 4 | `'A red fox '` at `[0:10]`; `'fox sleeps'` at `[6:16]`; `'eeps.'` at `[12:17]` |

Every chunk retains `source="example.txt"` and a zero-based index. For the first
example, the full output is:

```python
[
    {
        "id": "453a196c5082f5ddc6bbd394c8fa99b9f6c47f8ed923ed656046816bbd4c71a6",
        "text": "abcdef", "source": "example.txt", "index": 0, "start": 0, "end": 6,
    },
    {
        "id": "8194dfdf6ebe6efac289bad84b39770e50c61b774fbaebf4768b013e77cfbe2f",
        "text": "efghij", "source": "example.txt", "index": 1, "start": 4, "end": 10,
    },
]
```

Here, `text[0:6]` selects `abcdef`. The next start is `6 - 2 = 4`, which repeats
`ef`. The next end is `min(4 + 6, 10) = 10`, which selects `efghij` and stops.
With zero overlap, the next start is 6 and there is no repeated text.

The sentence example also exposes the limitation: character boundaries can cut
words, producing a final fragment such as `eeps.`. This implementation makes
the mechanics visible; sentence-, paragraph-, or token-aware chunking will need
a different boundary-selection strategy.

Smaller chunks isolate more specific passages but carry less context. Larger
chunks retain more context but may include unrelated material. Overlap repeats
context across boundaries and increases the text processed and stored. Actual
parameter choices should be evaluated using the intended documents and questions.
The current runner's 120/20 values let us inspect several passages from a short
sample; they are character counts, not model token counts.

The sample was also run using `data/sample.txt`, cleaned before chunking. Its
exact 345-character chunking input is:

```text
RAG study notes

A RAG system retrieves relevant passages from documents before generating an answer.

Chunk overlap repeats some text between neighboring passages. It can preserve context when an idea crosses a chunk boundary.

Source metadata records where a passage came from. A filename such as café-notes.txt may contain Unicode characters.
```

There is no final newline in this cleaned input. With size 120 and overlap 20,
the actual output passages are shown below using Python string representations
so paragraph separators remain visible:

```text
index=0, start=0, end=120
'RAG study notes\n\nA RAG system retrieves relevant passages from documents before generating an answer.\n\nChunk overlap rep'

index=1, start=100, end=220
'.\n\nChunk overlap repeats some text between neighboring passages. It can preserve context when an idea crosses a chunk bo'

index=2, start=200, end=320
'a crosses a chunk boundary.\n\nSource metadata records where a passage came from. A filename such as café-notes.txt may co'

index=3, start=300, end=345
'afé-notes.txt may contain Unicode characters.'
```

All four chunks carry the absolute path to `data/sample.txt` as their source.
Offsets describe the cleaned string. To rerun the full example, use
`python3 main.py` from the project root.

## Stable chunk identity

Chunk `index=0` describes order within a document; many documents have a first
chunk. A stable ID distinguishes the actual source passage and lets future
storage recognize it on repeated ingestion. It also distinguishes repeated
copies of the same text at different positions in one document.

`create_chunk_id(source, start, end, text)` builds this versioned record:

```python
{
    "version": 1,
    "source": "notes.txt",
    "start": 0,
    "end": 6,
    "text": "abcdef",
}
```

The helper serializes it with sorted keys and fixed separators. JSON preserves
field boundaries even when text contains punctuation, quotes, or newlines. It
then encodes the record as UTF-8 bytes and hashes it with SHA-256, returning all
64 hexadecimal characters. The fixed format and version are part of the ID
contract; changing them changes the fingerprints.

The function is called as each chunk is created, before the chunk dictionary is
appended to the results. Its `id` joins the existing text and source metadata.
The sequence index is excluded: a passage with identical source, positions, and
text retains its ID even if a different chunking configuration changes its
sequence number. Chunk size and overlap affect IDs through the passages they
produce, rather than being separate identity fields.

Actual input used for the repeatability check:

```python
document = {"text": "abcdefghij", "source": "notes.txt"}
first = chunk_document(document, chunk_size=6, overlap=2)
second = chunk_document(document, chunk_size=6, overlap=2)
print(first == second)
```

Output: `True`. Both runs return these passages and IDs:

```text
[0:6]  'abcdef'
81d339606e747275c108694cdf3b946016a58f573c983432af588d827e712521

[4:10] 'efghij'
8e9cdfa7241e85beafa0a33b6872952effd6482d79da308c39b680224e589fee
```

Keeping the text and parameters the same but setting `source="other.txt"`
produces these different IDs:

```text
[0:6]  'abcdef'
0339ebe08ff4cfab1134f91c4b907cc43f6003d1647ed6d310a71ca4f8afc2b5

[4:10] 'efghij'
67b708adceac84b583b6a5da7661ca0e0cb2d07f1d195a1e1de2fe21619d390e
```

IDs use the source string exactly as supplied. The file loader supplies absolute
paths, so moving the project or renaming a file changes its IDs. A stable
document key independent of local filesystem layout can be introduced when
storage is added. Editing text or shifting chunk positions also changes the
affected IDs; these are fingerprints of passages in a specific source location,
not permanent identities that follow a passage through arbitrary edits.

There is no persistence or deduplication operation yet. These IDs provide the
keys for future storage updates; removing stale chunks after document edits
will require an explicit reconciliation step.

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

Identity checks cover fixed version-one fingerprints, repeatable ingestion,
distinct sources and positions, edited text, complete-text hashing, sequence
index independence, and Unicode stability across separate Python processes.

## Project structure

```text
rag-agent-systems/
├── main.py                 # Runnable entry point
├── rag/
│   ├── __init__.py
│   ├── chunking.py         # Overlapping chunks and stable passage IDs
│   ├── documents.py        # Document loading
│   └── preprocessing.py   # Prose whitespace cleanup
├── data/
│   └── sample.txt          # Sample input
├── tests/
│   ├── test_chunk_ids.py
│   ├── test_chunking.py
│   └── test_preprocessing.py
└── README.md              # Implementation status and usage
```

## Planned work

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
- Added versioned SHA-256 chunk IDs derived from source, positions, and complete
  text; checked repeated ingestion, source separation, and cross-process stability.
- Replaced the Python 3.10-only union annotation with `typing.Union` and checked
  startup and all tests on Python 3.9.6 and 3.12.4.

The reference module mainly implements ingestion and low-level vector queries.
This project will also build the question-to-answer path and agent behavior,
with explicit failure handling and checks for each feature.
