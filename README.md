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

A standalone cosine-similarity function compares numeric vector directions.
The local runner demonstrates it using hand-written vectors after displaying
the document chunks. The default embedding demo now uses a free local MiniLM
model to produce real 384-dimensional vectors and compare a question with the
four sample passages. Downloading and cached offline inference have both been
verified on this Mac. No API key or paid request is needed for local embeddings.
The demo also retrieves the top matching passages in descending cosine order,
preserving their text, IDs, sources, and offsets. It formats the selected
passages as source-labeled context for future answer generation. Persistence
and answer generation remain pending.

## Run

The local example requires Python 3.9 or newer and no third-party dependencies.
The local embedding model uses a Python 3.12 virtual environment and pinned
dependencies. The model files are downloaded once and cached in `.local/models`.

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
The final section displays a cosine-similarity demonstration using the query
vector `[1.0, 0.0]` and four explicit candidate vectors.

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

The project can load, clean, and chunk documents, then generate local embeddings
and measure cosine similarity to a question. Retrieval ranking, answer generation,
and agent behavior remain pending.

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
9. A separate math demonstration compares a hand-written query vector with
   four hand-written candidates using `cosine_similarity`. The runner prints
   every input vector and its score, formatted to four decimal places. The
   document chunks are not assigned vectors or ranked in this step.

```text
main.py → load_document(path) → {text, source}
        → clean_text(document["text"]) → cleaned string
        → chunk_document({text: cleaned string, source}) → list of chunks
        → print original text, cleaned text, and chunks

hand-written query and candidate vectors → cosine_similarity → print scores
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

## Cosine similarity: intuition, calculation, and executed examples

We have prepared source-bearing chunks. The local embedding runner represents
questions and passages as vectors from the same model. The separate math demo
in `main.py` uses explicit numbers to make that comparison easy to inspect.

A vector is a sequence of numbers. In two dimensions, imagine an arrow from
the origin to those coordinates. Cosine similarity compares the directions of
two nonzero arrows:

```text
cosine(A, B) = dot(A, B) / (length(A) * length(B))
dot(A, B)    = sum(A[i] * B[i])
length(A)   = sqrt(sum(A[i] * A[i]))
```

For `A=[1, 0]` and `B=[1, 1]`, the dot product is `1*1 + 0*1 = 1`.
The lengths are `1` and `sqrt(2)`, so the score is approximately `0.7071`.
Two arrows pointing in exactly the same direction score `1`; perpendicular
arrows score `0`; arrows pointing in opposite directions score `-1`.
Scores measure geometry. These hand-written vectors do not encode document
meaning or establish that a real passage is relevant to a question.

Actual inputs run by `python3 main.py`:

```python
query_vector = [1.0, 0.0]
candidate_vectors = {
    "same direction": [2.0, 0.0],
    "diagonal": [1.0, 1.0],
    "perpendicular": [0.0, 1.0],
    "opposite direction": [-1.0, 0.0],
}
```

Actual output after the document-ingestion output:

```text
Cosine similarity demo (hand-written vectors):
Query: [1.0, 0.0]
same direction: [2.0, 0.0] -> 1.0000
diagonal: [1.0, 1.0] -> 0.7071
perpendicular: [0.0, 1.0] -> 0.0000
opposite direction: [-1.0, 0.0] -> -1.0000
```

`cosine_similarity(a, b)` in `rag/similarity.py` follows this flow:

1. Reject empty vectors and unequal dimensions. Dimension is the coordinate
   count; mismatched coordinates cannot be compared by this function.
2. Reject non-finite coordinates such as NaN or infinity.
3. Find each vector's largest absolute coordinate. Reject zero vectors, which
   have no direction and would make the denominator zero.
4. Divide each coordinate by its vector's largest absolute coordinate. Positive
   rescaling preserves direction. It keeps intermediate values manageable for
   both very large and very small finite inputs.
5. Calculate the dot product and both lengths using those scaled coordinates.
6. Divide by the product of the lengths, then clamp floating-point rounding to
   the mathematical score range `[-1, 1]`. Return a new float without modifying
   either input vector.

For example, `[2, 0]` is rescaled to `[1, 0]`. The positive rescaling changes
its length but leaves the cosine unchanged. `math.fsum` adds floating-point
terms carefully; `math.sqrt` calculates the vector length from squared terms.

| Syntax | Role in this implementation |
| --- | --- |
| `Sequence[float]` | Describe an ordered collection of numeric coordinates, such as a list or tuple. Type hints document intent; runtime checks handle invalid values. |
| `all(math.isfinite(value) for value in vector)` | Check that every coordinate is finite. |
| `max(abs(value) for value in a)` | Find the largest coordinate magnitude to use as a positive scaling factor. |
| `[value / scale_a for value in a]` | Construct a new list containing the scaled coordinates. |
| `zip(scaled_a, scaled_b)` | Pair corresponding coordinates; the length check prevents silent truncation of mismatched vectors. |
| `math.fsum(x * y for x, y in zip(...))` | Sum coordinate products to calculate the dot product. |
| `math.sqrt(...)` | Take the square root when calculating a vector's length. |
| `candidate_vectors.items()` | Iterate over dictionary labels together with their vectors. |
| `{score:.4f}` | Format a floating-point result with four digits after the decimal point. |

This is a numeric comparison function, not a probability or a percentage. It
does not generate embeddings, automatically select relevant chunks, or generate
answers. The local embedding runner below supplies real model vectors to this
function. No API key or third-party library is needed for the math demonstration.

## Free local embeddings: connecting passages and questions

The default model is `sentence-transformers/all-MiniLM-L6-v2`, an Apache-2.0
licensed sentence encoder that produces 384-dimensional vectors. Its
[model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
describes the model and its input limit. The weights are downloaded once; the
model then runs on this Mac's CPU. There are no per-request API charges or key
requirements. ChatGPT subscription usage and OpenAI API billing are separate;
see [OpenAI pricing guidance](https://learn.chatgpt.com/docs/pricing).

### Setup, storage, and manual run

For a fresh checkout, install the local dependencies and download the model:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-embeddings.txt
.venv/bin/python embedding_demo.py --download-model --query "Why do chunks overlap?"
```

That setup has already completed in this workspace. For ordinary runs, use the
cached model without network access:

```bash
.venv/bin/python embedding_demo.py --query "Why do chunks overlap?"
```

The default is `local_files_only=True`; `--download-model` explicitly allows
fetching public model files. The model revision is pinned to
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The loader uses CPU, disables remote
custom code, and loads safetensors weights. The pinned dependency versions were
resolved on Python 3.12 / macOS arm64; the core `main.py` example still supports
Python 3.9, while the local model command should use the project environment.

Measured disk usage after setup:

| Component | Approximate disk usage |
| --- | --- |
| `.venv` with installed libraries | 1.1 GB |
| `.local/models` with model weights and configuration | 87 MB |
| Temporary pip download cache after cleanup | 4 KB |
| Total environment and model cache | About 1.2 GB |

The Mac had about 31 GB available after setup. The environment and caches are
ignored by Git and are not uploaded with the source. The model is loaded into
memory while the process runs; the figures above measure disk storage.

### Concept, intuition, and control flow

An embedding model converts text into a vector whose direction can be compared
with other text encoded by the same model. The question becomes a query vector;
each chunk becomes a passage vector. Both use this same MiniLM revision and
384-dimensional space. OpenAI's earlier 1,536-dimensional vectors are not used
in this local flow.

```text
load sample → clean → chunk with IDs → collect four passages + question
            → load cached MiniLM → encode all five texts locally
            → validate all vectors → compare query with each passage
            → print model, input texts, vector previews, and scores
```

The runner uses the same four 120-character chunks with 20-character overlap
shown in the earlier chunking example. The fifth input is the exact question
`Why do chunks overlap?`. Each run prints the full input text for inspection.

`load_local_model` loads the pinned model once for the process.
`embed_texts_locally(texts, model)` validates the inputs and checks token lengths
with truncation disabled. MiniLM's maximum input length is 256 tokens; an
oversized passage or question raises an error instead of being silently cut.
Character counts and token counts are different units.

The encoding call is:

```python
encoded = model.encode(
    list(texts),
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=False,
)
vectors = encoded.tolist()
```

`.encode` runs inference on the local CPU. `normalize_embeddings=True` scales
vectors to unit length. `.tolist()` converts the numeric array into ordinary
Python lists for our cosine function. The helper checks the batch count and
requires every vector to contain 384 finite numbers with a nonzero direction.
It rejects the entire result on failure and never substitutes invented vectors.
The original chunk IDs and source metadata remain attached to the displayed
passages. The demo displays all vector comparisons in source order, followed
by the selected passages ranked by similarity.

### Actual local input and output

The successful run used `data/sample.txt`, cleaned and split into these offsets:
`[0:120]`, `[100:220]`, `[200:320]`, and `[300:345]`. The query was
`Why do chunks overlap?`. The model produced five real 384-dimensional vectors.
The first five coordinates of the query vector were:

```text
[-0.028171665966510773, -0.041171178221702576, 0.0023428520653396845, -0.042692750692367554, 0.011299017816781998]
```

The resulting cosine scores were:

| Chunk | Offsets in cleaned text | Cosine similarity |
| --- | --- | --- |
| 0 | `[0:120]` | 0.3663 |
| 1 | `[100:220]` | 0.5839 |
| 2 | `[200:320]` | 0.3363 |
| 3 | `[300:345]` | 0.0270 |

Chunk 1 contains the explanation about repeating text and preserving context
across boundaries. It has the highest score for this example. These measured
scores are not probabilities or a general accuracy benchmark. The second run
loaded the cache without network access and reproduced the displayed scores.

## Retrieve the most similar passages

Retrieval selects document passages to use as context for a question. The local
embedding demo now performs this step after creating the vectors. It does not
generate an answer. `--top-k` controls the maximum number of passages returned;
its default is 2 and it must be a positive integer.

```bash
.venv/bin/python embedding_demo.py --query "Why do chunks overlap?" --top-k 2
```

The flow is: prepare the four sample chunks, embed them and the question with
the same cached MiniLM model, score every passage, sort scores from highest to
lowest, and return up to `top_k` passages. No new dependency, model download, or
paid request is needed for this feature.

`rank_chunks` in `rag/retrieval.py` accepts chunks, their corresponding vectors,
the query vector, and `top_k`. Each result is a `(chunk, score)` tuple. The chunk
is copied so changing the result does not change its original metadata.
The selection uses:

```python
ranked_chunks = sorted(scored_chunks, key=lambda result: result[1], reverse=True)
return ranked_chunks[:top_k]
```

`sorted` creates a new ordered list. `lambda result: result[1]` is a small
function that extracts the score (the tuple's second item). `reverse=True`
orders scores from highest to lowest. `[:top_k]` takes the first `top_k` items.
Equal scores keep their original input order. A limit larger than the collection
returns all available passages, and an empty collection returns no passages.

The helper checks that each chunk has one vector and validates the query and
every passage vector before selection. Invalid dimensions, zero vectors,
non-finite values, and incomplete batches raise errors instead of silently
dropping passages. As with any cosine comparison, the caller must supply
vectors from the same model; matching dimensions alone do not establish this.

Actual output from the command above, using 120-character chunks with
20-character overlap and five real 384-dimensional embeddings:

```text
Retrieved passages: 2 of 4 (top_k=2)

Rank 1: chunk 1 | cosine=0.5839
Offsets: [100:220]
Text: '.\n\nChunk overlap repeats some text between neighboring passages. It can preserve context when an idea crosses a chunk bo'

Rank 2: chunk 0 | cosine=0.3663
Offsets: [0:120]
Text: 'RAG study notes\n\nA RAG system retrieves relevant passages from documents before generating an answer.\n\nChunk overlap rep'
```

Each selected result also prints its complete source path and existing chunk ID.
To inspect selection manually, change `--top-k 2` to `--top-k 1`: only the first
ranked passage should remain. This feature scans all vectors in memory on every
run. There is no persistent vector index or relevance threshold yet. A top
result can still be unrelated when the document lacks an answer; a similarity
score is not a confidence percentage.

## Build source-labeled context

After retrieval, an answer model will need the selected passage text and a way
to cite its source. `build_context(matches)` in `rag/context.py` converts the
selected `(chunk, score)` pairs into one string containing `[S1]`, `[S2]`, and
subsequent evidence blocks in retrieval order. The runner now prints that
context after the ranked results. This step uses standard-library Python and
requires no new dependencies, downloads, or paid requests.

The flow is: load → clean → chunk → embed → rank → build labeled context →
display. Answer generation remains planned. The question is printed separately
by the runner; this function formats the retrieved evidence only.

The central Python syntax is:

```python
blocks = []
for number, (chunk, _score) in enumerate(matches, start=1):
    # Validate metadata, then assemble a block with f-strings.
    blocks.append(block)
return "\n\n".join(blocks)
```

`enumerate(..., start=1)` adds numbering beginning at one. The loop unpacks each
numbered `(chunk, score)` pair into `number`, `chunk`, and `_score`; the underscore
prefix indicates that the score is intentionally unused when formatting evidence.
An f-string such as `f"[S{number}]\n"` inserts the number and a newline. Adjacent
strings inside parentheses form one block. `.append(block)` collects each block,
and `"\n\n".join(blocks)` joins them with a blank line between passages.

The formatter requires nonblank text, source, and chunk ID, plus nonnegative
integer character offsets with an exclusive end. The offset span must equal the
passage's character length. Invalid metadata raises an error instead of producing
incomplete evidence. Text, Unicode, whitespace, and paragraph breaks are retained
exactly, and input dictionaries are unchanged. An empty selection returns `""`;
the runner reports that no context is available. Labels are local to the current
result: `[S1]` means the first retrieved passage, while the chunk ID retains its
existing identity.

Run the complete local example:

```bash
.venv/bin/python embedding_demo.py --query "Why do chunks overlap?" --top-k 2
```

The real cached-model check used `data/sample.txt`, 120-character chunks with
20-character overlap, five 384-dimensional embeddings, and the question above.
It selected chunk 1 (0.5839) then chunk 0 (0.3663). The new output was:

```text
Context for answer generation:
[S1]
Source: /Users/balakuthyaru/Documents/AI/rag-agent-systems/data/sample.txt
Chunk ID: fa85445b534fc63f802420b2ed3e8ed1b0ded208caade6879e74d99e922c2fe1
Offsets: [100:220]
Text:
.

Chunk overlap repeats some text between neighboring passages. It can preserve context when an idea crosses a chunk bo

[S2]
Source: /Users/balakuthyaru/Documents/AI/rag-agent-systems/data/sample.txt
Chunk ID: 5e2e42cefa2164ad1559073e11fab7800ecc8eb89fe0de1a5ec65589913271fc
Offsets: [0:120]
Text:
RAG study notes

A RAG system retrieves relevant passages from documents before generating an answer.

Chunk overlap rep
```

The partial words above come from the existing character-based chunk boundaries;
formatting preserves them. For manual review, run with `--top-k 1`: the final
context should contain `[S1]` only, with offsets `[100:220]` and the same text.
The context labels provide references for a future answer model; generated
answers and citation verification are still planned.

## Optional historical OpenAI example

The previous paid integration is retained separately in
`openai_embedding_demo.py`, `rag/embeddings.py`, and `requirements-openai.txt`.
It is not imported or called by the default local runner. Its key configuration
in `.env` is ignored by Git and is not read for local inference.

The earlier OpenAI check returned HTTP 429 with `credit_balance_exhausted`, so
it returned no real vectors. We switched to local inference to honor the user's
no-paid-usage preference. No further paid API requests will be made unless
explicitly requested. Historical API tests use simulated transports.

## Checks

```bash
python3 -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -v
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

Similarity checks cover known geometric scores, symmetry, invariance under
positive rescaling, very large and very small finite coordinates, invalid inputs,
unchanged input vectors, list/tuple support, and the bounded score range. All 32
tests and the runner were verified on Python 3.9.6 and 3.12.4 for this feature.

The embedding feature adds checks for response order, missing or duplicate
indexes, model and dimension mismatches, non-finite or zero vectors, blank input,
and propagated failures. SDK tests use simulated
HTTP responses and require neither an API key nor a network connection. They
also verify that exhausted-credit failures produce an actionable message,
expose no key or raw error body, and are not counted as successful embeddings.

Local embedding checks cover offline defaults, key-free execution, pinned model
loading, token limits without truncation, consistent dimensions, complete batches,
finite nonzero vectors, and preserved inputs. The real local model was also
run twice, including a cached run without network access.

Retrieval adds 12 tests covering descending scores, top-k limits, stable ties,
negative scores, empty collections, retained source metadata, unchanged inputs,
invalid vectors and incomplete batches, and CLI selection and validation. The
ranked example was also run with the real cached model and returned chunks 1 then 0.

Context formatting adds eight tests for retrieval-order labels and provenance,
verbatim Unicode and whitespace, empty selections, invalid or incomplete metadata,
inconsistent offsets, unchanged inputs, and CLI formatting of selected passages
only. All 82 tests pass in the project environment. On Python 3.9 and the
development shell's Python, 79 pass and three optional SDK tests are skipped.
The original local runner starts on both Python versions; the dependency-backed
example was run in the Python 3.12 environment with the real cached model.

## Project structure

```text
rag-agent-systems/
├── main.py                 # Runnable entry point
├── embedding_demo.py       # Free local embedding example
├── openai_embedding_demo.py # Optional paid API example, never run by default
├── requirements-embeddings.txt # Local model dependency pins
├── requirements-openai.txt # Optional API dependency pins
├── .env.example            # Placeholder for optional API configuration
├── rag/
│   ├── __init__.py
│   ├── chunking.py         # Overlapping chunks and stable passage IDs
│   ├── context.py          # Source-labeled text from retrieved passages
│   ├── documents.py        # Document loading
│   ├── embeddings.py       # Embedding requests and response validation
│   ├── local_embeddings.py # Local MiniLM loading and encoding
│   ├── preprocessing.py    # Prose whitespace cleanup
│   ├── retrieval.py        # Top-k passages ranked by cosine similarity
│   └── similarity.py       # Cosine similarity for numeric vectors
├── data/
│   └── sample.txt          # Sample input
├── tests/
│   ├── test_chunk_ids.py
│   ├── test_chunking.py
│   ├── test_context.py
│   ├── test_embeddings.py
│   ├── test_embedding_sdk.py
│   ├── test_local_embeddings.py
│   ├── test_preprocessing.py
│   ├── test_retrieval.py
│   └── test_similarity.py
└── README.md              # Implementation status and usage
```

## Planned work

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
- Added cosine similarity with numeric validation, safe rescaling, an explicit
  vector demonstration, and checks for geometric and numerical behavior.
- Added an optional OpenAI embedding adapter and runner with consistent model
  and dimensions, checked response mapping, isolated dependencies, and ignored
  key configuration. Live verification reached the API but is blocked by an
  exhausted API credit balance; the runner now reports this cause explicitly.
- Switched the default embedding command to free local MiniLM inference with
  a pinned model revision, offline caching, token-length validation, and no API
  credentials. Verified real vectors and scores and removed temporary package
  downloads to keep the total runtime environment around 1.2 GB on this Mac.
- Added top-k semantic retrieval with stable ranking, complete source metadata,
  validation of every candidate, and a configurable CLI limit. Verified the
  ranked output with the cached local model and checked both Python versions.
- Added source-labeled context from selected passages, preserving complete text
  and source metadata, with explicit empty results and validation of offsets.
  Verified the context with the real model and checked Python 3.9 compatibility.

The reference module mainly implements ingestion and low-level vector queries.
This project will also build the question-to-answer path and agent behavior,
with explicit failure handling and checks for each feature.
