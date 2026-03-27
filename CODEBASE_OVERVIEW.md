# Codebase Overview: Parts Holding Europe Financial RAG Pipeline

---

## 1. What It Does and How It Works

This pipeline answers English financial questions about Parts Holding Europe by reading French annual reports (PDFs from 2011–2023), extracting and translating the content, and retrieving the most relevant evidence before calling a language model to write the answer.

The core challenge it solves: the source documents are in French, but the questions and answers need to be in English. Rather than translate everything at query time, the pipeline pre-translates all text chunks into English and stores both versions — the English is used for search, the French is kept for audit citations.

**The end-to-end flow in plain language:**

1. PDFs are uploaded to a Databricks Volume folder, one subfolder per fiscal year (e.g. `.../2022/report.pdf`).
2. Notebook 02 scans those folders and registers every PDF in a tracking table (`document_inventory`), giving each file a stable ID based on its path.
3. Notebook 03 opens each PDF with PyMuPDF and extracts raw text page by page, storing it in `document_pages` in French.
4. Notebook 04 groups consecutive pages into overlapping chunks (roughly 2,500–6,000 chars each), then calls Databricks' `ai_translate()` to translate each chunk from French to English. Both the French original and English translation are kept.
5. Notebook 05 creates a Vector Search index over the English chunks using a Databricks-hosted embedding model. This index lets you find the most semantically relevant chunks for any question.
6. Notebook 06 runs a benchmark of 14 financial questions, retrieves the top chunks for each, and drafts answers — mainly to validate that the pipeline is working.
7. Notebook 07 is the interactive demo: a user types a question in English, the pipeline retrieves the best evidence chunks, and a language model writes a cited answer.
8. Notebook 08 is a specialised analysis that extracts off-balance-sheet liability disclosures across all years, builds a pivot table (categories × years), writes a narrative, and outputs the whole thing as LaTeX.

---

## 2. Key Architecture Decisions

### French source, English retrieval, French citations

The documents are French-language annual reports. Databricks does not offer a multilingual embedding model, so retrieval must happen over English text. Rather than translating at query time (slow, lossy, no audit trail), the pipeline pre-translates every chunk once and stores both languages. The LLM receives English evidence; the citation pointers always reference the original French pages.

### Deterministic SHA-based IDs everywhere

Every document, chunk, and page fingerprint is a SHA-256 hash of stable inputs (e.g. file path, page range, chunk content). This means:
- Re-running any notebook produces the same IDs.
- All writes are MERGE-based upserts, not appends, so re-runs are safe.
- Content changes are detected by comparing the stored SHA against the freshly computed one — if the French chunk text changed, the downstream translation and index statuses are automatically reset to `pending`.

### Status columns as the orchestration backbone

Each stage has its own `status` column: `parse_status`, `translate_status`, `index_status`. Each notebook filters on `WHERE status = 'pending'` and writes `status = 'done'` (or `error`) when finished. This means:
- Any failed document can be retried by setting its status back to `pending`.
- Stages are independent — chunking only processes fully-parsed documents, indexing only processes fully-translated chunks.
- There is no external orchestrator needed; the tables are the state machine.

### Keeping full text out of the Vector Search index

The Vector Search index stores only the embedding vector and a set of metadata columns (fiscal year, page range, file name, chunk ID). The full `chunk_text_en` and `chunk_text_fr` live only in the Delta table. After retrieval, the notebook does a second lookup to join the chunk IDs back to the Delta table to get the full text. This keeps the index small and fast, and means the LLM always gets the full untruncated evidence — not whatever fits in the VS metadata columns.

### Dynamic prompt budgeting

`databricks-gpt-oss-20b` has a ~4,096 token context window. The prompt builder calculates `chars_per_chunk = (max_prompt_chars - header_len) / num_chunks` so that the total prompt stays within budget regardless of how many chunks are retrieved. This prevents the silent NULL responses that plagued the original code.

---

## 3. Data Structure Design

### `document_inventory`

The central tracking table. One row per PDF file. Every downstream table joins back to this via `document_id`.

| Column | Purpose |
|--------|---------|
| `document_id` | SHA256(lower(file_path_dbfs)) — stable across re-uploads |
| `file_fingerprint` | SHA256(path \|\| size \|\| mtime) — detects file changes |
| `fiscal_year` | Extracted from folder path (`source=.../YYYY/`) |
| `parse_status` / `chunk_status` / `index_status` | Pipeline state machine — `pending` / `done` / `error` |
| `is_present` | False if file was deleted from Volume (history preserved) |

### `document_pages`

Raw French text, one row per page per document.

| Column | Purpose |
|--------|---------|
| `page_text_fr` | Raw PyMuPDF extraction for this page |
| `page_num` | 1-indexed page number |
| `page_char_count` | Size validation |

### `document_chunks`

The working corpus. Every retrieval operation ends up here.

| Column | Purpose |
|--------|---------|
| `chunk_id` | SHA256(document_id \|\| page_start \|\| page_end \|\| chunk_idx \|\| type) |
| `chunk_text_fr` | French original (audit / citation) |
| `chunk_text_en` | English translation (used for embedding and LLM prompts) |
| `chunk_sha_fr` | SHA256(chunk_text_fr) — detects translation invalidation |
| `page_start` / `page_end` | Stable page range for citations |
| `section_hint` | First detected section heading — heuristic |
| `translate_status` / `index_status` | Sub-stage tracking |

### `retrieval_eval_runs`

Stores benchmark results for reproducibility and debugging.

| Column | Purpose |
|--------|---------|
| `retrieved_json` | Full VS response + enriched chunks (JSON) |
| `answer_en` | LLM-drafted answer (NULL if generation failed) |
| `model_name` | Which model generated the answer |

### `pipeline_errors`

Every caught exception from every stage lands here with full stacktrace, enabling post-hoc debugging without re-running.

---

## 4. Key Algorithms

### Chunking: Sliding Window with Overlap

Pages are accumulated into chunks targeting 2,500–6,000 characters (French), up to 5 pages, with 1-page overlap between consecutive chunks.

```
i = 0
while i < len(pages):
    start = i
    accumulate pages until:
        - chars >= MIN_CHARS (2500), AND
        - next page would exceed MAX_CHARS (6000) or MAX_PAGES (5)
    emit chunk [start, end]
    advance: i = (end + 1) - OVERLAP_PAGES
```

Overlap ensures that a topic spanning a page boundary appears in two adjacent chunks, so retrieval is less sensitive to arbitrary page splits.

Each chunk is prefixed with `[PAGE N]` markers so the LLM and reader can always trace which page produced which sentence.

### Embedding: Databricks-hosted English encoder

The embedding model (either `databricks-bge-large-en` or `databricks-gte-large-en` depending on workspace) converts each `chunk_text_en` into a dense vector stored in the Delta Sync Vector Search index. At query time, the question text is embedded with the same model and cosine similarity is used to rank chunks.

Both models produce 1,024-dimensional vectors with a 512-token input limit. BGE has a marginal advantage on retrieval benchmarks (~0.5%), but the difference is small in practice.

### Prompting: Budget-constrained, citation-driven

The prompt structure is:

```
[System instructions — role, constraints, citation format]
QUESTION: <user question>
EVIDENCE:
[C1] FY{year} p{start}-{end} {filename}
{chunk_text_en trimmed to chars_per_chunk}

---

[C2] ...
```

The `chars_per_chunk` is calculated dynamically:
```
budget = max_prompt_chars - len(header)
chars_per_chunk = max(200, budget // num_chunks - 120)
```

The LLM is instructed to cite every claim using `[C1]`, `[C2]`, etc. and to say "evidence insufficient" rather than hallucinate. The citation numbers map directly to the ordered list printed below the answer.

For the off-balance-sheet study, a different prompt format is used: the LLM is asked to output `YYYY: <brief finding>` lines, which are then parsed with regex into `{year: finding}` dictionaries for the pivot table.

---

## 5. Why Results Differ Between the Two Laptops

You have confirmed chunk counts and indexed row counts are identical between the two workspaces. So the data pipeline is not the cause. The remaining explanations:

### Most likely: different embedding models

Your personal laptop uses `databricks-bge-large-en`; your company laptop uses `databricks-gte-large-en`. The Vector Search index on each workspace was built with a different embedding model. At query time, the question is also embedded by the model configured in that workspace's index.

The problem is not that one model is strictly better — both are competent. The difference is **domain fit for financial French-origin text**. BGE (BAAI General Embedding) was trained with explicit hard-negative mining on retrieval tasks and tends to produce tighter clusters for domain-specific terminology. For queries containing financial terms like "covenant", "securitization", "recourse obligation", BGE may return more targeted chunks while GTE (trained more broadly) returns chunks with surface-level keyword overlap but less semantic precision.

**Practical consequence**: GTE retrieves chunks that contain the right keywords (e.g. "garantie") but not necessarily the specific financial context of the question. The LLM then produces generic summaries because the evidence itself is generic.

### Secondary: query–index mismatch sensitivity

If the embedding model was changed after the index was already partially built (e.g. some chunks were indexed with one model, then the endpoint was swapped), cosine similarities in the index would be inconsistent. This would cause unpredictable retrieval — sometimes good, sometimes poor — depending on which fiscal year's chunks were indexed under which model. Check whether notebook 05 was run to full completion (`index_status = 'done'` for all chunks) after switching to GTE on the company laptop.

### Possible: `ai_translate` quality variation

If both workspaces ran notebook 04 independently, the `ai_translate()` calls may have produced slightly different English translations for the same French source text. Embedding quality is sensitive to translation quality — awkward phrasing from a less confident translation reduces cosine similarity against clean English questions.

### Unlikely but worth noting: cluster DBR version

If the two workspaces run different Databricks Runtime versions, the `ai_translate()` function, the Vector Search client, or the underlying embedding model checkpoint may differ. This is uncommon for production tenants but possible if one workspace is on an older LTS release.

**Recommended action**: On the company laptop, delete the Vector Search index and rebuild it from scratch using `05_vector_index.py` after confirming `databricks-gte-large-en` is available and all chunks have `translate_status = 'done'`. A partial or mixed-model index is the most plausible root cause of selective retrieval failure.
