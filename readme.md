# RepoBrain

**RepoBrain** is an AI-powered Repository Intelligence System designed to understand software repositories beyond simple text search.

It scans a repository, extracts code structure using AST analysis, resolves symbols and relationships, builds searchable code chunks, and retrieves relevant code using symbol search, BM25 lexical search, and semantic vector search.

The long-term goal is to build an **agentic repository intelligence system** capable of answering questions such as:

* Where is a feature implemented?
* Which function is responsible for a behavior?
* What calls this method?
* What depends on this module?
* How does a request flow through the codebase?
* What files may be affected by a change?

---

# Current Status

RepoBrain has currently completed:

```text
Phase 1      Repository Scanner                  ✅
Phase 2      Python AST Intelligence             ✅
Phase 2.5    Cross-File Symbol Resolution        ✅

Phase 3A     Symbol Search                       ✅
Phase 3B     BM25 Lexical Search                 ✅
Phase 3C     Semantic Vector Search              ✅
Phase 3C.1   Embedding Model Evaluation          ✅
Phase 3C.2   Semantic Code Representation        ✅
Phase 3C.3   Intent-Aware Semantic Reranking     ✅

Phase 4      Code Knowledge Graph                NEXT
```

Current automated test suite:

```text
122 tests passing
```

---

# Core Architecture

RepoBrain follows a deterministic-first architecture.

```text
Repository
    │
    ▼
Repository Scanner
    │
    ▼
Python AST Parser
    │
    ▼
Symbol + Relationship Extraction
    │
    ▼
Cross-File Symbol Resolution
    │
    ├───────────────┐
    │               │
    ▼               ▼
Code Chunks      Symbol Index
    │               │
    │               ▼
    │           Symbol Search
    │
    ├───────────────┐
    │               │
    ▼               ▼
BM25 Search     Semantic Documents
                    │
                    ▼
               CodeRankEmbed
                    │
                    ▼
                  FAISS
                    │
                    ▼
             Intent-Aware Reranker
```

Future phases will add:

```text
AST Relationships
       │
       ▼
Knowledge Graph
       │
       ▼
Graph Retrieval
       │
       ▼
Hybrid Retrieval

Symbol
+
BM25
+
Semantic
+
Graph
       │
       ▼
Agent Reasoning
```

---

# Design Principle

RepoBrain treats deterministic repository analysis as the source of truth.

```text
Source Code
    ↓
AST / Static Analysis
    ↓
Evidence
    ↓
Retrieval
    ↓
LLM Reasoning
```

The LLM is not responsible for inventing repository structure.

Repository facts should come from deterministic analysis whenever possible.

---

# Features

## Repository Scanner

RepoBrain can:

* recursively scan repositories
* detect programming languages
* ignore unnecessary directories
* ignore binary files
* identify test files
* identify configuration files
* detect generated files
* calculate SHA-256 file hashes
* generate deterministic repository IDs
* generate deterministic file IDs
* protect repository path boundaries
* isolate recoverable scanning errors

Common ignored directories include:

```text
.git
.venv
venv
node_modules
__pycache__
.pytest_cache
build
dist
*.egg-info
```

---

# Python AST Intelligence

RepoBrain currently extracts:

```text
MODULE
CLASS
FUNCTION
METHOD
```

It also detects relationships including:

```text
DEFINES
IMPORTS
CALLS
INHERITS
DECORATED_BY
```

Extracted symbol information includes:

* symbol name
* qualified name
* module
* source location
* signature
* docstring
* decorators
* async information
* parent symbol

---

# Cross-File Resolution

RepoBrain resolves many static relationships across Python files.

Currently supported examples include:

* same-module function calls
* same-module class references
* imported functions
* imported classes
* imported class methods
* `self.method()`
* `cls.method()`
* inheritance
* decorators
* direct qualified references

Resolution states are:

```text
EXACT
INFERRED
UNRESOLVED
```

RepoBrain only marks relationships as exact when static analysis can prove the target.

---

# Symbol Search

Symbol Search provides fast deterministic lookup for identifiers.

Example searches:

```text
RepositoryScanner
RepositoryScanner.scan
calculate_sha256
SemanticSearchEngine
PythonASTParser
```

It supports:

* exact matches
* qualified-name matches
* suffix matches
* prefix matches
* substring matches
* fuzzy fallback

---

# BM25 Lexical Search

RepoBrain contains an in-memory BM25 code-search engine.

The tokenizer understands identifiers such as:

```text
calculate_sha256
```

as concepts similar to:

```text
calculate
sha256
calculate_sha256
```

This improves lexical retrieval for source-code terminology.

---

# Semantic Search

RepoBrain uses:

```text
nomic-ai/CodeRankEmbed
```

for code-oriented embeddings.

Current embedding configuration:

```text
Embedding dimension : 768
Maximum sequence    : 512
Batch size          : 4
Vector index        : FAISS IndexFlatIP
Similarity          : normalized inner product
```

GPU acceleration is used for embedding generation when CUDA is available.

FAISS currently runs using the CPU index.

---

# Semantic Code Representation

Raw code alone was not sufficient for reliable natural-language-to-code retrieval.

RepoBrain therefore builds deterministic semantic documents using AST information.

A function such as:

```python
def _calculate_sha256(...):
    ...
```

may be represented to the embedding system using:

```text
Symbol
Qualified name
Symbol type
Identifier components
Signature
Docstring
Decorators
Known CALL relationships
Raw source code
```

No LLM-generated summaries are used during indexing.

---

# Intent-Aware Semantic Reranking

Semantic similarity alone can cause tests or documentation to rank above the actual implementation.

RepoBrain therefore classifies queries into simple retrieval intents:

```text
IMPLEMENTATION
DOCUMENTATION
DEFINITION
GENERAL
```

Repository sources are also classified as:

```text
PRODUCTION
TEST
DOCUMENTATION
CONFIG
SCRIPT
OTHER
```

For an implementation query such as:

```text
which function computes a stable digest of file contents?
```

RepoBrain favors production functions and methods while still keeping tests available as supporting evidence.

Example improvement:

```text
_calculate_sha256

Raw semantic ranking:
approximately #18

After semantic reranking:
#4
```

Another benchmark:

```text
where does RepoBrain prevent unsafe paths
from leaving the repository?
```

returns:

```text
RepositoryScanner._ensure_inside_repository
```

at rank:

```text
#1
```

For:

```text
how does the scanner avoid wasting time
walking folders we don't care about?
```

RepoBrain returns:

```text
RepositoryScanner._walk_repository
```

at:

```text
#1
```

---

# Project Structure

```text
repobrain/
│
├── repobrain/
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── sentence_transformer.py
│   │
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── resolver.py
│   │   └── symbol_index.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── filters.py
│   │   ├── language.py
│   │   └── scanner.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── repository.py
│   │   ├── retrieval.py
│   │   └── symbols.py
│   │
│   ├── parsing/
│   │   ├── __init__.py
│   │   └── python_ast.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── chunks.py
│   │   ├── lexical.py
│   │   ├── semantic.py
│   │   ├── semantic_document.py
│   │   ├── semantic_ranking.py
│   │   └── symbols.py
│   │
│   ├── __init__.py
│   └── config.py
│
├── scripts/
│   ├── compare_embedding_models.py
│   ├── inspect_python.py
│   ├── inspect_resolution.py
│   ├── inspect_semantic_search.py
│   └── inspect_symbol_search.py
│
├── tests/
│   ├── test_scanner.py
│   ├── test_python_ast.py
│   ├── test_symbol_index.py
│   ├── test_resolver.py
│   ├── test_symbol_search.py
│   ├── test_lexical_search.py
│   ├── test_semantic_search.py
│   ├── test_semantic_document.py
│   ├── test_semantic_ranking.py
│   └── test_embedding_config.py
│
├── requirements.txt
├── pyproject.toml
├── run.py
├── README.md
└── .gitignore
```

---

# Requirements

Recommended:

```text
Python 3.11+
```

Main dependencies:

```text
PyTorch
Sentence Transformers
Transformers
CodeRankEmbed
FAISS
NumPy
Pydantic
einops
pytest
```

---

# Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd repobrain
```

Create a virtual environment.

Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Install the appropriate PyTorch build for your system first.

For NVIDIA GPU users, use the installation command recommended by PyTorch for your CUDA configuration.

Then install RepoBrain dependencies:

```powershell
pip install -r requirements.txt
```

---

# Verify CUDA

Run:

```powershell
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Example:

```text
Torch: 2.14.0+cu130
CUDA: True
GPU: NVIDIA GeForce RTX 5060 Ti
```

CUDA is optional.

RepoBrain semantic embedding also works on CPU, although indexing will be slower.

---

# Run Tests

Run:

```powershell
python -m pytest
```

Current expected result:

```text
122 passed
```

---

# Run Repository Scanner

```powershell
python run.py .
```

You can replace `.` with another local repository:

```powershell
python run.py D:\path\to\repository
```

RepoBrain operates on the target repository in read-only analysis mode.

---

# Semantic Search

Example:

```powershell
python scripts\inspect_semantic_search.py . "where does RepoBrain prevent unsafe paths from leaving the repository?"
```

Example expected implementation:

```text
RepositoryScanner._ensure_inside_repository
```

---

## Search for File Hashing

```powershell
python scripts\inspect_semantic_search.py . "which function computes a stable digest of file contents?"
```

Relevant implementation:

```text
RepositoryScanner._calculate_sha256
```

---

## Search Directory Traversal

```powershell
python scripts\inspect_semantic_search.py . "how does the scanner avoid wasting time walking folders we don't care about?"
```

Relevant implementations include:

```text
RepositoryScanner._walk_repository
should_ignore_directory
```

---

# Disable Semantic Reranking

Phase 3C.3 reranking can be disabled for debugging:

```powershell
python scripts\inspect_semantic_search.py . "which function computes a stable digest of file contents?" --no-rerank
```

This allows comparison between:

```text
Raw FAISS similarity
```

and:

```text
Intent-aware RepoBrain ranking
```

---

# Retrieve More Results

```powershell
python scripts\inspect_semantic_search.py . "file hashing implementation" --top-k 50
```

---

# Current Limitations

RepoBrain is still under active development.

Current limitations include:

* Python is currently the primary AST-supported language.
* Semantic retrieval is currently in-memory.
* FAISS persistence has not yet been implemented.
* BM25 persistence has not yet been implemented.
* Semantic intent classification is deterministic and intentionally simple.
* Static analysis cannot resolve arbitrary runtime behavior.
* Dependency injection is not fully resolved.
* Factory-return type inference is not implemented.
* dynamic `getattr()` relationships are not statically resolved.
* runtime monkey-patching cannot be reliably inferred.
* graph-based repository navigation is not implemented yet.
* hybrid retrieval is not implemented yet.
* the final agent/orchestrator is not implemented yet.

These are intentionally deferred to later phases.

---

# Next Phase

The next major milestone is:

```text
Phase 4 — Code Knowledge Graph
```

Phase 4 will transform existing AST relationships into a navigable repository graph.

Planned capabilities include:

```text
symbol → callees
symbol → callers

module → imports
module → imported by

class → parents
class → children

symbol → decorators

file → symbols

symbol → neighboring repository evidence
```

This will enable RepoBrain to answer deeper questions such as:

```text
Who calls this function?

What functions does this method depend on?

How is this class connected to another class?

What code path leads to this implementation?

If I change this function, what may be affected?
```

After graph intelligence, RepoBrain will combine:

```text
Symbol Search
+
BM25
+
Semantic Search
+
Graph Retrieval
```

into a unified hybrid retrieval system.

---

# Vision

RepoBrain is not intended to be just another code search tool.

The goal is to build a system that can:

```text
Search repository
      ↓
Understand structure
      ↓
Follow relationships
      ↓
Gather evidence
      ↓
Reason about code
      ↓
Verify conclusions
      ↓
Answer with source-backed evidence
```

The final system will use an AI agent for reasoning and orchestration while keeping deterministic repository analysis as the underlying source of truth.

---

# License

Add the appropriate license for your project before public distribution.
