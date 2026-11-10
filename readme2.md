# RepoBrain

RepoBrain is a local repository intelligence system designed to understand, search, and reason about software repositories using deterministic static analysis, hybrid retrieval, a code knowledge graph, evidence assembly, agentic investigation, and grounded AI answering.

The deterministic repository-analysis layers remain the source of truth.

The language model does not decide repository structure on its own. It only explains evidence gathered by RepoBrain.

---

## Overview

RepoBrain combines multiple repository-understanding techniques into one pipeline.

```text
Repository
    ↓
Repository Scanner
    ↓
Python AST Analysis
    ↓
Symbol Resolution
    ↓
Symbol Search
    +
BM25 Lexical Search
    +
Semantic Search
    ↓
Code Knowledge Graph
    ↓
Graph Expansion
    ↓
Hybrid Retrieval
    ↓
Evidence Assembly
    ↓
Deterministic Agent Orchestrator
    ↓
Grounded Answer Generator
    ↓
Local Ollama
    ↓
Qwen2.5-Coder 14B