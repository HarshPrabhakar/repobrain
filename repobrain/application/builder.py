from __future__ import annotations

import inspect
from pathlib import Path
from types import ModuleType
from typing import Callable

import repobrain.agent as agent_package
import repobrain.llm as llm_package

from repobrain.answering import GroundedAnswerGenerator
from repobrain.application.runtime import RepositoryRuntime
from repobrain.embeddings import SentenceTransformerEmbeddingProvider
from repobrain.evidence import EvidenceAssembler, EvidenceBudget
from repobrain.graph import RepositoryKnowledgeGraph
from repobrain.indexing import PythonSymbolResolver, SymbolIndex
from repobrain.ingestion import RepositoryScanner
from repobrain.parsing import PythonRepositoryAnalyzer
from repobrain.retrieval import (
    GraphEvidenceExpander,
    HybridRetrievalEngine,
    SemanticSearchEngine,
    SymbolSearchEngine,
)
from repobrain.retrieval.chunks import RepositoryChunkBuilder
from repobrain.retrieval.lexical import BM25Index


ProgressCallback = Callable[[int, int, str, dict[str, object]], None]


class RepositoryRuntimeBuilder:
    """
    Build one long-lived RepositoryRuntime from a local repository.

    Phase 10.2 moves the expensive repository-intelligence pipeline
    out of the CLI and into the application layer.

    The builder owns construction only. The resulting runtime owns
    the reusable agent runner and grounded answer generator.

    Conversation state is created later through:

        runtime.create_conversation()
    """

    TOTAL_STEPS = 12

    def __init__(
        self,
        *,
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5-coder:14b",
        temperature: float = 0.0,
        max_steps: int = 4,
        retrieval_top_k: int = 20,
        max_evidence_items: int = 8,
        max_characters: int = 24_000,
        max_graph_relations: int = 20,
        progress_callback: ProgressCallback | None = None,
    ) -> None:

        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        if retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be >= 1")

        if max_evidence_items < 1:
            raise ValueError("max_evidence_items must be >= 1")

        if max_characters < 1:
            raise ValueError("max_characters must be >= 1")

        if max_graph_relations < 1:
            raise ValueError("max_graph_relations must be >= 1")

        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        self.temperature = temperature
        self.max_steps = max_steps
        self.retrieval_top_k = retrieval_top_k
        self.max_evidence_items = max_evidence_items
        self.max_characters = max_characters
        self.max_graph_relations = max_graph_relations
        self.progress_callback = progress_callback

    # ==================================================================
    # Public API
    # ==================================================================

    def build(
        self,
        repository_root: Path,
    ) -> RepositoryRuntime:
        """
        Build the complete reusable RepoBrain runtime.

        The returned RepositoryRuntime can create multiple independent
        conversations without rebuilding repository intelligence.
        """

        repository_root = self._validate_repository_root(
            repository_root
        )

        # ==============================================================
        # 1. Repository scan
        # ==============================================================

        self._progress(
            1,
            "Scanning repository...",
        )

        scanner = RepositoryScanner()

        scan_result = scanner.scan(
            repository_root
        )

        self._progress(
            1,
            "Scanning repository...",
            files_discovered=len(scan_result.files),
        )

        # ==============================================================
        # 2. Python AST
        # ==============================================================

        self._progress(
            2,
            "Extracting Python AST...",
        )

        analyzer = PythonRepositoryAnalyzer()

        analysis = analyzer.analyze(
            scan_result
        )

        self._progress(
            2,
            "Extracting Python AST...",
            symbols=len(analysis.symbols),
            relationships=len(analysis.relationships),
        )

        # ==============================================================
        # 3. Symbol resolution
        # ==============================================================

        self._progress(
            3,
            "Resolving repository symbols...",
        )

        resolver = PythonSymbolResolver()

        (
            resolved_analysis,
            resolution_summary,
        ) = resolver.resolve_repository(
            analysis
        )

        _ = resolution_summary

        self._progress(
            3,
            "Resolving repository symbols...",
            resolved_symbols=len(
                resolved_analysis.symbols
            ),
        )

        # ==============================================================
        # 4. Repository chunks
        # ==============================================================

        self._progress(
            4,
            "Building repository chunks...",
        )

        chunk_builder = RepositoryChunkBuilder()

        chunks = chunk_builder.build(
            scan_result=scan_result,
            analysis=resolved_analysis,
        )

        self._progress(
            4,
            "Building repository chunks...",
            chunks=len(chunks),
        )

        # ==============================================================
        # 5. Symbol index
        # ==============================================================

        self._progress(
            5,
            "Building symbol index...",
        )

        symbol_index = SymbolIndex(
            resolved_analysis.symbols
        )

        symbol_engine = SymbolSearchEngine(
            symbol_index
        )

        self._progress(
            5,
            "Building symbol index...",
            indexed_symbols=len(
                resolved_analysis.symbols
            ),
        )

        # ==============================================================
        # 6. BM25
        # ==============================================================

        self._progress(
            6,
            "Building BM25 index...",
        )

        lexical_engine = BM25Index(
            chunks
        )

        self._progress(
            6,
            "Building BM25 index...",
            bm25_documents=len(chunks),
        )

        # ==============================================================
        # 7. Semantic retrieval
        # ==============================================================

        self._progress(
            7,
            "Loading semantic model...",
        )

        embedding_provider = (
            SentenceTransformerEmbeddingProvider()
        )

        semantic_engine = SemanticSearchEngine(
            chunks,
            embedding_provider,
            symbols=resolved_analysis.symbols,
            relationships=(
                resolved_analysis.relationships
            ),
        )

        self._progress(
            7,
            "Loading semantic model...",
            embedding_model=(
                embedding_provider.model_name
            ),
            device=(
                embedding_provider.device
            ),
            dimension=(
                embedding_provider.dimension
            ),
            semantic_index="READY",
        )

        # ==============================================================
        # 8. Knowledge graph
        # ==============================================================

        self._progress(
            8,
            "Building knowledge graph...",
        )

        graph = RepositoryKnowledgeGraph(
            symbols=resolved_analysis.symbols,
            relationships=(
                resolved_analysis.relationships
            ),
        )

        graph_stats = graph.stats()

        node_count = self._first_existing_attribute(
            graph_stats,
            "node_count",
            "nodes",
            "total_nodes",
            default=len(
                resolved_analysis.symbols
            ),
        )

        edge_count = self._first_existing_attribute(
            graph_stats,
            "edge_count",
            "edges",
            "total_edges",
            default=0,
        )

        self._progress(
            8,
            "Building knowledge graph...",
            graph_nodes=node_count,
            graph_edges=edge_count,
        )

        # ==============================================================
        # 9. Hybrid retrieval
        # ==============================================================

        self._progress(
            9,
            "Building hybrid retrieval...",
        )

        graph_expander = (
            self._construct_from_supported_kwargs(
                GraphEvidenceExpander,
                available={
                    "graph": graph,
                    "knowledge_graph": graph,
                    "repository_graph": graph,
                    "symbol_index": symbol_index,
                },
            )
        )

        hybrid_engine = HybridRetrievalEngine(
            symbol_engine=symbol_engine,
            lexical_engine=lexical_engine,
            semantic_engine=semantic_engine,
            graph_expander=graph_expander,
        )

        self._progress(
            9,
            "Building hybrid retrieval...",
            hybrid_engine="READY",
        )

        # ==============================================================
        # 10. Evidence assembler
        # ==============================================================

        self._progress(
            10,
            "Building evidence assembler...",
        )

        evidence_budget = EvidenceBudget(
            max_items=self.max_evidence_items,
            max_characters=self.max_characters,
        )

        evidence_assembler = (
            self._construct_from_supported_kwargs(
                EvidenceAssembler,
                available={
                    "budget": evidence_budget,
                    "chunks": chunks,
                    "graph": graph,
                    "knowledge_graph": graph,
                    "symbol_index": symbol_index,
                    "symbols": (
                        resolved_analysis.symbols
                    ),
                    "relationships": (
                        resolved_analysis.relationships
                    ),
                },
            )
        )

        self._progress(
            10,
            "Building evidence assembler...",
            evidence_layer="READY",
        )

        # ==============================================================
        # 11. Deterministic agent
        # ==============================================================

        self._progress(
            11,
            "Building deterministic agent...",
        )

        agent_tools = self._build_agent_tools(
            hybrid_engine=hybrid_engine,
            graph=graph,
            evidence_assembler=evidence_assembler,
            symbol_index=symbol_index,
            chunks=chunks,
            resolved_analysis=resolved_analysis,
        )

        base_agent = self._build_base_agent(
            tools=agent_tools,
            hybrid_engine=hybrid_engine,
            graph=graph,
            evidence_assembler=evidence_assembler,
            symbol_index=symbol_index,
        )

        agent_run = getattr(
            base_agent,
            "run",
            None,
        )

        if not callable(agent_run):
            raise RuntimeError(
                f"{type(base_agent).__name__} "
                "does not expose a callable run(query) method."
            )

        self._progress(
            11,
            "Building deterministic agent...",
            agent=type(base_agent).__name__,
            stateful_agent="READY",
        )

        # ==============================================================
        # 12. Grounded answer generator
        # ==============================================================

        self._progress(
            12,
            "Loading grounded answer layer...",
        )

        llm_provider = (
            self._build_ollama_provider()
        )

        answer_generator = (
            GroundedAnswerGenerator(
                provider=llm_provider
            )
        )

        provider_model = (
            self._first_existing_attribute(
                llm_provider,
                "model_name",
                "model",
                default=self.ollama_model,
            )
        )

        self._progress(
            12,
            "Loading grounded answer layer...",
            llm_model=provider_model,
            grounded_qa="READY",
        )

        return RepositoryRuntime(
            repository_root=repository_root,
            agent_runner=agent_run,
            answer_generator=answer_generator,
        )

    # ==================================================================
    # Agent construction
    # ==================================================================

    def _build_agent_tools(
        self,
        *,
        hybrid_engine: HybridRetrievalEngine,
        graph: RepositoryKnowledgeGraph,
        evidence_assembler: EvidenceAssembler,
        symbol_index: SymbolIndex,
        chunks: list[object],
        resolved_analysis: object,
    ) -> object:

        tools_class = self._find_class(
            agent_package,
            preferred_names=(
                "AgentToolbox",
                "RepoBrainAgentTools",
                "AgentTools",
                "RepositoryAgentTools",
            ),
            name_contains=(
                "tool",
            ),
        )

        return self._construct_from_supported_kwargs(
            tools_class,
            available={
                "hybrid_engine": hybrid_engine,
                "hybrid_retriever": hybrid_engine,
                "retrieval_engine": hybrid_engine,
                "retriever": hybrid_engine,
                "graph": graph,
                "knowledge_graph": graph,
                "repository_graph": graph,
                "evidence_assembler": evidence_assembler,
                "assembler": evidence_assembler,
                "symbol_index": symbol_index,
                "index": symbol_index,
                "chunks": chunks,
                "symbols": (
                    resolved_analysis.symbols
                ),
                "relationships": (
                    resolved_analysis.relationships
                ),
            },
        )

    def _build_base_agent(
        self,
        *,
        tools: object,
        hybrid_engine: HybridRetrievalEngine,
        graph: RepositoryKnowledgeGraph,
        evidence_assembler: EvidenceAssembler,
        symbol_index: SymbolIndex,
    ) -> object:

        orchestrator_class = self._find_class(
            agent_package,
            preferred_names=(
                "RepoBrainAgentOrchestrator",
                "AgentOrchestrator",
                "DeterministicRepoBrainAgent",
                "RepoBrainAgent",
            ),
            name_contains=(
                "orchestrator",
            ),
        )

        return self._construct_from_supported_kwargs(
            orchestrator_class,
            available={
                "toolbox": tools,
                "tools": tools,
                "agent_tools": tools,
                "hybrid_engine": hybrid_engine,
                "hybrid_retriever": hybrid_engine,
                "retrieval_engine": hybrid_engine,
                "graph": graph,
                "knowledge_graph": graph,
                "evidence_assembler": evidence_assembler,
                "assembler": evidence_assembler,
                "symbol_index": symbol_index,
                "max_steps": self.max_steps,
                "max_agent_steps": self.max_steps,
                "retrieval_top_k": (
                    self.retrieval_top_k
                ),
                "top_k": self.retrieval_top_k,
                "max_graph_relations": (
                    self.max_graph_relations
                ),
            },
        )

    # ==================================================================
    # Ollama construction
    # ==================================================================

    def _build_ollama_provider(
        self,
    ) -> object:

        provider_class = self._find_class(
            llm_package,
            preferred_names=(
                "OllamaLLMProvider",
            ),
            name_contains=(
                "ollama",
                "provider",
            ),
        )

        return self._construct_from_supported_kwargs(
            provider_class,
            available={
                "host": self.ollama_host,
                "base_url": self.ollama_host,
                "ollama_host": self.ollama_host,
                "url": self.ollama_host,
                "model": self.ollama_model,
                "model_name": self.ollama_model,
                "temperature": self.temperature,
            },
        )

    # ==================================================================
    # Progress
    # ==================================================================

    def _progress(
        self,
        step: int,
        message: str,
        **details: object,
    ) -> None:

        if self.progress_callback is None:
            return

        self.progress_callback(
            step,
            self.TOTAL_STEPS,
            message,
            details,
        )

    # ==================================================================
    # Compatibility helpers
    # ==================================================================

    @staticmethod
    def _find_class(
        module: ModuleType,
        *,
        preferred_names: tuple[str, ...],
        name_contains: tuple[str, ...] = (),
    ) -> type:

        for name in preferred_names:

            value = getattr(
                module,
                name,
                None,
            )

            if inspect.isclass(
                value
            ):
                return value

        candidates: list[type] = []

        for name in dir(
            module
        ):

            value = getattr(
                module,
                name,
            )

            if not inspect.isclass(
                value
            ):
                continue

            lowered = name.lower()

            if all(
                token.lower()
                in lowered
                for token in name_contains
            ):

                candidates.append(
                    value
                )

        if len(candidates) == 1:
            return candidates[0]

        exported = [
            name
            for name in dir(module)
            if not name.startswith("_")
        ]

        raise RuntimeError(
            "Could not determine the required RepoBrain class.\n"
            f"Module: {module.__name__}\n"
            f"Preferred names: {preferred_names}\n"
            f"Available exports: {exported}"
        )

    @staticmethod
    def _construct_from_supported_kwargs(
        cls: type,
        *,
        available: dict[str, object],
    ) -> object:

        signature = inspect.signature(
            cls
        )

        kwargs: dict[str, object] = {}

        missing_required: list[str] = []

        for (
            name,
            parameter,
        ) in signature.parameters.items():

            if name in {
                "self",
                "cls",
            }:
                continue

            if (
                parameter.kind
                in {
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                }
            ):
                continue

            if name in available:

                kwargs[name] = (
                    available[name]
                )

                continue

            if (
                parameter.default
                is inspect.Parameter.empty
            ):

                missing_required.append(
                    name
                )

        if missing_required:

            raise RuntimeError(
                f"Cannot construct "
                f"{cls.__module__}.{cls.__name__}.\n"
                "Unsupported required constructor parameter(s): "
                f"{', '.join(missing_required)}\n"
                f"Signature: {signature}"
            )

        return cls(
            **kwargs
        )

    @staticmethod
    def _first_existing_attribute(
        obj: object,
        *names: str,
        default: object = None,
    ) -> object:

        for name in names:

            if hasattr(
                obj,
                name,
            ):

                return getattr(
                    obj,
                    name,
                )

        return default

    # ==================================================================
    # Validation
    # ==================================================================

    @staticmethod
    def _validate_repository_root(
        repository_root: Path,
    ) -> Path:

        normalized_root = (
            repository_root
            .expanduser()
            .resolve()
        )

        if not normalized_root.exists():

            raise ValueError(
                "repository_root does not exist: "
                f"{normalized_root}"
            )

        if not normalized_root.is_dir():

            raise ValueError(
                "repository_root is not a directory: "
                f"{normalized_root}"
            )

        return normalized_root
