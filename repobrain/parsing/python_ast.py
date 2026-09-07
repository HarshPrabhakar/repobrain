from __future__ import annotations

import ast
import uuid

from pathlib import Path

from repobrain.models.repository import (
    FileMetadata,
    RepositoryScanResult,
)

from repobrain.models.symbols import (
    CodeRelationship,
    CodeSymbol,
    ParseError,
    PythonFileAnalysis,
    PythonRepositoryAnalysis,
    RelationshipType,
    ResolutionType,
    SymbolType,
)


class PythonASTParser:
    """
    Deterministic Python AST intelligence extractor.

    Responsibilities:
    - parse Python source
    - create module symbols
    - extract classes
    - extract functions
    - extract methods
    - extract imports
    - extract function/method calls
    - extract inheritance relationships
    - extract decorators

    This class does NOT:
    - perform semantic retrieval
    - build embeddings
    - build the code graph
    - invoke an LLM
    - perform full Python type inference
    """

    def parse_file(
        self,
        file_metadata: FileMetadata,
    ) -> PythonFileAnalysis:
        """
        Parse one Phase 1 Python file.
        """

        if file_metadata.language != "Python":
            raise ValueError(
                "PythonASTParser can only parse Python files. "
                f"Received: {file_metadata.relative_path}"
            )

        path = Path(file_metadata.absolute_path)

        source = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        return self.parse_source(
            source=source,
            file_metadata=file_metadata,
        )

    def parse_source(
        self,
        source: str,
        file_metadata: FileMetadata,
    ) -> PythonFileAnalysis:
        """
        Parse supplied Python source.

        Exposed separately so tests can use deterministic source fixtures.
        """

        tree = ast.parse(
            source,
            filename=file_metadata.relative_path,
        )

        module_name = self.module_name_from_path(
            file_metadata.relative_path
        )

        visitor = _PythonSymbolVisitor(
            file_metadata=file_metadata,
            module_name=module_name,
            source=source,
        )

        visitor.visit(tree)

        return PythonFileAnalysis(
            file_id=file_metadata.file_id,
            relative_path=file_metadata.relative_path,
            module=module_name,
            symbols=visitor.symbols,
            relationships=visitor.relationships,
        )

    @staticmethod
    def module_name_from_path(
        relative_path: str,
    ) -> str:
        """
        Convert repository-relative Python path to module-style notation.

        Examples:

        app/auth/service.py
        -> app.auth.service

        app/auth/__init__.py
        -> app.auth

        run.py
        -> run
        """

        path = Path(relative_path)

        parts = list(path.parts)

        if not parts:
            return ""

        filename = parts[-1]

        if filename == "__init__.py":
            module_parts = parts[:-1]

        else:
            stem = Path(filename).stem
            module_parts = [
                *parts[:-1],
                stem,
            ]

        return ".".join(module_parts)


class PythonRepositoryAnalyzer:
    """
    Runs PythonASTParser across a Phase 1 RepositoryScanResult.

    Syntax errors in individual files are collected and do not abort
    analysis of the remaining repository.
    """

    def __init__(
        self,
        parser: PythonASTParser | None = None,
    ) -> None:
        self.parser = parser or PythonASTParser()

    def analyze(
        self,
        scan_result: RepositoryScanResult,
    ) -> PythonRepositoryAnalysis:

        all_symbols: list[CodeSymbol] = []
        all_relationships: list[CodeRelationship] = []
        errors: list[ParseError] = []

        files_analyzed = 0

        for file_metadata in scan_result.files:

            if file_metadata.language != "Python":
                continue

            try:
                result = self.parser.parse_file(
                    file_metadata
                )

            except SyntaxError as exc:
                errors.append(
                    ParseError(
                        file_id=file_metadata.file_id,
                        relative_path=file_metadata.relative_path,
                        error_type="SyntaxError",
                        message=exc.msg,
                        line_number=exc.lineno,
                        column=exc.offset,
                    )
                )

                continue

            except Exception as exc:
                errors.append(
                    ParseError(
                        file_id=file_metadata.file_id,
                        relative_path=file_metadata.relative_path,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )

                continue

            files_analyzed += 1

            all_symbols.extend(
                result.symbols
            )

            all_relationships.extend(
                result.relationships
            )

        return PythonRepositoryAnalysis(
            repository_id=scan_result.metadata.repository_id,
            files_analyzed=files_analyzed,
            symbols=all_symbols,
            relationships=all_relationships,
            errors=errors,
        )


class _PythonSymbolVisitor(ast.NodeVisitor):
    """
    Internal AST visitor.

    One instance processes one Python file.
    """

    def __init__(
        self,
        file_metadata: FileMetadata,
        module_name: str,
        source: str,
    ) -> None:

        self.file_metadata = file_metadata

        self.repository_id = (
            file_metadata.repository_id
        )

        self.file_id = (
            file_metadata.file_id
        )

        self.module_name = module_name

        self.source = source

        self.symbols: list[CodeSymbol] = []

        self.relationships: list[
            CodeRelationship
        ] = []

        self.symbol_stack: list[
            CodeSymbol
        ] = []

        self.import_aliases: dict[
            str,
            str,
        ] = {}

        self.module_symbol = self._create_module_symbol()

        self.symbols.append(
            self.module_symbol
        )

        self.symbol_stack.append(
            self.module_symbol
        )

    # ---------------------------------------------------------
    # Module
    # ---------------------------------------------------------

    def _create_module_symbol(
        self,
    ) -> CodeSymbol:

        line_count = (
            self.file_metadata.line_count
            or 1
        )

        symbol_id = self._symbol_id(
            symbol_type=SymbolType.MODULE,
            qualified_name=self.module_name,
            start_line=1,
        )

        return CodeSymbol(
            symbol_id=symbol_id,

            repository_id=self.repository_id,
            file_id=self.file_id,

            name=(
                self.module_name.split(".")[-1]
                if self.module_name
                else self.file_metadata.file_name
            ),

            qualified_name=self.module_name,

            fully_qualified_name=self.module_name,

            symbol_type=SymbolType.MODULE,

            module=self.module_name,

            start_line=1,
            end_line=max(
                line_count,
                1,
            ),

            parent_symbol_id=None,
        )

    # ---------------------------------------------------------
    # Imports
    # ---------------------------------------------------------

    def visit_Import(
        self,
        node: ast.Import,
    ) -> None:

        for alias in node.names:

            local_name = (
                alias.asname
                or alias.name.split(".")[0]
            )

            self.import_aliases[
                local_name
            ] = alias.name

            self._add_relationship(
                relationship_type=RelationshipType.IMPORTS,

                source=self.module_symbol,

                target_qualified_name=alias.name,

                target_symbol_id=None,

                line_number=node.lineno,

                resolution=ResolutionType.EXACT,

                metadata={
                    "alias": alias.asname,
                    "import_type": "import",
                },
            )

        self.generic_visit(node)

    def visit_ImportFrom(
        self,
        node: ast.ImportFrom,
    ) -> None:

        module = self._resolve_relative_import_module(
            node
        )

        for alias in node.names:

            if alias.name == "*":
                target = (
                    f"{module}.*"
                    if module
                    else "*"
                )

                self._add_relationship(
                    relationship_type=RelationshipType.IMPORTS,

                    source=self.module_symbol,

                    target_qualified_name=target,

                    target_symbol_id=None,

                    line_number=node.lineno,

                    resolution=ResolutionType.UNRESOLVED,

                    metadata={
                        "alias": None,
                        "import_type": "from",
                        "star_import": True,
                    },
                )

                continue

            target = (
                f"{module}.{alias.name}"
                if module
                else alias.name
            )

            local_name = (
                alias.asname
                or alias.name
            )

            self.import_aliases[
                local_name
            ] = target

            self._add_relationship(
                relationship_type=RelationshipType.IMPORTS,

                source=self.module_symbol,

                target_qualified_name=target,

                target_symbol_id=None,

                line_number=node.lineno,

                resolution=ResolutionType.EXACT,

                metadata={
                    "alias": alias.asname,
                    "import_type": "from",
                },
            )

        self.generic_visit(node)

    # ---------------------------------------------------------
    # Classes
    # ---------------------------------------------------------

    def visit_ClassDef(
        self,
        node: ast.ClassDef,
    ) -> None:

        parent = self.symbol_stack[-1]

        qualified_name = self._child_qualified_name(
            parent=parent,
            child_name=node.name,
        )

        decorators = [
            self._expression_name(
                decorator
            )
            for decorator in node.decorator_list
        ]

        symbol = CodeSymbol(
            symbol_id=self._symbol_id(
                symbol_type=SymbolType.CLASS,
                qualified_name=qualified_name,
                start_line=node.lineno,
            ),

            repository_id=self.repository_id,
            file_id=self.file_id,

            name=node.name,
            qualified_name=qualified_name,

            fully_qualified_name=self._fully_qualified_name(
                qualified_name
            ),

            symbol_type=SymbolType.CLASS,

            module=self.module_name,

            start_line=node.lineno,
            end_line=self._end_line(node),

            parent_symbol_id=parent.symbol_id,

            signature=None,

            docstring=ast.get_docstring(
                node,
                clean=True,
            ),

            decorators=[
                item
                for item in decorators
                if item
            ],
        )

        self.symbols.append(symbol)

        self._add_relationship(
            relationship_type=RelationshipType.DEFINES,

            source=parent,

            target_qualified_name=symbol.qualified_name,

            target_symbol_id=symbol.symbol_id,

            line_number=node.lineno,

            resolution=ResolutionType.EXACT,
        )

        for base in node.bases:

            base_name = self._expression_name(
                base
            )

            if not base_name:
                continue

            resolved_name, resolution = (
                self._resolve_reference_name(
                    base_name
                )
            )

            self._add_relationship(
                relationship_type=RelationshipType.INHERITS,

                source=symbol,

                target_qualified_name=resolved_name,

                target_symbol_id=None,

                line_number=getattr(
                    base,
                    "lineno",
                    node.lineno,
                ),

                resolution=resolution,
            )

        for decorator in symbol.decorators:

            resolved_name, resolution = (
                self._resolve_reference_name(
                    decorator
                )
            )

            self._add_relationship(
                relationship_type=RelationshipType.DECORATED_BY,

                source=symbol,

                target_qualified_name=resolved_name,

                target_symbol_id=None,

                line_number=node.lineno,

                resolution=resolution,
            )

        self.symbol_stack.append(symbol)

        self.generic_visit(node)

        self.symbol_stack.pop()

    # ---------------------------------------------------------
    # Functions / methods
    # ---------------------------------------------------------

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef,
    ) -> None:

        self._visit_function(
            node=node,
            is_async=False,
        )

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:

        self._visit_function(
            node=node,
            is_async=True,
        )

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_async: bool,
    ) -> None:

        parent = self.symbol_stack[-1]

        if parent.symbol_type == SymbolType.CLASS:
            symbol_type = SymbolType.METHOD
        else:
            symbol_type = SymbolType.FUNCTION

        qualified_name = self._child_qualified_name(
            parent=parent,
            child_name=node.name,
        )

        decorators = [
            self._expression_name(
                decorator
            )
            for decorator in node.decorator_list
        ]

        symbol = CodeSymbol(
            symbol_id=self._symbol_id(
                symbol_type=symbol_type,
                qualified_name=qualified_name,
                start_line=node.lineno,
            ),

            repository_id=self.repository_id,
            file_id=self.file_id,

            name=node.name,
            qualified_name=qualified_name,

            fully_qualified_name=self._fully_qualified_name(
                qualified_name
            ),

            symbol_type=symbol_type,

            module=self.module_name,

            start_line=node.lineno,
            end_line=self._end_line(node),

            parent_symbol_id=parent.symbol_id,

            signature=self._function_signature(
                node
            ),

            docstring=ast.get_docstring(
                node,
                clean=True,
            ),

            is_async=is_async,

            decorators=[
                item
                for item in decorators
                if item
            ],
        )

        self.symbols.append(symbol)

        self._add_relationship(
            relationship_type=RelationshipType.DEFINES,

            source=parent,

            target_qualified_name=symbol.qualified_name,

            target_symbol_id=symbol.symbol_id,

            line_number=node.lineno,

            resolution=ResolutionType.EXACT,
        )

        for decorator in symbol.decorators:

            resolved_name, resolution = (
                self._resolve_reference_name(
                    decorator
                )
            )

            self._add_relationship(
                relationship_type=RelationshipType.DECORATED_BY,

                source=symbol,

                target_qualified_name=resolved_name,

                target_symbol_id=None,

                line_number=node.lineno,

                resolution=resolution,
            )

        self.symbol_stack.append(symbol)

        self.generic_visit(node)

        self.symbol_stack.pop()

    # ---------------------------------------------------------
    # Calls
    # ---------------------------------------------------------

    def visit_Call(
        self,
        node: ast.Call,
    ) -> None:

        source = self.symbol_stack[-1]

        # Calls occurring at module level are still useful.
        target_name = self._expression_name(
            node.func
        )

        if target_name:

            resolved_name, resolution = (
                self._resolve_reference_name(
                    target_name
                )
            )

            self._add_relationship(
                relationship_type=RelationshipType.CALLS,

                source=source,

                target_qualified_name=resolved_name,

                target_symbol_id=None,

                line_number=getattr(
                    node,
                    "lineno",
                    None,
                ),

                resolution=resolution,

                metadata={
                    "raw_target": target_name,
                },
            )

        else:

            self._add_relationship(
                relationship_type=RelationshipType.CALLS,

                source=source,

                target_qualified_name="<dynamic-call>",

                target_symbol_id=None,

                line_number=getattr(
                    node,
                    "lineno",
                    None,
                ),

                resolution=ResolutionType.UNRESOLVED,
            )

        self.generic_visit(node)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _resolve_relative_import_module(
        self,
        node: ast.ImportFrom,
    ) -> str:

        module = node.module or ""

        if node.level == 0:
            return module

        current_parts = (
            self.module_name.split(".")
            if self.module_name
            else []
        )

        # A Python module lives one level below its containing package.
        if self.file_metadata.file_name != "__init__.py":
            current_parts = current_parts[:-1]

        levels_up = max(
            node.level - 1,
            0,
        )

        if levels_up:
            current_parts = current_parts[
                : max(
                    len(current_parts) - levels_up,
                    0,
                )
            ]

        if module:
            current_parts.extend(
                module.split(".")
            )

        return ".".join(
            part
            for part in current_parts
            if part
        )

    def _resolve_reference_name(
        self,
        raw_name: str,
    ) -> tuple[str, ResolutionType]:

        if not raw_name:
            return (
                raw_name,
                ResolutionType.UNRESOLVED,
            )

        root, *remaining = raw_name.split(".")

        imported = self.import_aliases.get(
            root
        )

        if imported:

            if remaining:
                return (
                    ".".join(
                        [
                            imported,
                            *remaining,
                        ]
                    ),
                    ResolutionType.INFERRED,
                )

            return (
                imported,
                ResolutionType.EXACT,
            )

        if "." in raw_name:
            return (
                raw_name,
                ResolutionType.INFERRED,
            )

        return (
            raw_name,
            ResolutionType.UNRESOLVED,
        )

    @staticmethod
    def _expression_name(
        node: ast.AST,
    ) -> str | None:

        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Attribute):

            parent = _PythonSymbolVisitor._expression_name(
                node.value
            )

            if parent:
                return (
                    f"{parent}.{node.attr}"
                )

            return node.attr

        if isinstance(node, ast.Call):
            return (
                _PythonSymbolVisitor._expression_name(
                    node.func
                )
            )

        if isinstance(node, ast.Subscript):
            return (
                _PythonSymbolVisitor._expression_name(
                    node.value
                )
            )

        try:
            return ast.unparse(node)

        except Exception:
            return None

    @staticmethod
    def _function_signature(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> str:

        try:
            arguments = ast.unparse(
                node.args
            )

        except Exception:
            arguments = "..."

        prefix = (
            "async "
            if isinstance(
                node,
                ast.AsyncFunctionDef,
            )
            else ""
        )

        return (
            f"{prefix}{node.name}"
            f"({arguments})"
        )

    def _child_qualified_name(
        self,
        parent: CodeSymbol,
        child_name: str,
    ) -> str:

        if parent.symbol_type == SymbolType.MODULE:

            if parent.qualified_name:
                return (
                    f"{parent.qualified_name}."
                    f"{child_name}"
                )

            return child_name

        return (
            f"{parent.qualified_name}."
            f"{child_name}"
        )

    def _fully_qualified_name(
        self,
        qualified_name: str,
    ) -> str:

        return qualified_name

    @staticmethod
    def _end_line(
        node: ast.AST,
    ) -> int:

        return int(
            getattr(
                node,
                "end_lineno",
                getattr(
                    node,
                    "lineno",
                    1,
                ),
            )
        )

    def _symbol_id(
        self,
        symbol_type: SymbolType,
        qualified_name: str,
        start_line: int,
    ) -> str:

        value = (
            f"{self.repository_id}:"
            f"{self.file_id}:"
            f"{symbol_type.value}:"
            f"{qualified_name}:"
            f"{start_line}"
        )

        identifier = uuid.uuid5(
            uuid.NAMESPACE_URL,
            value,
        )

        return (
            f"sym_{identifier.hex[:16]}"
        )

    def _relationship_id(
        self,
        relationship_type: RelationshipType,
        source_symbol_id: str,
        target_qualified_name: str,
        line_number: int | None,
    ) -> str:

        value = (
            f"{self.repository_id}:"
            f"{self.file_id}:"
            f"{relationship_type.value}:"
            f"{source_symbol_id}:"
            f"{target_qualified_name}:"
            f"{line_number}"
        )

        identifier = uuid.uuid5(
            uuid.NAMESPACE_URL,
            value,
        )

        return (
            f"rel_{identifier.hex[:16]}"
        )

    def _add_relationship(
        self,
        relationship_type: RelationshipType,
        source: CodeSymbol,
        target_qualified_name: str,
        target_symbol_id: str | None,
        line_number: int | None,
        resolution: ResolutionType,
        metadata: dict | None = None,
    ) -> None:

        relationship_id = self._relationship_id(
            relationship_type=relationship_type,
            source_symbol_id=source.symbol_id,
            target_qualified_name=target_qualified_name,
            line_number=line_number,
        )

        relationship = CodeRelationship(
            relationship_id=relationship_id,

            repository_id=self.repository_id,
            file_id=self.file_id,

            relationship_type=relationship_type,

            source_symbol_id=source.symbol_id,

            target_symbol_id=target_symbol_id,

            source_qualified_name=(
                source.qualified_name
            ),

            target_qualified_name=(
                target_qualified_name
            ),

            line_number=line_number,

            resolution=resolution,

            metadata=metadata or {},
        )

        self.relationships.append(
            relationship
        )