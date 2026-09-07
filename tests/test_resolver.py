from __future__ import annotations

from datetime import datetime, timezone

from repobrain.indexing import (
    PythonSymbolResolver,
)

from repobrain.models.repository import (
    FileMetadata,
)

from repobrain.models.symbols import (
    PythonRepositoryAnalysis,
    RelationshipType,
    ResolutionType,
)

from repobrain.parsing import (
    PythonASTParser,
)


def make_metadata(
    relative_path: str,
    file_id: str,
) -> FileMetadata:

    return FileMetadata(
        file_id=file_id,

        repository_id="repo_test",

        relative_path=relative_path,

        absolute_path=relative_path,

        file_name=(
            relative_path.split("/")[-1]
        ),

        extension=".py",
        language="Python",

        size_bytes=100,
        line_count=100,

        content_hash="hash",

        is_test=False,
        is_config=False,
        is_generated=False,
        is_symlink=False,

        modified_at=datetime.now(
            timezone.utc
        ),
    )


def repository_analysis(
    sources: list[
        tuple[str, str, str]
    ],
) -> PythonRepositoryAnalysis:
    """
    sources:
        [(relative_path, file_id, source)]
    """

    parser = PythonASTParser()

    symbols = []
    relationships = []

    for (
        relative_path,
        file_id,
        source,
    ) in sources:

        metadata = make_metadata(
            relative_path=relative_path,
            file_id=file_id,
        )

        analysis = parser.parse_source(
            source=source,
            file_metadata=metadata,
        )

        symbols.extend(
            analysis.symbols
        )

        relationships.extend(
            analysis.relationships
        )

    return PythonRepositoryAnalysis(
        repository_id="repo_test",

        files_analyzed=len(sources),

        symbols=symbols,

        relationships=relationships,

        errors=[],
    )


def find_call(
    analysis: PythonRepositoryAnalysis,
    source_suffix: str,
    target_suffix: str,
):
    return next(
        relationship
        for relationship
        in analysis.relationships
        if (
            relationship.relationship_type
            == RelationshipType.CALLS
        )
        and relationship.source_qualified_name.endswith(
            source_suffix
        )
        and relationship.target_qualified_name.endswith(
            target_suffix
        )
    )


def test_resolves_self_method() -> None:

    analysis = repository_analysis(
        [
            (
                "app/service.py",
                "file_service",
                """
class Service:

    def execute(self):
        return self.validate()

    def validate(self):
        return True
""",
            )
        ]
    )

    resolver = PythonSymbolResolver()

    resolved, _ = (
        resolver.resolve_repository(
            analysis
        )
    )

    call = find_call(
        resolved,
        "Service.execute",
        "Service.validate",
    )

    assert (
        call.target_symbol_id
        is not None
    )

    assert (
        call.resolution
        == ResolutionType.EXACT
    )


def test_resolves_cls_method() -> None:

    analysis = repository_analysis(
        [
            (
                "app/service.py",
                "file_service",
                """
class Service:

    @classmethod
    def build(cls):
        return cls.create()

    @classmethod
    def create(cls):
        return cls()
""",
            )
        ]
    )

    resolver = PythonSymbolResolver()

    resolved, _ = (
        resolver.resolve_repository(
            analysis
        )
    )

    call = find_call(
        resolved,
        "Service.build",
        "Service.create",
    )

    assert (
        call.target_symbol_id
        is not None
    )


def test_resolves_same_module_function() -> None:

    analysis = repository_analysis(
        [
            (
                "app/auth.py",
                "file_auth",
                """
def authenticate():
    return True


def login():
    return authenticate()
""",
            )
        ]
    )

    resolver = PythonSymbolResolver()

    resolved, _ = (
        resolver.resolve_repository(
            analysis
        )
    )

    call = find_call(
        resolved,
        "login",
        "authenticate",
    )

    assert (
        call.target_qualified_name
        == "app.auth.authenticate"
    )

    assert (
        call.target_symbol_id
        is not None
    )


def test_resolves_same_module_class_constructor() -> None:

    analysis = repository_analysis(
        [
            (
                "app/service.py",
                "file_service",
                """
class Service:
    pass


def create():
    return Service()
""",
            )
        ]
    )

    resolver = PythonSymbolResolver()

    resolved, _ = (
        resolver.resolve_repository(
            analysis
        )
    )

    call = find_call(
        resolved,
        "create",
        "Service",
    )

    assert (
        call.target_symbol_id
        is not None
    )

    assert (
        call.target_qualified_name
        == "app.service.Service"
    )


def test_resolves_cross_file_imported_class() -> None:

    analysis = repository_analysis(
        [
            (
                "app/config.py",
                "file_config",
                """
class ScannerConfig:
    pass
""",
            ),
            (
                "app/main.py",
                "file_main",
                """
from app.config import ScannerConfig


def build():
    return ScannerConfig()
""",
            ),
        ]
    )

    resolver = PythonSymbolResolver()

    resolved, _ = (
        resolver.resolve_repository(
            analysis
        )
    )

    call = find_call(
        resolved,
        "build",
        "ScannerConfig",
    )

    assert (
        call.target_qualified_name
        == "app.config.ScannerConfig"
    )

    assert (
        call.target_symbol_id
        is not None
    )


def test_resolves_cross_file_imported_function() -> None:

    analysis = repository_analysis(
        [
            (
                "app/helpers.py",
                "file_helpers",
                """
def validate():
    return True
""",
            ),
            (
                "app/main.py",
                "file_main",
                """
from app.helpers import validate


def run():
    return validate()
""",
            ),
        ]
    )

    resolver = PythonSymbolResolver()

    resolved, _ = (
        resolver.resolve_repository(
            analysis
        )
    )

    call = find_call(
        resolved,
        "run",
        "validate",
    )

    assert (
        call.target_qualified_name
        == "app.helpers.validate"
    )

    assert (
        call.target_symbol_id
        is not None
    )


def test_resolves_imported_class_method() -> None:

    analysis = repository_analysis(
        [
            (
                "services/jwt.py",
                "file_jwt",
                """
class JWTService:

    @staticmethod
    def verify(token):
        return True
""",
            ),
            (
                "app/auth.py",
                "file_auth",
                """
from services.jwt import JWTService


def login(token):
    return JWTService.verify(token)
""",
            ),
        ]
    )

    resolver = PythonSymbolResolver()

    resolved, _ = (
        resolver.resolve_repository(
            analysis
        )
    )

    call = find_call(
        resolved,
        "login",
        "JWTService.verify",
    )

    assert (
        call.target_qualified_name
        == "services.jwt.JWTService.verify"
    )

    assert (
        call.target_symbol_id
        is not None
    )


def test_resolves_same_module_inheritance() -> None:

    analysis = repository_analysis(
        [
            (
                "app/service.py",
                "file_service",
                """
class BaseService:
    pass


class AuthService(BaseService):
    pass
""",
            )
        ]
    )

    resolver = PythonSymbolResolver()

    resolved, _ = (
        resolver.resolve_repository(
            analysis
        )
    )

    inheritance = next(
        relationship
        for relationship
        in resolved.relationships
        if (
            relationship.relationship_type
            == RelationshipType.INHERITS
        )
        and relationship.source_qualified_name.endswith(
            "AuthService"
        )
    )

    assert (
        inheritance.target_qualified_name
        == "app.service.BaseService"
    )

    assert (
        inheritance.target_symbol_id
        is not None
    )


def test_builtin_remains_unlinked() -> None:

    analysis = repository_analysis(
        [
            (
                "app/main.py",
                "file_main",
                """
def run():
    print("hello")
""",
            )
        ]
    )

    resolver = PythonSymbolResolver()

    resolved, _ = (
        resolver.resolve_repository(
            analysis
        )
    )

    call = find_call(
        resolved,
        "run",
        "print",
    )

    assert (
        call.target_symbol_id
        is None
    )


def test_external_symbol_remains_unlinked() -> None:

    analysis = repository_analysis(
        [
            (
                "app/main.py",
                "file_main",
                """
import requests


def load():
    return requests.get("https://example.com")
""",
            )
        ]
    )

    resolver = PythonSymbolResolver()

    resolved, _ = (
        resolver.resolve_repository(
            analysis
        )
    )

    call = find_call(
        resolved,
        "load",
        "requests.get",
    )

    assert (
        call.target_symbol_id
        is None
    )


def test_existing_defines_relationship_is_preserved() -> None:

    analysis = repository_analysis(
        [
            (
                "app/main.py",
                "file_main",
                """
def run():
    pass
""",
            )
        ]
    )

    defines_before = next(
        relationship
        for relationship
        in analysis.relationships
        if (
            relationship.relationship_type
            == RelationshipType.DEFINES
        )
    )

    resolver = PythonSymbolResolver()

    resolved, _ = (
        resolver.resolve_repository(
            analysis
        )
    )

    defines_after = next(
        relationship
        for relationship
        in resolved.relationships
        if (
            relationship.relationship_id
            == defines_before.relationship_id
        )
    )

    assert (
        defines_after.target_symbol_id
        == defines_before.target_symbol_id
    )


def test_summary_reports_newly_resolved() -> None:

    analysis = repository_analysis(
        [
            (
                "app/main.py",
                "file_main",
                """
def helper():
    return True


def run():
    return helper()
""",
            )
        ]
    )

    resolver = PythonSymbolResolver()

    _, summary = (
        resolver.resolve_repository(
            analysis
        )
    )

    assert (
        summary.newly_resolved
        >= 1
    )