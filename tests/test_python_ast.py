from __future__ import annotations

from datetime import datetime, timezone

import pytest

from repobrain.models.repository import (
    FileMetadata,
)

from repobrain.models.symbols import (
    RelationshipType,
    ResolutionType,
    SymbolType,
)

from repobrain.parsing.python_ast import (
    PythonASTParser,
)


def make_file_metadata(
    relative_path: str = "app/auth/service.py",
    line_count: int = 100,
) -> FileMetadata:

    now = datetime.now(
        timezone.utc
    )

    return FileMetadata(
        file_id="file_test123",
        repository_id="repo_test123",

        relative_path=relative_path,
        absolute_path=relative_path,

        file_name=relative_path.split("/")[-1],

        extension=".py",
        language="Python",

        size_bytes=100,
        line_count=line_count,

        content_hash="abc",

        is_test=False,
        is_config=False,
        is_generated=False,
        is_symlink=False,

        modified_at=now,
    )


def test_module_name_from_python_path() -> None:

    assert (
        PythonASTParser.module_name_from_path(
            "app/auth/service.py"
        )
        == "app.auth.service"
    )


def test_module_name_from_init_file() -> None:

    assert (
        PythonASTParser.module_name_from_path(
            "app/auth/__init__.py"
        )
        == "app.auth"
    )


def test_extracts_module_symbol() -> None:

    parser = PythonASTParser()

    metadata = make_file_metadata()

    result = parser.parse_source(
        source="x = 1\n",
        file_metadata=metadata,
    )

    modules = [
        symbol
        for symbol in result.symbols
        if symbol.symbol_type
        == SymbolType.MODULE
    ]

    assert len(modules) == 1

    assert (
        modules[0].qualified_name
        == "app.auth.service"
    )


def test_extracts_class() -> None:

    source = """
class AuthService:
    pass
"""

    parser = PythonASTParser()

    result = parser.parse_source(
        source=source,
        file_metadata=make_file_metadata(),
    )

    classes = [
        symbol
        for symbol in result.symbols
        if symbol.symbol_type
        == SymbolType.CLASS
    ]

    assert len(classes) == 1

    assert (
        classes[0].qualified_name
        == "app.auth.service.AuthService"
    )


def test_extracts_function() -> None:

    source = """
def verify_token(token):
    return token
"""

    parser = PythonASTParser()

    result = parser.parse_source(
        source=source,
        file_metadata=make_file_metadata(),
    )

    functions = [
        symbol
        for symbol in result.symbols
        if symbol.symbol_type
        == SymbolType.FUNCTION
    ]

    assert len(functions) == 1

    assert (
        functions[0].qualified_name
        == "app.auth.service.verify_token"
    )


def test_extracts_method() -> None:

    source = """
class AuthService:

    def verify_token(self, token):
        return token
"""

    parser = PythonASTParser()

    result = parser.parse_source(
        source=source,
        file_metadata=make_file_metadata(),
    )

    methods = [
        symbol
        for symbol in result.symbols
        if symbol.symbol_type
        == SymbolType.METHOD
    ]

    assert len(methods) == 1

    assert (
        methods[0].qualified_name
        == "app.auth.service.AuthService.verify_token"
    )


def test_extracts_async_function() -> None:

    source = """
async def fetch_user(user_id):
    return user_id
"""

    parser = PythonASTParser()

    result = parser.parse_source(
        source=source,
        file_metadata=make_file_metadata(),
    )

    function = next(
        symbol
        for symbol in result.symbols
        if symbol.symbol_type
        == SymbolType.FUNCTION
    )

    assert function.is_async is True

    assert function.signature.startswith(
        "async fetch_user("
    )


def test_extracts_docstring() -> None:

    source = '''
def verify_token(token):
    """Validate a JWT token."""
    return token
'''

    parser = PythonASTParser()

    result = parser.parse_source(
        source=source,
        file_metadata=make_file_metadata(),
    )

    function = next(
        symbol
        for symbol in result.symbols
        if symbol.symbol_type
        == SymbolType.FUNCTION
    )

    assert (
        function.docstring
        == "Validate a JWT token."
    )


def test_creates_defines_relationship() -> None:

    source = """
class AuthService:
    pass
"""

    parser = PythonASTParser()

    result = parser.parse_source(
        source=source,
        file_metadata=make_file_metadata(),
    )

    relationships = [
        relationship
        for relationship in result.relationships
        if relationship.relationship_type
        == RelationshipType.DEFINES
    ]

    assert len(relationships) == 1

    assert (
        relationships[0].target_qualified_name
        == "app.auth.service.AuthService"
    )

    assert (
        relationships[0].resolution
        == ResolutionType.EXACT
    )


def test_extracts_import() -> None:

    source = """
import json
"""

    parser = PythonASTParser()

    result = parser.parse_source(
        source=source,
        file_metadata=make_file_metadata(),
    )

    imports = [
        relationship
        for relationship in result.relationships
        if relationship.relationship_type
        == RelationshipType.IMPORTS
    ]

    assert len(imports) == 1

    assert (
        imports[0].target_qualified_name
        == "json"
    )


def test_extracts_from_import() -> None:

    source = """
from services.jwt import JWTService
"""

    parser = PythonASTParser()

    result = parser.parse_source(
        source=source,
        file_metadata=make_file_metadata(),
    )

    imports = [
        relationship
        for relationship in result.relationships
        if relationship.relationship_type
        == RelationshipType.IMPORTS
    ]

    assert len(imports) == 1

    assert (
        imports[0].target_qualified_name
        == "services.jwt.JWTService"
    )


def test_import_alias_resolves_call() -> None:

    source = """
from services.jwt import JWTService as JWT

def verify(token):
    return JWT.verify(token)
"""

    parser = PythonASTParser()

    result = parser.parse_source(
        source=source,
        file_metadata=make_file_metadata(),
    )

    calls = [
        relationship
        for relationship in result.relationships
        if relationship.relationship_type
        == RelationshipType.CALLS
    ]

    call = next(
        relationship
        for relationship in calls
        if relationship.target_qualified_name
        == "services.jwt.JWTService.verify"
    )

    assert (
        call.resolution
        == ResolutionType.INFERRED
    )


def test_extracts_call_relationship() -> None:

    source = """
def login():
    return authenticate()
"""

    parser = PythonASTParser()

    result = parser.parse_source(
        source=source,
        file_metadata=make_file_metadata(),
    )

    calls = [
        relationship
        for relationship in result.relationships
        if relationship.relationship_type
        == RelationshipType.CALLS
    ]

    assert len(calls) == 1

    assert (
        calls[0].target_qualified_name
        == "authenticate"
    )


def test_extracts_inheritance() -> None:

    source = """
class BaseService:
    pass

class AuthService(BaseService):
    pass
"""

    parser = PythonASTParser()

    result = parser.parse_source(
        source=source,
        file_metadata=make_file_metadata(),
    )

    inheritance = [
        relationship
        for relationship in result.relationships
        if relationship.relationship_type
        == RelationshipType.INHERITS
    ]

    assert len(inheritance) == 1

    assert (
        inheritance[0].target_qualified_name
        == "BaseService"
    )


def test_extracts_decorator() -> None:

    source = """
@staticmethod
def verify():
    pass
"""

    parser = PythonASTParser()

    result = parser.parse_source(
        source=source,
        file_metadata=make_file_metadata(),
    )

    decorated = [
        relationship
        for relationship in result.relationships
        if relationship.relationship_type
        == RelationshipType.DECORATED_BY
    ]

    assert len(decorated) == 1

    assert (
        decorated[0].target_qualified_name
        == "staticmethod"
    )


def test_method_defines_relationship() -> None:

    source = """
class AuthService:

    def verify(self):
        return True
"""

    parser = PythonASTParser()

    result = parser.parse_source(
        source=source,
        file_metadata=make_file_metadata(),
    )

    defines = [
        relationship
        for relationship in result.relationships
        if relationship.relationship_type
        == RelationshipType.DEFINES
    ]

    assert len(defines) == 2

    method_relation = next(
        relationship
        for relationship in defines
        if relationship.target_qualified_name.endswith(
            ".verify"
        )
    )

    assert (
        method_relation.source_qualified_name
        == "app.auth.service.AuthService"
    )


def test_symbol_ids_are_deterministic() -> None:

    source = """
def verify(token):
    return token
"""

    parser = PythonASTParser()

    metadata = make_file_metadata()

    first = parser.parse_source(
        source=source,
        file_metadata=metadata,
    )

    second = parser.parse_source(
        source=source,
        file_metadata=metadata,
    )

    first_ids = [
        symbol.symbol_id
        for symbol in first.symbols
    ]

    second_ids = [
        symbol.symbol_id
        for symbol in second.symbols
    ]

    assert first_ids == second_ids


def test_relationship_ids_are_deterministic() -> None:

    source = """
def login():
    authenticate()
"""

    parser = PythonASTParser()

    metadata = make_file_metadata()

    first = parser.parse_source(
        source=source,
        file_metadata=metadata,
    )

    second = parser.parse_source(
        source=source,
        file_metadata=metadata,
    )

    first_ids = [
        relationship.relationship_id
        for relationship in first.relationships
    ]

    second_ids = [
        relationship.relationship_id
        for relationship in second.relationships
    ]

    assert first_ids == second_ids


def test_invalid_python_raises_syntax_error() -> None:

    parser = PythonASTParser()

    source = """
def broken(
"""

    with pytest.raises(SyntaxError):

        parser.parse_source(
            source=source,
            file_metadata=make_file_metadata(),
        )


def test_nested_function_is_function_not_method() -> None:

    source = """
class Service:

    def outer(self):

        def inner():
            return True

        return inner()
"""

    parser = PythonASTParser()

    result = parser.parse_source(
        source=source,
        file_metadata=make_file_metadata(),
    )

    inner = next(
        symbol
        for symbol in result.symbols
        if symbol.name == "inner"
    )

    assert (
        inner.symbol_type
        == SymbolType.FUNCTION
    )


def test_relative_import() -> None:

    source = """
from .tokens import TokenService
"""

    parser = PythonASTParser()

    result = parser.parse_source(
        source=source,
        file_metadata=make_file_metadata(
            "app/auth/service.py"
        ),
    )

    imports = [
        relationship
        for relationship in result.relationships
        if relationship.relationship_type
        == RelationshipType.IMPORTS
    ]

    assert (
        imports[0].target_qualified_name
        == "app.auth.tokens.TokenService"
    )