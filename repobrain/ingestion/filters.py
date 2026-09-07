from __future__ import annotations

from pathlib import Path

from repobrain.config import ScannerConfig


GENERATED_FILE_PATTERNS: tuple[str, ...] = (
    ".min.js",
    ".min.css",
    "_pb2.py",
    "_pb2_grpc.py",
)


CONFIG_FILENAMES: frozenset[str] = frozenset(
    {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "tox.ini",
        "pytest.ini",

        "requirements.txt",

        "package.json",
        "package-lock.json",

        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",

        ".env",
        ".env.example",

        ".gitignore",

        "makefile",
    }
)


def should_ignore_directory(
    path: Path,
    config: ScannerConfig,
) -> bool:
    """
    Return True when a directory should be excluded from indexing.

    In addition to explicitly ignored directories, Python packaging metadata
    directories such as *.egg-info are excluded because they contain generated
    installation metadata rather than repository source code.
    """

    directory_name = path.name

    if directory_name in config.ignored_directories:
        return True

    lower_name = directory_name.lower()

    if lower_name.endswith(".egg-info"):
        return True

    if lower_name == ".eggs":
        return True

    return False


def should_ignore_file(
    path: Path,
    config: ScannerConfig,
) -> bool:
    """
    Return True when a file should not be indexed.
    """

    if path.name in config.ignored_files:
        return True

    if path.suffix.lower() in config.ignored_extensions:
        return True

    try:
        if path.stat().st_size > config.max_file_size_bytes:
            return True
    except OSError:
        return True

    return False


def is_test_file(path: Path) -> bool:
    """
    Identify common test-file conventions.

    This is classification only. More accurate test-to-symbol relationships
    belong to later phases.
    """

    lower_name = path.name.lower()

    parts = {part.lower() for part in path.parts}

    if "tests" in parts or "test" in parts:
        return True

    if lower_name.startswith("test_"):
        return True

    if lower_name.endswith("_test.py"):
        return True

    if lower_name.endswith(".test.js"):
        return True

    if lower_name.endswith(".test.ts"):
        return True

    if lower_name.endswith(".spec.js"):
        return True

    if lower_name.endswith(".spec.ts"):
        return True

    return False


def is_config_file(path: Path) -> bool:
    """
    Identify common repository configuration files.
    """

    lower_name = path.name.lower()

    if lower_name in CONFIG_FILENAMES:
        return True

    if path.suffix.lower() in {
        ".ini",
        ".cfg",
        ".conf",
        ".toml",
        ".yaml",
        ".yml",
    }:
        return True

    return False


def is_generated_file(path: Path) -> bool:
    """
    Detect common generated-source patterns.

    We still index these unless configured otherwise because a generated file
    may participate in real repository relationships.
    """

    lower_name = path.name.lower()

    return any(
        lower_name.endswith(pattern)
        for pattern in GENERATED_FILE_PATTERNS
    )