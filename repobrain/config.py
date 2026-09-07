from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_IGNORED_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",

        ".idea",
        ".vscode",

        ".venv",
        "venv",
        "env",

        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",

        "node_modules",
        "vendor",

        "dist",
        "build",
        "target",
        "out",

        ".next",
        ".nuxt",

        "coverage",
        "htmlcov",

        ".cache",
    }
)


DEFAULT_IGNORED_FILES: frozenset[str] = frozenset(
    {
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
    }
)


DEFAULT_IGNORED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pyc",
        ".pyo",

        ".dll",
        ".so",
        ".dylib",
        ".exe",

        ".class",
        ".jar",

        ".o",
        ".obj",

        ".zip",
        ".tar",
        ".gz",
        ".7z",
        ".rar",

        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",

        ".mp3",
        ".wav",
        ".mp4",
        ".mov",
        ".avi",

        ".woff",
        ".woff2",
        ".ttf",
        ".otf",

        ".pdf",
    }
)


@dataclass(slots=True)
class ScannerConfig:
    """
    Configuration used by the RepoBrain repository scanner.
    """

    ignored_directories: frozenset[str] = field(
        default_factory=lambda: DEFAULT_IGNORED_DIRECTORIES
    )

    ignored_files: frozenset[str] = field(
        default_factory=lambda: DEFAULT_IGNORED_FILES
    )

    ignored_extensions: frozenset[str] = field(
        default_factory=lambda: DEFAULT_IGNORED_EXTENSIONS
    )

    max_file_size_bytes: int = 2 * 1024 * 1024

    calculate_hashes: bool = True

    count_lines: bool = True

    follow_symlinks: bool = False