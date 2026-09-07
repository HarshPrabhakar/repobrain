from __future__ import annotations

from pathlib import Path


LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "Python",
    ".pyi": "Python",

    ".js": "JavaScript",
    ".jsx": "JavaScript",

    ".ts": "TypeScript",
    ".tsx": "TypeScript",

    ".java": "Java",

    ".go": "Go",

    ".rs": "Rust",

    ".c": "C",
    ".h": "C",

    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".hpp": "C++",

    ".cs": "C#",

    ".rb": "Ruby",

    ".php": "PHP",

    ".swift": "Swift",

    ".kt": "Kotlin",
    ".kts": "Kotlin",

    ".scala": "Scala",

    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",

    ".ps1": "PowerShell",

    ".sql": "SQL",

    ".html": "HTML",
    ".htm": "HTML",

    ".css": "CSS",
    ".scss": "SCSS",
    ".sass": "Sass",

    ".xml": "XML",

    ".json": "JSON",

    ".yaml": "YAML",
    ".yml": "YAML",

    ".toml": "TOML",

    ".ini": "INI",

    ".cfg": "Config",
    ".conf": "Config",

    ".md": "Markdown",
    ".markdown": "Markdown",

    ".txt": "Text",

    ".rst": "reStructuredText",

    ".proto": "Protocol Buffers",

    ".graphql": "GraphQL",
    ".gql": "GraphQL",
}


SPECIAL_FILENAMES: dict[str, str] = {
    "Dockerfile": "Dockerfile",

    "Makefile": "Makefile",

    "Procfile": "Procfile",

    ".gitignore": "Git",

    ".dockerignore": "Docker",

    ".env": "Environment",

    ".env.example": "Environment",
}


def detect_language(path: Path) -> str:
    """
    Detect a file's language/type using its filename and extension.

    Phase 1 intentionally uses deterministic detection.

    More advanced language detection can be introduced later if necessary.
    """

    filename = path.name

    if filename in SPECIAL_FILENAMES:
        return SPECIAL_FILENAMES[filename]

    extension = path.suffix.lower()

    if extension in LANGUAGE_BY_EXTENSION:
        return LANGUAGE_BY_EXTENSION[extension]

    return "Unknown"