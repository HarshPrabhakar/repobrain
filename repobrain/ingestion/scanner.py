from __future__ import annotations

import hashlib
import os
import uuid

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from repobrain.config import ScannerConfig
from repobrain.ingestion.filters import (
    is_config_file,
    is_generated_file,
    is_test_file,
    should_ignore_directory,
    should_ignore_file,
)
from repobrain.ingestion.language import detect_language
from repobrain.models.repository import (
    FileMetadata,
    RepositoryMetadata,
    RepositoryScanResult,
    ScanError,
)


class RepositoryScanner:
    """
    Phase 1 repository scanner.

    Responsibilities:
    - validate repository root
    - recursively discover files
    - apply ignore rules
    - classify files
    - detect basic file language/type
    - collect stable metadata
    - produce a RepositoryScanResult

    The scanner does NOT:
    - parse ASTs
    - extract symbols
    - generate embeddings
    - build graphs
    - invoke an LLM
    """

    def __init__(
        self,
        config: ScannerConfig | None = None,
    ) -> None:
        self.config = config or ScannerConfig()

    def scan(
        self,
        repository_path: str | Path,
    ) -> RepositoryScanResult:
        """
        Scan a local repository and return structured metadata.
        """

        root = self._validate_repository_root(repository_path)

        scan_started_at = datetime.now(timezone.utc)

        repository_id = self._create_repository_id(root)

        files: list[FileMetadata] = []

        errors: list[ScanError] = []

        total_files_discovered = 0

        ignored_files = 0

        for path in self._walk_repository(
            root=root,
            errors=errors,
        ):
            total_files_discovered += 1

            if should_ignore_file(path, self.config):
                ignored_files += 1
                continue

            try:
                metadata = self._build_file_metadata(
                    root=root,
                    path=path,
                    repository_id=repository_id,
                )

                files.append(metadata)

            except Exception as exc:
                errors.append(
                    ScanError(
                        path=str(path),
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )

        language_counter = Counter(
            file.language
            for file in files
        )

        primary_language = self._determine_primary_language(
            language_counter
        )

        indexed_at = datetime.now(timezone.utc)

        metadata = RepositoryMetadata(
            repository_id=repository_id,
            name=root.name,
            root_path=str(root),

            primary_language=primary_language,

            languages=dict(
                sorted(
                    language_counter.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),

            total_files_discovered=total_files_discovered,

            indexed_files=len(files),

            ignored_files=ignored_files,

            test_files=sum(
                1
                for file in files
                if file.is_test
            ),

            config_files=sum(
                1
                for file in files
                if file.is_config
            ),

            generated_files=sum(
                1
                for file in files
                if file.is_generated
            ),

            created_at=scan_started_at,

            indexed_at=indexed_at,

            schema_version="1.0",
        )

        return RepositoryScanResult(
            metadata=metadata,
            files=files,
            errors=errors,
        )

    def _validate_repository_root(
        self,
        repository_path: str | Path,
    ) -> Path:
        """
        Validate and normalize the supplied repository path.
        """

        raw_path = Path(repository_path).expanduser()

        try:
            root = raw_path.resolve(strict=True)

        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Repository path does not exist: {raw_path}"
            ) from exc

        if not root.is_dir():
            raise NotADirectoryError(
                f"Repository path is not a directory: {root}"
            )

        return root

    def _walk_repository(
        self,
        root: Path,
        errors: list[ScanError],
    ) -> Iterator[Path]:
        """
        Walk repository files while pruning ignored directories.

        os.walk is used intentionally because modifying dirnames in-place
        prevents traversal into ignored directory trees.
        """

        def onerror(error: OSError) -> None:
            errors.append(
                ScanError(
                    path=str(
                        getattr(
                            error,
                            "filename",
                            root,
                        )
                    ),
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )

        for current_root, dirnames, filenames in os.walk(
            root,
            topdown=True,
            onerror=onerror,
            followlinks=self.config.follow_symlinks,
        ):
            current_path = Path(current_root)

            filtered_directories: list[str] = []

            for dirname in dirnames:
                directory_path = current_path / dirname

                if should_ignore_directory(
                    directory_path,
                    self.config,
                ):
                    continue

                if (
                    directory_path.is_symlink()
                    and not self.config.follow_symlinks
                ):
                    continue

                filtered_directories.append(dirname)

            dirnames[:] = filtered_directories

            for filename in filenames:
                path = current_path / filename

                if (
                    path.is_symlink()
                    and not self.config.follow_symlinks
                ):
                    continue

                if not path.is_file():
                    continue

                yield path

    def _build_file_metadata(
        self,
        root: Path,
        path: Path,
        repository_id: str,
    ) -> FileMetadata:
        """
        Build metadata for one repository file.
        """

        resolved_path = path.resolve()

        self._ensure_inside_repository(
            root=root,
            path=resolved_path,
        )

        stat = resolved_path.stat()

        relative_path = resolved_path.relative_to(root).as_posix()

        content_hash: str | None = None

        if self.config.calculate_hashes:
            content_hash = self._calculate_sha256(
                resolved_path
            )

        line_count: int | None = None

        if self.config.count_lines:
            line_count = self._count_lines(
                resolved_path
            )

        file_id = self._create_file_id(
            repository_id=repository_id,
            relative_path=relative_path,
        )

        modified_at = datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        )

        return FileMetadata(
            file_id=file_id,

            repository_id=repository_id,

            relative_path=relative_path,

            absolute_path=str(resolved_path),

            file_name=resolved_path.name,

            extension=resolved_path.suffix.lower(),

            language=detect_language(
                resolved_path
            ),

            size_bytes=stat.st_size,

            line_count=line_count,

            content_hash=content_hash,

            is_test=is_test_file(
                Path(relative_path)
            ),

            is_config=is_config_file(
                Path(relative_path)
            ),

            is_generated=is_generated_file(
                Path(relative_path)
            ),

            is_symlink=path.is_symlink(),

            modified_at=modified_at,
        )

    @staticmethod
    def _create_repository_id(
        root: Path,
    ) -> str:
        """
        Create a deterministic repository ID based on canonical root path.

        UUID5 provides repeatable IDs for the same repository location.
        """

        canonical_path = str(root).casefold()

        repository_uuid = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"repobrain://repository/{canonical_path}",
        )

        return f"repo_{repository_uuid.hex[:16]}"

    @staticmethod
    def _create_file_id(
        repository_id: str,
        relative_path: str,
    ) -> str:
        """
        Create a deterministic file ID.
        """

        file_uuid = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{repository_id}:{relative_path}",
        )

        return f"file_{file_uuid.hex[:16]}"

    @staticmethod
    def _calculate_sha256(
        path: Path,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        """
        Calculate SHA-256 without loading the entire file into memory.
        """

        digest = hashlib.sha256()

        with path.open("rb") as file_handle:
            while True:
                chunk = file_handle.read(chunk_size)

                if not chunk:
                    break

                digest.update(chunk)

        return digest.hexdigest()

    @staticmethod
    def _count_lines(
        path: Path,
    ) -> int:
        """
        Count logical file lines from binary data.

        This avoids relying on text encoding during Phase 1.
        """

        count = 0

        last_byte: bytes | None = None

        with path.open("rb") as file_handle:
            while True:
                block = file_handle.read(
                    1024 * 1024
                )

                if not block:
                    break

                count += block.count(b"\n")

                last_byte = block[-1:]

        if last_byte is not None and last_byte != b"\n":
            count += 1

        return count

    @staticmethod
    def _ensure_inside_repository(
        root: Path,
        path: Path,
    ) -> None:
        """
        Prevent a resolved path from escaping the repository boundary.
        """

        try:
            path.relative_to(root)

        except ValueError as exc:
            raise PermissionError(
                f"Path escapes repository root: {path}"
            ) from exc

    @staticmethod
    def _determine_primary_language(
        languages: Counter[str],
    ) -> str | None:
        """
        Pick the most common known language.
        """

        filtered = {
            language: count
            for language, count in languages.items()
            if language != "Unknown"
        }

        if not filtered:
            return None

        return max(
            filtered,
            key=filtered.get,
        )