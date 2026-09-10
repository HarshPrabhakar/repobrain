from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from repobrain.ingestion import (
    RepositoryScanner,
)

from repobrain.models.application import (
    RepositoryFingerprint,
)


class RepositoryScanLike(
    Protocol
):
    files: object


class RepositoryFingerprinter:
    """
    Calculate deterministic repository fingerprints.

    The default calculate() path uses RepoBrain's own RepositoryScanner
    so cache invalidation observes the same repository file set as the
    indexing pipeline.

    The builder can call from_scan_result() directly after Phase-1 scan
    to avoid scanning the repository twice during an initial build.
    """

    ALGORITHM = "sha256-v1"

    def __init__(
        self,
        *,
        scanner: RepositoryScanner | None = None,
    ) -> None:

        self._scanner = (
            scanner
            if scanner is not None
            else RepositoryScanner()
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def calculate(
        self,
        repository_root: Path,
    ) -> RepositoryFingerprint:
        """
        Scan a repository and calculate its deterministic fingerprint.
        """

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

        scan_result = (
            self._scanner.scan(
                normalized_root
            )
        )

        return self.from_scan_result(
            scan_result
        )

    @classmethod
    def from_scan_result(
        cls,
        scan_result: RepositoryScanLike,
    ) -> RepositoryFingerprint:
        """
        Calculate a fingerprint from an existing RepoBrain scan result.

        File ordering is normalized before hashing.

        File identity contributes:

            relative path
            content SHA-256
            file size

        If the scanner did not provide content_hash for a file, the
        fingerprinter calculates SHA-256 directly from absolute_path.
        """

        files = list(
            scan_result.files
        )

        files.sort(
            key=lambda item: (
                str(
                    item.relative_path
                ).casefold(),
                str(
                    item.relative_path
                ),
            )
        )

        digest = (
            hashlib.sha256()
        )

        total_size_bytes = 0

        for file_metadata in files:

            relative_path = str(
                file_metadata.relative_path
            )

            size_bytes = int(
                file_metadata.size_bytes
            )

            total_size_bytes += (
                size_bytes
            )

            content_hash = getattr(
                file_metadata,
                "content_hash",
                None,
            )

            if not content_hash:

                absolute_path = Path(
                    file_metadata.absolute_path
                )

                content_hash = (
                    cls._calculate_file_sha256(
                        absolute_path
                    )
                )

            cls._update_digest_field(
                digest,
                relative_path,
            )

            cls._update_digest_field(
                digest,
                str(
                    content_hash
                ).lower(),
            )

            cls._update_digest_field(
                digest,
                str(
                    size_bytes
                ),
            )

        return RepositoryFingerprint(
            algorithm=(
                cls.ALGORITHM
            ),
            value=(
                digest.hexdigest()
            ),
            file_count=(
                len(files)
            ),
            total_size_bytes=(
                total_size_bytes
            ),
        )

    # ==================================================================
    # Hash helpers
    # ==================================================================

    @staticmethod
    def _update_digest_field(
        digest: object,
        value: str,
    ) -> None:
        """
        Add one unambiguous UTF-8 field to the repository digest.
        """

        encoded = (
            value.encode(
                "utf-8"
            )
        )

        digest.update(
            len(encoded)
            .to_bytes(
                8,
                byteorder="big",
                signed=False,
            )
        )

        digest.update(
            encoded
        )

    @staticmethod
    def _calculate_file_sha256(
        path: Path,
        *,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        """
        Hash one file without loading it fully into memory.
        """

        digest = (
            hashlib.sha256()
        )

        with path.open(
            "rb"
        ) as file_handle:

            while True:

                chunk = (
                    file_handle.read(
                        chunk_size
                    )
                )

                if not chunk:
                    break

                digest.update(
                    chunk
                )

        return digest.hexdigest()
