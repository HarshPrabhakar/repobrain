from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from repobrain.application import (
    RepositoryFingerprinter,
)


@dataclass
class FakeFileMetadata:
    relative_path: str
    absolute_path: str
    size_bytes: int
    content_hash: str | None = None


@dataclass
class FakeScanResult:
    files: list[FakeFileMetadata]


def sha256_text(
    value: str,
) -> str:

    return (
        hashlib.sha256(
            value.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


def test_same_files_produce_same_fingerprint(
    tmp_path: Path,
) -> None:

    one = tmp_path / "a.py"
    two = tmp_path / "b.py"

    one.write_text(
        "print('a')",
        encoding="utf-8",
    )

    two.write_text(
        "print('b')",
        encoding="utf-8",
    )

    first_scan = FakeScanResult(
        files=[
            FakeFileMetadata(
                relative_path="b.py",
                absolute_path=str(two),
                size_bytes=two.stat().st_size,
                content_hash=(
                    sha256_text(
                        "print('b')"
                    )
                ),
            ),
            FakeFileMetadata(
                relative_path="a.py",
                absolute_path=str(one),
                size_bytes=one.stat().st_size,
                content_hash=(
                    sha256_text(
                        "print('a')"
                    )
                ),
            ),
        ]
    )

    second_scan = FakeScanResult(
        files=list(
            reversed(
                first_scan.files
            )
        )
    )

    first = (
        RepositoryFingerprinter
        .from_scan_result(
            first_scan
        )
    )

    second = (
        RepositoryFingerprinter
        .from_scan_result(
            second_scan
        )
    )

    assert first == second


def test_content_change_changes_fingerprint(
    tmp_path: Path,
) -> None:

    path = tmp_path / "a.py"

    path.write_text(
        "one",
        encoding="utf-8",
    )

    first = (
        RepositoryFingerprinter
        .from_scan_result(
            FakeScanResult(
                files=[
                    FakeFileMetadata(
                        relative_path="a.py",
                        absolute_path=str(path),
                        size_bytes=3,
                        content_hash=(
                            sha256_text(
                                "one"
                            )
                        ),
                    )
                ]
            )
        )
    )

    second = (
        RepositoryFingerprinter
        .from_scan_result(
            FakeScanResult(
                files=[
                    FakeFileMetadata(
                        relative_path="a.py",
                        absolute_path=str(path),
                        size_bytes=3,
                        content_hash=(
                            sha256_text(
                                "two"
                            )
                        ),
                    )
                ]
            )
        )
    )

    assert (
        first.value
        != second.value
    )


def test_path_change_changes_fingerprint(
    tmp_path: Path,
) -> None:

    path = tmp_path / "a.py"

    path.write_text(
        "same",
        encoding="utf-8",
    )

    content_hash = (
        sha256_text(
            "same"
        )
    )

    first = (
        RepositoryFingerprinter
        .from_scan_result(
            FakeScanResult(
                files=[
                    FakeFileMetadata(
                        relative_path="a.py",
                        absolute_path=str(path),
                        size_bytes=4,
                        content_hash=content_hash,
                    )
                ]
            )
        )
    )

    second = (
        RepositoryFingerprinter
        .from_scan_result(
            FakeScanResult(
                files=[
                    FakeFileMetadata(
                        relative_path="renamed.py",
                        absolute_path=str(path),
                        size_bytes=4,
                        content_hash=content_hash,
                    )
                ]
            )
        )
    )

    assert (
        first.value
        != second.value
    )


def test_missing_content_hash_falls_back_to_file_hash(
    tmp_path: Path,
) -> None:

    path = tmp_path / "a.py"

    path.write_text(
        "hello",
        encoding="utf-8",
    )

    result = (
        RepositoryFingerprinter
        .from_scan_result(
            FakeScanResult(
                files=[
                    FakeFileMetadata(
                        relative_path="a.py",
                        absolute_path=str(path),
                        size_bytes=(
                            path.stat().st_size
                        ),
                        content_hash=None,
                    )
                ]
            )
        )
    )

    assert (
        len(
            result.value
        )
        == 64
    )

    assert (
        result.file_count
        == 1
    )

    assert (
        result.total_size_bytes
        == path.stat().st_size
    )


def test_empty_repository_snapshot_is_valid() -> None:

    result = (
        RepositoryFingerprinter
        .from_scan_result(
            FakeScanResult(
                files=[]
            )
        )
    )

    assert (
        result.file_count
        == 0
    )

    assert (
        result.total_size_bytes
        == 0
    )

    assert (
        len(
            result.value
        )
        == 64
    )
