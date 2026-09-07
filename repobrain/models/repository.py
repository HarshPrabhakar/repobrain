from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    """
    Metadata describing one indexed repository file.
    """

    file_id: str

    repository_id: str

    relative_path: str

    absolute_path: str

    file_name: str

    extension: str

    language: str

    size_bytes: int = Field(ge=0)

    line_count: Optional[int] = Field(
        default=None,
        ge=0,
    )

    content_hash: Optional[str] = None

    is_test: bool = False

    is_config: bool = False

    is_generated: bool = False

    is_symlink: bool = False

    modified_at: Optional[datetime] = None


class RepositoryMetadata(BaseModel):
    """
    Repository-level metadata produced by Phase 1.
    """

    repository_id: str

    name: str

    root_path: str

    primary_language: Optional[str] = None

    languages: Dict[str, int] = Field(
        default_factory=dict
    )

    total_files_discovered: int = Field(
        default=0,
        ge=0,
    )

    indexed_files: int = Field(
        default=0,
        ge=0,
    )

    ignored_files: int = Field(
        default=0,
        ge=0,
    )

    test_files: int = Field(
        default=0,
        ge=0,
    )

    config_files: int = Field(
        default=0,
        ge=0,
    )

    generated_files: int = Field(
        default=0,
        ge=0,
    )

    created_at: datetime

    indexed_at: datetime

    schema_version: str = "1.0"


class ScanError(BaseModel):
    """
    Non-fatal error encountered while scanning.
    """

    path: str

    error_type: str

    message: str


class RepositoryScanResult(BaseModel):
    """
    Complete output of a repository scan.
    """

    metadata: RepositoryMetadata

    files: List[FileMetadata] = Field(
        default_factory=list
    )

    errors: List[ScanError] = Field(
        default_factory=list
    )