from __future__ import annotations

from pathlib import Path

import pytest

from repobrain.config import ScannerConfig
from repobrain.ingestion.scanner import RepositoryScanner


@pytest.fixture
def sample_repository(
    tmp_path: Path,
) -> Path:
    """
    Create a controlled fake repository for scanner tests.
    """

    repo = tmp_path / "sample_repo"

    repo.mkdir()

    # Main source
    app = repo / "app"
    app.mkdir()

    (app / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )

    (app / "main.py").write_text(
        "def main():\n"
        "    return 'hello'\n",
        encoding="utf-8",
    )

    # Authentication module
    auth = app / "auth"
    auth.mkdir()

    (auth / "service.py").write_text(
        "class AuthService:\n"
        "    def login(self):\n"
        "        return True\n",
        encoding="utf-8",
    )

    # Tests
    tests = repo / "tests"
    tests.mkdir()

    (tests / "test_auth.py").write_text(
        "def test_auth():\n"
        "    assert True\n",
        encoding="utf-8",
    )

    # Configuration
    (repo / "pyproject.toml").write_text(
        "[project]\n"
        'name = "sample"\n',
        encoding="utf-8",
    )

    # Documentation
    (repo / "README.md").write_text(
        "# Sample Repository\n",
        encoding="utf-8",
    )

    # Ignored Git content
    git_dir = repo / ".git"
    git_dir.mkdir()

    (git_dir / "config").write_text(
        "should not be indexed",
        encoding="utf-8",
    )

    # Ignored virtual environment
    venv = repo / ".venv"
    venv.mkdir()

    (venv / "fake.py").write_text(
        "print('ignore me')",
        encoding="utf-8",
    )

    # Ignored binary extension
    (repo / "image.png").write_bytes(
        b"fake binary image"
    )

    return repo


def test_scan_returns_repository_metadata(
    sample_repository: Path,
) -> None:
    scanner = RepositoryScanner()

    result = scanner.scan(
        sample_repository
    )

    assert (
        result.metadata.name
        == "sample_repo"
    )

    assert result.metadata.repository_id.startswith(
        "repo_"
    )

    assert (
        Path(result.metadata.root_path)
        == sample_repository.resolve()
    )


def test_scanner_indexes_expected_files(
    sample_repository: Path,
) -> None:
    scanner = RepositoryScanner()

    result = scanner.scan(
        sample_repository
    )

    paths = {
        file.relative_path
        for file in result.files
    }

    assert "app/__init__.py" in paths

    assert "app/main.py" in paths

    assert "app/auth/service.py" in paths

    assert "tests/test_auth.py" in paths

    assert "pyproject.toml" in paths

    assert "README.md" in paths


def test_scanner_ignores_git_directory(
    sample_repository: Path,
) -> None:
    scanner = RepositoryScanner()

    result = scanner.scan(
        sample_repository
    )

    paths = {
        file.relative_path
        for file in result.files
    }

    assert not any(
        path.startswith(".git/")
        for path in paths
    )


def test_scanner_ignores_virtual_environment(
    sample_repository: Path,
) -> None:
    scanner = RepositoryScanner()

    result = scanner.scan(
        sample_repository
    )

    paths = {
        file.relative_path
        for file in result.files
    }

    assert not any(
        path.startswith(".venv/")
        for path in paths
    )


def test_scanner_ignores_binary_extensions(
    sample_repository: Path,
) -> None:
    scanner = RepositoryScanner()

    result = scanner.scan(
        sample_repository
    )

    paths = {
        file.relative_path
        for file in result.files
    }

    assert "image.png" not in paths


def test_language_detection(
    sample_repository: Path,
) -> None:
    scanner = RepositoryScanner()

    result = scanner.scan(
        sample_repository
    )

    python_files = [
        file
        for file in result.files
        if file.language == "Python"
    ]

    assert len(python_files) == 4

    assert result.metadata.primary_language == "Python"


def test_test_file_detection(
    sample_repository: Path,
) -> None:
    scanner = RepositoryScanner()

    result = scanner.scan(
        sample_repository
    )

    test_file = next(
        file
        for file in result.files
        if file.relative_path
        == "tests/test_auth.py"
    )

    assert test_file.is_test is True

    assert result.metadata.test_files == 1


def test_config_file_detection(
    sample_repository: Path,
) -> None:
    scanner = RepositoryScanner()

    result = scanner.scan(
        sample_repository
    )

    config_file = next(
        file
        for file in result.files
        if file.relative_path
        == "pyproject.toml"
    )

    assert config_file.is_config is True


def test_content_hash_is_created(
    sample_repository: Path,
) -> None:
    scanner = RepositoryScanner()

    result = scanner.scan(
        sample_repository
    )

    main_file = next(
        file
        for file in result.files
        if file.relative_path
        == "app/main.py"
    )

    assert main_file.content_hash is not None

    assert len(main_file.content_hash) == 64


def test_file_ids_are_deterministic(
    sample_repository: Path,
) -> None:
    scanner = RepositoryScanner()

    first_scan = scanner.scan(
        sample_repository
    )

    second_scan = scanner.scan(
        sample_repository
    )

    first_ids = {
        file.relative_path: file.file_id
        for file in first_scan.files
    }

    second_ids = {
        file.relative_path: file.file_id
        for file in second_scan.files
    }

    assert first_ids == second_ids


def test_repository_id_is_deterministic(
    sample_repository: Path,
) -> None:
    scanner = RepositoryScanner()

    first_scan = scanner.scan(
        sample_repository
    )

    second_scan = scanner.scan(
        sample_repository
    )

    assert (
        first_scan.metadata.repository_id
        == second_scan.metadata.repository_id
    )


def test_line_count(
    sample_repository: Path,
) -> None:
    scanner = RepositoryScanner()

    result = scanner.scan(
        sample_repository
    )

    main_file = next(
        file
        for file in result.files
        if file.relative_path
        == "app/main.py"
    )

    assert main_file.line_count == 2


def test_missing_repository_raises_error(
    tmp_path: Path,
) -> None:
    scanner = RepositoryScanner()

    missing = tmp_path / "does_not_exist"

    with pytest.raises(
        FileNotFoundError
    ):
        scanner.scan(
            missing
        )


def test_file_path_cannot_be_used_as_repository(
    tmp_path: Path,
) -> None:
    scanner = RepositoryScanner()

    file_path = tmp_path / "file.txt"

    file_path.write_text(
        "hello",
        encoding="utf-8",
    )

    with pytest.raises(
        NotADirectoryError
    ):
        scanner.scan(
            file_path
        )


def test_hashing_can_be_disabled(
    sample_repository: Path,
) -> None:
    config = ScannerConfig(
        calculate_hashes=False
    )

    scanner = RepositoryScanner(
        config=config
    )

    result = scanner.scan(
        sample_repository
    )

    assert all(
        file.content_hash is None
        for file in result.files
    )

def test_scanner_ignores_egg_info_directory(
    sample_repository: Path,
) -> None:
    egg_info = sample_repository / "sample.egg-info"

    egg_info.mkdir()

    (egg_info / "PKG-INFO").write_text(
        "generated package metadata",
        encoding="utf-8",
    )

    (egg_info / "SOURCES.txt").write_text(
        "app/main.py",
        encoding="utf-8",
    )

    scanner = RepositoryScanner()

    result = scanner.scan(
        sample_repository
    )

    paths = {
        file.relative_path
        for file in result.files
    }

    assert not any(
        path.startswith("sample.egg-info/")
        for path in paths
    )
