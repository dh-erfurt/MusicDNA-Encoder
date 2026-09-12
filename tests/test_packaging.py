from __future__ import annotations

import json
from importlib.metadata import metadata
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_public_readme_is_the_packaged_long_description() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    readme = (ROOT / "README.rst").read_text(encoding="utf-8")
    package_metadata = metadata("musicdna-encoder")

    assert 'readme = "README.rst"' in pyproject
    assert package_metadata["Description-Content-Type"] == "text/x-rst"
    assert package_metadata.get_payload().strip() == readme.strip()


def test_public_readme_states_the_preliminary_contract_and_limits() -> None:
    readme = (ROOT / "README.rst").read_text(encoding="utf-8")

    for required in (
        "This is release ``0.9.0``",
        "pre-1.0",
        "python -m pip install .",
        "musicdna-encoder @ git+https://github.com/dh-erfurt/MusicDNA-Encoder.git",
        "musicdna-encoder path/to/events.json --format pae",
        '"schema_version": "musicdna-events-v0"',
        "musicdna-analysis-v0",
        "musicdna-melody-encoding-v0",
        "external or neural detector",
        "does not read audio",
        "to_plaine_easie",
        "lossy projection",
        "MIT License",
    ):
        assert required in readme

    assert not (ROOT / "README.md").exists()


def test_public_packaging_uses_the_encoder_identity_consistently() -> None:
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package_text = (ROOT / "package.json").read_text(encoding="utf-8")
    package_lock_text = (ROOT / "package-lock.json").read_text(encoding="utf-8")
    package = json.loads(package_text)
    package_lock = json.loads(package_lock_text)

    assert 'name = "musicdna-encoder"' in pyproject_text
    assert "authors = [" in pyproject_text
    assert 'name = "Anna Neovesky"' in pyproject_text
    assert 'name = "Finn Johann Romeis"' in pyproject_text
    assert 'musicdna-encoder = "musicdna_encoder.encoder:main"' in pyproject_text
    assert 'packages = ["src/musicdna_encoder"]' in pyproject_text
    assert 'Homepage = "https://github.com/dh-erfurt/MusicDNA-Encoder"' in pyproject_text
    assert "dependencies = []" in pyproject_text
    assert "[tool.hatch.build.targets.wheel.force-include]" not in pyproject_text
    assert package["name"] == "musicdna-encoder-test-tools"
    assert package_lock["name"] == package["name"]

    public_metadata = (
        pyproject_text,
        package_text,
        package_lock_text,
        (ROOT / "README.rst").read_text(encoding="utf-8"),
        (ROOT / "CITATION.cff").read_text(encoding="utf-8"),
        (ROOT / "THIRD_PARTY_NOTICES.txt").read_text(encoding="utf-8"),
    )
    assert all("musicdna-transformer" not in text.lower() for text in public_metadata)
    assert all("musicdna_transformer" not in text.lower() for text in public_metadata)


def test_citation_contains_planned_release_authors_and_date() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")

    assert "title: MusicDNA-Encoder" in citation
    assert 'repository-code: "https://github.com/dh-erfurt/MusicDNA-Encoder"' in citation
    assert "date-released: 2026-09-15" in citation
    assert "family-names: Neovesky" in citation
    assert "family-names: Romeis" in citation
    assert "doi:" not in citation.lower()
