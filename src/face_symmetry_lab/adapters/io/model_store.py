"""Pinned model manifest and verified local install (§10.6, FR-002).

Rules enforced here:

* HTTPS only, and only to hosts on the allow list. No free-form URL input.
* Download to a temporary file, verify size and sha256, then atomic rename.
* A byte-different asset is never installed, and a failed install leaves the
  previously verified file untouched.
* Nothing in this module runs implicitly: ``install`` is only reached from an
  explicit user action or from ``scripts/bootstrap_models.py``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

APP_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST_PATH = APP_ROOT / "configs" / "model-manifest.json"
DEFAULT_MODELS_DIR = APP_ROOT / "models"

#: §10.6 requires a fixed allow list rather than arbitrary download targets.
ALLOWED_MODEL_HOSTS: Final[frozenset[str]] = frozenset(
    {"storage.googleapis.com", "storage.openvinotoolkit.org"}
)

#: Hard ceiling so a misbehaving host cannot fill the disk.
MAX_MODEL_BYTES: Final[int] = 512 * 1024 * 1024

_CHUNK_BYTES: Final[int] = 1024 * 1024
_TEMP_SUFFIX: Final[str] = ".part"


class ModelStoreError(RuntimeError):
    """Manifest, download, or verification failure."""


class InstallStatus(StrEnum):
    MISSING = "MISSING"
    MISMATCH = "MISMATCH"
    READY = "READY"


class ModelAsset(BaseModel):
    """One pinned model file. Immutable and fully specified."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    filename: str
    url: str = Field(min_length=1)
    size_bytes: int = Field(gt=0, le=MAX_MODEL_BYTES)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str = Field(min_length=1)
    license_url: str = Field(min_length=1)
    input_contract: str = Field(min_length=1)

    @field_validator("url")
    @classmethod
    def validate_url(cls, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise ValueError("モデル取得は HTTPS のみを許可します。")
        if parsed.hostname not in ALLOWED_MODEL_HOSTS:
            raise ValueError("許可されていないホストです。")
        return url

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, filename: str) -> str:
        if not filename or filename != Path(filename).name or filename in {".", ".."}:
            raise ValueError("ファイル名が不正です。パス区切りや空文字は使えません。")
        return filename


#: A downloader yields the response body in chunks. Injected for tests.
Downloader = Callable[[str], Iterator[bytes]]


def _default_downloader(url: str) -> Iterator[bytes]:
    from urllib.request import urlopen

    if urlparse(url).scheme != "https":
        raise ModelStoreError("モデル取得は HTTPS のみを許可します。")
    with urlopen(url, timeout=60) as response:
        while chunk := response.read(_CHUNK_BYTES):
            yield chunk


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> tuple[ModelAsset, ...]:
    """Read and validate the pinned manifest."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ModelStoreError(f"model manifest が見つかりません: {path}") from error
    except json.JSONDecodeError as error:
        raise ModelStoreError(f"model manifest を解析できません: {path}") from error

    entries = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ModelStoreError("model manifest に models 配列がありません。")

    try:
        return tuple(ModelAsset.model_validate(entry) for entry in entries)
    except ValidationError as error:
        raise ModelStoreError(f"model manifest の内容が不正です: {error}") from error


def find_asset(assets: tuple[ModelAsset, ...], name: str) -> ModelAsset:
    for asset in assets:
        if asset.name == name:
            return asset
    raise ModelStoreError(f"model manifest に {name} がありません。")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_status(path: Path, asset: ModelAsset) -> InstallStatus:
    """Verify one local path against a pinned manifest entry without network I/O."""
    if not path.is_file():
        return InstallStatus.MISSING
    if path.stat().st_size != asset.size_bytes:
        return InstallStatus.MISMATCH
    if file_sha256(path) != asset.sha256:
        return InstallStatus.MISMATCH
    return InstallStatus.READY


class ModelStore:
    """Local model directory with manifest-verified installs."""

    def __init__(
        self,
        *,
        models_dir: Path = DEFAULT_MODELS_DIR,
        downloader: Downloader = _default_downloader,
    ) -> None:
        self._models_dir = models_dir
        self._downloader = downloader

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def path_for(self, asset: ModelAsset) -> Path:
        return self._models_dir / asset.filename

    def status(self, asset: ModelAsset) -> InstallStatus:
        """Check the local file only. Never touches the network."""
        return asset_status(self.path_for(asset), asset)

    def install(self, asset: ModelAsset) -> Path:
        """Download, verify, and atomically install. Explicit action only."""
        destination = self.path_for(asset)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + _TEMP_SUFFIX)

        digest = sha256()
        received = 0
        try:
            with temporary.open("wb") as handle:
                for chunk in self._downloader(asset.url):
                    received += len(chunk)
                    if received > asset.size_bytes:
                        raise ModelStoreError("取得サイズが manifest と一致しません。")
                    digest.update(chunk)
                    handle.write(chunk)
        except ModelStoreError:
            temporary.unlink(missing_ok=True)
            raise
        except Exception as error:
            temporary.unlink(missing_ok=True)
            raise ModelStoreError(f"モデルを取得できませんでした: {asset.name}") from error

        if received != asset.size_bytes:
            temporary.unlink(missing_ok=True)
            raise ModelStoreError("取得サイズが manifest と一致しません。")
        if digest.hexdigest() != asset.sha256:
            temporary.unlink(missing_ok=True)
            raise ModelStoreError("取得したモデルの sha256 が manifest と一致しません。")

        os.replace(temporary, destination)
        return destination
