"""固定した model manifest と、検証付きのローカル install（§10.6, FR-002）。

ここで強制する規則は次のとおり。

* HTTPS のみ、かつ許可リストにあるホストのみ。自由入力の URL は受け付けない。
* 一時ファイルへ取得し、サイズと sha256 を検証してから atomic に rename する。
* バイト列が異なる asset は決して install せず、install に失敗しても以前に検証
  済みのファイルは手つかずのまま残す。
* このモジュールは暗黙には何も実行しない。``install`` へ到達するのは明示的な
  利用者操作か ``scripts/bootstrap_models.py`` からだけである。
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

#: FR-020: すべてのエクスポート記録が帰属する解決済み依存集合。`file_sha256`
#: でハッシュ化する。`APP_ROOT` の隣で定義することで、呼び出し側が自分の
#: ファイル深さから repository root を導き直さずに済む。
LOCK_PATH = APP_ROOT / "uv.lock"

#: §10.6 は任意の取得先ではなく、固定した許可リストを要求する。
ALLOWED_MODEL_HOSTS: Final[frozenset[str]] = frozenset(
    {"storage.googleapis.com", "storage.openvinotoolkit.org"}
)

#: 不正な挙動のホストがディスクを埋め尽くさないための上限。
MAX_MODEL_BYTES: Final[int] = 512 * 1024 * 1024

_CHUNK_BYTES: Final[int] = 1024 * 1024
_TEMP_SUFFIX: Final[str] = ".part"
_DOWNLOAD_ATTEMPTS: Final[int] = 3


class ModelStoreError(RuntimeError):
    """manifest・取得・検証のいずれかが失敗した。"""


class InstallStatus(StrEnum):
    MISSING = "MISSING"
    MISMATCH = "MISMATCH"
    READY = "READY"


class ModelAsset(BaseModel):
    """固定されたモデルファイル 1 件。不変で、完全に指定されている。"""

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


class DerivedModelAsset(BaseModel):
    """ローカル生成のモデルファイル 1 件。元 asset と変換手順に固定される。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    filename: str
    size_bytes: int = Field(gt=0, le=MAX_MODEL_BYTES)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str = Field(min_length=1)
    license_url: str = Field(min_length=1)
    input_contract: str = Field(min_length=1)
    source_assets: tuple[str, ...] = Field(min_length=1)
    converter: str = Field(min_length=1)
    converter_version: str = Field(min_length=1)
    tools: dict[str, str] = Field(min_length=1)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, filename: str) -> str:
        return ModelAsset.validate_filename(filename)

    @field_validator("converter")
    @classmethod
    def validate_converter(cls, converter: str) -> str:
        path = Path(converter)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
            raise ValueError("変換scriptはrepository内のPython相対pathに限定します。")
        return converter


#: downloader はレスポンス本文を chunk 単位で yield する。テストのため注入可能。
Downloader = Callable[[str], Iterator[bytes]]


def _default_downloader(url: str) -> Iterator[bytes]:
    from urllib.request import urlopen

    if urlparse(url).scheme != "https":
        raise ModelStoreError("モデル取得は HTTPS のみを許可します。")
    with urlopen(url, timeout=60) as response:
        while chunk := response.read(_CHUNK_BYTES):
            yield chunk


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> tuple[ModelAsset, ...]:
    """固定された manifest を読み込んで検証する。"""
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


def load_derived_assets(
    path: Path = DEFAULT_MANIFEST_PATH,
) -> tuple[DerivedModelAsset, ...]:
    """manifest からローカル派生モデルの項目を読み込んで検証する。"""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ModelStoreError(f"model manifest が見つかりません: {path}") from error
    except json.JSONDecodeError as error:
        raise ModelStoreError(f"model manifest を解析できません: {path}") from error

    entries = raw.get("derived_models") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ModelStoreError("model manifest に derived_models 配列がありません。")
    try:
        return tuple(DerivedModelAsset.model_validate(entry) for entry in entries)
    except ValidationError as error:
        raise ModelStoreError(f"derived model manifest の内容が不正です: {error}") from error


def find_asset(assets: tuple[ModelAsset, ...], name: str) -> ModelAsset:
    for asset in assets:
        if asset.name == name:
            return asset
    raise ModelStoreError(f"model manifest に {name} がありません。")


def find_derived_asset(
    assets: tuple[DerivedModelAsset, ...],
    name: str,
) -> DerivedModelAsset:
    for asset in assets:
        if asset.name == name:
            return asset
    raise ModelStoreError(f"derived model manifest に {name} がありません。")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_status(path: Path, asset: ModelAsset | DerivedModelAsset) -> InstallStatus:
    """ローカルの path 1 つを、固定 manifest の項目と照合する。通信は行わない。"""
    if not path.is_file():
        return InstallStatus.MISSING
    if path.stat().st_size != asset.size_bytes:
        return InstallStatus.MISMATCH
    if file_sha256(path) != asset.sha256:
        return InstallStatus.MISMATCH
    return InstallStatus.READY


class ModelStore:
    """manifest で検証した install を行う、ローカルのモデルディレクトリ。"""

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
        """ローカルのファイルだけを確認する。通信は一切行わない。"""
        return asset_status(self.path_for(asset), asset)

    def install(self, asset: ModelAsset) -> Path:
        """取得・検証し、atomic に install する。明示的な操作からのみ呼ぶ。

        モデルのホストや中継が、短いエラーページを返したり応答を途中で打ち切ったり
        することがまれにある。そうした未検証の応答は再試行する。ただし、サイズと
        sha256 が厳密に一致したものしか install しないという規則は保つ。
        """
        destination = self.path_for(asset)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + _TEMP_SUFFIX)

        for attempt in range(_DOWNLOAD_ATTEMPTS):
            digest = sha256()
            received = 0
            try:
                with temporary.open("wb") as handle:
                    for chunk in self._downloader(asset.url):
                        received += len(chunk)
                        if received > asset.size_bytes:
                            raise ModelStoreError(
                                "取得サイズが manifest と一致しません"
                                f"（expected={asset.size_bytes}, received={received}）。"
                            )
                        digest.update(chunk)
                        handle.write(chunk)

                if received != asset.size_bytes:
                    raise ModelStoreError(
                        "取得サイズが manifest と一致しません"
                        f"（expected={asset.size_bytes}, received={received}）。"
                    )
                if digest.hexdigest() != asset.sha256:
                    raise ModelStoreError("取得したモデルの sha256 が manifest と一致しません。")
            except ModelStoreError:
                temporary.unlink(missing_ok=True)
                if attempt == _DOWNLOAD_ATTEMPTS - 1:
                    raise
                continue
            except Exception as error:
                temporary.unlink(missing_ok=True)
                if attempt == _DOWNLOAD_ATTEMPTS - 1:
                    raise ModelStoreError(f"モデルを取得できませんでした: {asset.name}") from error
                continue

            os.replace(temporary, destination)
            return destination

        raise AssertionError("download retry loop exited unexpectedly")
