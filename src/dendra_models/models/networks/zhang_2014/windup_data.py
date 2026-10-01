"""Acquire the published Zhang et al. Wind-Up vector realization.

The ModelDB vectors are intentionally not distributed with ``dendra-models``.
Downloading is always an explicit user action.  Files are fetched from one
immutable upstream revision and checked against their expected SHA-256 digests
before they enter the local data cache.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Mapping
from urllib.request import Request, urlopen


WINDUP_DATA_REVISION = "f0ac25565d31b60fe676738c559d6ab5086c7308"
"""Immutable ModelDBRepository/168414 revision used by this translation."""

WINDUP_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/ModelDBRepository/168414/"
    f"{WINDUP_DATA_REVISION}/WindUp"
)
"""Pinned upstream directory from which the vectors are downloaded."""

WINDUP_DATA_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "DelayVector.txt": (
            "8b920c4a04cef63743d03797dc62371f07b93990e35ec649a5459693f30e3f13"
        ),
        "FromVector.txt": (
            "3004e62bce84ed9509462c219a24aa53fc252e134654556225303e28fd764f28"
        ),
        "SpikeStatsVector.txt": (
            "95edb3dff24abf1c9b54e2aae25bd95b0b11ac8a03753e710407fdffa3ea1467"
        ),
        "SpikeTimesVector.txt": (
            "4750899c8fdb914e032509475e190c5a1cdc03d556efacc5d51c6827b45c0c90"
        ),
        "SynapseVector.txt": (
            "3a05511a44d77cb592ed99e31b7d29596327aefe99246f3ce86538395e62ea4b"
        ),
        "ThresholdVector.txt": (
            "0250d4a897814835e2fb5f18c28e779ed4d01b109ed4ee383b1841f0b1aaefb0"
        ),
        "ToVector.txt": (
            "3af10451d3a6d249077ea785eeb910870a0b32ad3900d303ebff57b614771cf5"
        ),
        "WeightVector.txt": (
            "307f30dbd25a5b363af94e14bed6f6cd78a0ff153d724f46f4c672e1a7770f9d"
        ),
    }
)
"""Expected SHA-256 digest for every required vector."""

WINDUP_DATA_FILENAMES = tuple(WINDUP_DATA_SHA256)


class WindUpDataUnavailableError(FileNotFoundError):
    """Raised when the optional Wind-Up vectors have not been installed."""


class WindUpDataIntegrityError(RuntimeError):
    """Raised when installed or downloaded vectors fail hash verification."""


class WindUpDataDownloadError(RuntimeError):
    """Raised when a Wind-Up vector cannot be downloaded."""


def dendra_models_data_home() -> Path:
    """Return the base directory for downloaded Dendra Models data.

    ``DENDRA_MODELS_DATA_HOME`` takes precedence.  Otherwise this uses the
    native per-user cache location on macOS and Windows, or
    ``$XDG_CACHE_HOME`` (falling back to ``~/.cache``) on other platforms.
    """

    configured = os.environ.get("DENDRA_MODELS_DATA_HOME")
    if configured:
        return Path(configured).expanduser()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "dendra-models"
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = (
            Path(local_app_data).expanduser()
            if local_app_data
            else Path.home() / "AppData" / "Local"
        )
        return base / "dendra-models" / "Cache"

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    base = (
        Path(xdg_cache_home).expanduser()
        if xdg_cache_home
        else Path.home() / ".cache"
    )
    return base / "dendra-models"


def default_windup_data_dir() -> Path:
    """Return the revision-specific cache directory for the Wind-Up vectors."""

    return (
        dendra_models_data_home()
        / "zhang-2014"
        / WINDUP_DATA_REVISION
        / "WindUp"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_dir(path: str | Path | None) -> Path:
    return default_windup_data_dir() if path is None else Path(path).expanduser()


def verify_windup_data(data_dir: str | Path | None = None) -> Path:
    """Verify the pinned vector realization and return its directory.

    Parameters
    ----------
    data_dir:
        Directory containing the eight upstream vectors.  When omitted, the
        revision-specific user cache is checked.

    Raises
    ------
    WindUpDataUnavailableError
        If one or more vectors are absent.
    WindUpDataIntegrityError
        If a vector does not match the pinned upstream revision.
    """

    root = _data_dir(data_dir)
    missing = [name for name in WINDUP_DATA_FILENAMES if not (root / name).is_file()]
    if missing:
        names = ", ".join(missing)
        raise WindUpDataUnavailableError(
            f"Zhang2014 Wind-Up data are not installed at {root}. Missing: {names}. "
            "Install the pinned ModelDB vectors explicitly with "
            "`from dendra_models.models.networks.zhang_2014 import "
            "download_windup_data; download_windup_data()` or pass a directory "
            "containing your own realization as `data_dir`."
        )

    mismatched = [
        name
        for name, expected in WINDUP_DATA_SHA256.items()
        if _sha256_file(root / name) != expected
    ]
    if mismatched:
        names = ", ".join(mismatched)
        raise WindUpDataIntegrityError(
            f"Zhang2014 Wind-Up data at {root} do not match pinned ModelDB "
            f"revision {WINDUP_DATA_REVISION}. Invalid: {names}. Run "
            "`download_windup_data(force=True)` to replace the cached copy, or "
            "pass a deliberately modified realization as `data_dir`."
        )
    return root


def _download_vector(name: str, *, timeout: float) -> bytes:
    url = f"{WINDUP_DATA_BASE_URL}/{name}"
    request = Request(url, headers={"User-Agent": "dendra-models-data-downloader"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except OSError as exc:
        raise WindUpDataDownloadError(
            f"Could not download Zhang2014 vector {name} from {url}: {exc}"
        ) from exc


def download_windup_data(
    data_dir: str | Path | None = None,
    *,
    force: bool = False,
    timeout: float = 30.0,
) -> Path:
    """Download and verify the pinned ModelDB Wind-Up vector realization.

    The function performs network access only when called.  Valid cached files
    are reused unless ``force=True``.  All required downloads are staged and
    hash checked before any cached file is replaced.

    Parameters
    ----------
    data_dir:
        Destination directory.  By default, use the revision-specific user
        cache returned by :func:`default_windup_data_dir`.
    force:
        Download all vectors again even when the installed copy is valid.
    timeout:
        Per-file network timeout in seconds.
    """

    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError(f"timeout must be finite and positive; got {timeout!r}.")

    target = _data_dir(data_dir)
    if not force:
        try:
            return verify_windup_data(target)
        except (WindUpDataUnavailableError, WindUpDataIntegrityError):
            pass

    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".windup-download-", dir=target.parent) as tmp:
        staging = Path(tmp)
        replacements: list[str] = []

        for name, expected in WINDUP_DATA_SHA256.items():
            destination = target / name
            if not force and destination.is_file():
                if _sha256_file(destination) == expected:
                    continue

            payload = _download_vector(name, timeout=timeout)
            actual = _sha256_bytes(payload)
            if actual != expected:
                raise WindUpDataIntegrityError(
                    f"Downloaded Zhang2014 vector {name} has SHA-256 {actual}; "
                    f"expected {expected} for ModelDB revision "
                    f"{WINDUP_DATA_REVISION}. No cached files were changed."
                )
            (staging / name).write_bytes(payload)
            replacements.append(name)

        target.mkdir(parents=True, exist_ok=True)
        for name in replacements:
            os.replace(staging / name, target / name)

    return verify_windup_data(target)


__all__ = [
    "WINDUP_DATA_REVISION",
    "WINDUP_DATA_BASE_URL",
    "WINDUP_DATA_FILENAMES",
    "WINDUP_DATA_SHA256",
    "WindUpDataUnavailableError",
    "WindUpDataIntegrityError",
    "WindUpDataDownloadError",
    "dendra_models_data_home",
    "default_windup_data_dir",
    "verify_windup_data",
    "download_windup_data",
]
