from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from types import MappingProxyType

import pytest

from dendra_models.models.networks.zhang_2014 import (
    WINDUP_DATA_REVISION,
    WindUpDataIntegrityError,
    WindUpDataUnavailableError,
    default_windup_data_dir,
    download_windup_data,
    load_windup_data,
    verify_windup_data,
)
from dendra_models.models.networks.zhang_2014 import windup_data as data_module


def _use_synthetic_manifest(monkeypatch, payloads: dict[str, bytes]) -> None:
    digests = MappingProxyType(
        {name: sha256(payload).hexdigest() for name, payload in payloads.items()}
    )
    monkeypatch.setattr(data_module, "WINDUP_DATA_SHA256", digests)
    monkeypatch.setattr(data_module, "WINDUP_DATA_FILENAMES", tuple(digests))
    monkeypatch.setattr(data_module, "WINDUP_DATA_BASE_URL", "https://data.invalid")


def test_default_data_dir_is_revisioned_and_configurable(monkeypatch, tmp_path):
    monkeypatch.setenv("DENDRA_MODELS_DATA_HOME", str(tmp_path))
    assert default_windup_data_dir() == (
        tmp_path / "zhang-2014" / WINDUP_DATA_REVISION / "WindUp"
    )


def test_missing_default_data_never_triggers_a_download(monkeypatch, tmp_path):
    monkeypatch.setenv("DENDRA_MODELS_DATA_HOME", str(tmp_path))

    def unexpected_download(*args, **kwargs):
        raise AssertionError("ordinary data loading must not perform network access")

    monkeypatch.setattr(data_module, "_download_vector", unexpected_download)
    with pytest.raises(WindUpDataUnavailableError, match="download_windup_data"):
        load_windup_data()


def test_downloader_stages_verifies_and_reuses_cached_files(monkeypatch, tmp_path):
    payloads = {"First.txt": b"first\n", "Second.txt": b"second\n"}
    _use_synthetic_manifest(monkeypatch, payloads)
    requested: list[tuple[str, float]] = []

    def fake_urlopen(request, *, timeout):
        name = request.full_url.rsplit("/", 1)[-1]
        requested.append((name, timeout))
        return BytesIO(payloads[name])

    monkeypatch.setattr(data_module, "urlopen", fake_urlopen)
    destination = tmp_path / "vectors"
    assert download_windup_data(destination, timeout=7.5) == destination
    assert requested == [("First.txt", 7.5), ("Second.txt", 7.5)]
    assert verify_windup_data(destination) == destination

    def unexpected_urlopen(*args, **kwargs):
        raise AssertionError("a valid cache should not be downloaded again")

    monkeypatch.setattr(data_module, "urlopen", unexpected_urlopen)
    assert download_windup_data(destination) == destination


def test_bad_download_does_not_replace_existing_files(monkeypatch, tmp_path):
    payloads = {"First.txt": b"first\n", "Second.txt": b"second\n"}
    _use_synthetic_manifest(monkeypatch, payloads)
    destination = tmp_path / "vectors"
    destination.mkdir()
    (destination / "First.txt").write_bytes(b"old contents\n")

    def fake_urlopen(request, *, timeout):
        name = request.full_url.rsplit("/", 1)[-1]
        payload = payloads[name] if name == "First.txt" else b"corrupt\n"
        return BytesIO(payload)

    monkeypatch.setattr(data_module, "urlopen", fake_urlopen)
    with pytest.raises(WindUpDataIntegrityError, match="No cached files were changed"):
        download_windup_data(destination)
    assert (destination / "First.txt").read_bytes() == b"old contents\n"
    assert not (destination / "Second.txt").exists()


def test_custom_vector_directory_can_hold_a_modified_realization(tmp_path):
    vectors = {
        "FromVector.txt": "0 75\n",
        "ToVector.txt": "75 76\n",
        "SynapseVector.txt": "0 0\n",
        "WeightVector.txt": "1.25 2.5\n",
        "DelayVector.txt": "1 2\n",
        "ThresholdVector.txt": "-30 -31\n",
        "SpikeStatsVector.txt": "1 1 10\n",
        "SpikeTimesVector.txt": "5 -1e15\n",
    }
    for name, contents in vectors.items():
        (tmp_path / name).write_text(contents, encoding="utf-8")

    data = load_windup_data(tmp_path)
    assert data.n_connections == 2
    assert data.n_scheduled_connections == 1
    assert data.connection_spike_times[0].tolist() == [5.0]
    assert data.weight.tolist() == [1.25, 2.5]

