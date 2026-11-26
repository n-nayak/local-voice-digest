"""Strict offline enforcement and outbound network socket guard."""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", ""})

_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_create_connection = socket.create_connection
_orig_getaddrinfo = socket.getaddrinfo
_blocker_active = False


class OfflineNetworkError(RuntimeError):
    """Raised immediately if any code attempts an external network operation offline."""


def configure_model_cache_env(model_dir: Path | str) -> Path:
    """Point Hugging Face and PyTorch caches into the local model directory."""
    resolved = Path(model_dir).expanduser().resolve()
    hf_dir = resolved / "huggingface"
    torch_dir = resolved / "torch"

    os.environ["HF_HOME"] = str(hf_dir)
    os.environ["HF_HUB_CACHE"] = str(hf_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_dir / "hub")
    os.environ["TORCH_HOME"] = str(torch_dir)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    return resolved


def enforce_offline_environment(model_dir: Path | str | None = None) -> None:
    """Set strict offline environment variables prior to importing ML libraries."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    if model_dir is not None:
        configure_model_cache_env(model_dir)


def _is_loopback_address(family: int, address: Any) -> bool:
    if hasattr(socket, "AF_UNIX") and family == socket.AF_UNIX:
        return True
    if isinstance(address, tuple) and address:
        host = str(address[0]).strip().lower()
        return host in LOOPBACK_HOSTS
    return False


def _guarded_connect(self: socket.socket, address: Any) -> Any:
    if not _is_loopback_address(self.family, address):
        raise OfflineNetworkError(
            f"Blocked outbound socket connect to {address!r} in offline mode."
        )
    return _orig_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
    if not _is_loopback_address(self.family, address):
        raise OfflineNetworkError(
            f"Blocked outbound socket connect_ex to {address!r} in offline mode."
        )
    return _orig_connect_ex(self, address)


def _guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
    host = str(address[0]).strip().lower() if isinstance(address, tuple) and address else ""
    if host not in LOOPBACK_HOSTS:
        raise OfflineNetworkError(
            f"Blocked outbound create_connection to {address!r} in offline mode."
        )
    return _orig_create_connection(address, *args, **kwargs)


def _guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
    host_str = "" if host is None else str(host).strip().lower()
    if host_str not in LOOPBACK_HOSTS:
        raise OfflineNetworkError(
            f"Blocked external DNS resolution for {host!r} in offline mode."
        )
    return _orig_getaddrinfo(host, port, *args, **kwargs)


def enable_network_blocker() -> None:
    """Install a socket-level guard that immediately rejects non-local network I/O."""
    global _blocker_active
    if _blocker_active:
        return
    socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]
    socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]
    _blocker_active = True


def disable_network_blocker() -> None:
    """Restore original socket functions."""
    global _blocker_active
    if not _blocker_active:
        return
    socket.socket.connect = _orig_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _orig_connect_ex  # type: ignore[method-assign]
    socket.create_connection = _orig_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = _orig_getaddrinfo  # type: ignore[assignment]
    _blocker_active = False


def is_network_blocker_active() -> bool:
    return _blocker_active


@contextmanager
def offline_socket_guard() -> Iterator[None]:
    """Context manager enabling the outbound network blocker for a block."""
    was_active = _blocker_active
    if not was_active:
        enable_network_blocker()
    try:
        yield
    finally:
        if not was_active:
            disable_network_blocker()
