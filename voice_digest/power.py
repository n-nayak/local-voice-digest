"""macOS sleep prevention (caffeinate) and stage timing helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def prevent_system_sleep() -> Iterator[None]:
    """Prevent macOS idle sleep while a long transcription job is running."""
    proc: subprocess.Popen[bytes] | None = None
    if sys.platform == "darwin":
        caffeinate_bin = shutil.which("caffeinate")
        if caffeinate_bin:
            try:
                proc = subprocess.Popen(
                    [caffeinate_bin, "-i", "-w", str(os.getpid())],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                proc = None
    try:
        yield
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds cleanly (e.g. '42.3s' or '2m 14s')."""
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(round(seconds)), 60)
    return f"{mins}m {secs:02d}s"


@contextmanager
def timed_stage(label: str) -> Iterator[None]:
    """Print start/completion banners with wall-clock timing for a pipeline stage."""
    print(f"\n==> {label}...", flush=True)
    t0 = time.monotonic()
    yield
    dt = time.monotonic() - t0
    print(f"    Done in {format_elapsed(dt)}", flush=True)
