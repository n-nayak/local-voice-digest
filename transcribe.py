#!/usr/bin/env python3
"""Entry point for Local Audio Transcription, Diarization & Acoustic Analysis.

Enforces strict offline environment variables before any ML libraries can be imported,
unless '--download-models' is explicitly requested.
"""

from __future__ import annotations

import os
import sys

# Enforce offline environment variables at earliest startup for inference runs
if "--download-models" not in sys.argv[1:]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"

from voice_digest.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
