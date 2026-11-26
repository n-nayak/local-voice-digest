"""Tests for strict offline environment enforcement and socket blocking."""

from __future__ import annotations

import os
import socket
import tempfile
import unittest
from pathlib import Path

from local_transcribe.offline import (
    OfflineNetworkError,
    enforce_offline_environment,
    is_network_blocker_active,
    offline_socket_guard,
)


class TestOfflineEnforcement(unittest.TestCase):
    def test_enforce_offline_env_vars_and_cache_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models"
            enforce_offline_environment(model_dir)
            self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
            self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")
            self.assertEqual(os.environ.get("HF_HUB_DISABLE_TELEMETRY"), "1")
            self.assertIn(str(model_dir.resolve()), os.environ.get("HF_HOME", ""))
            self.assertIn(str(model_dir.resolve()), os.environ.get("TORCH_HOME", ""))

    def test_socket_guard_blocks_external_network_immediately(self) -> None:
        self.assertFalse(is_network_blocker_active())
        with offline_socket_guard():
            self.assertTrue(is_network_blocker_active())
            with self.assertRaises(OfflineNetworkError):
                socket.create_connection(("huggingface.co", 443), timeout=1)
            with self.assertRaises(OfflineNetworkError):
                socket.getaddrinfo("example.com", 80)
            with self.assertRaises(OfflineNetworkError):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    s.connect(("8.8.8.8", 53))
                finally:
                    s.close()
        self.assertFalse(is_network_blocker_active())


if __name__ == "__main__":
    unittest.main()
