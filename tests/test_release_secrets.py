"""Offline checks for credential isolation and release scanning."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config_runtime
import validate_delivery


class ReleaseSecretTests(unittest.TestCase):
    def test_env_load_preserves_exported_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(
                "KAFU_API_TOKEN=local-token\nDEEPSEEK_API_KEY=local-key\n",
                encoding="utf-8",
            )
            with (
                patch.object(config_runtime, "__file__", str(root / "config_runtime.py")),
                patch.dict(os.environ, {"KAFU_API_TOKEN": "exported-token"}, clear=True),
            ):
                config_runtime.apply_default_env()
                self.assertEqual(os.environ["KAFU_API_TOKEN"], "exported-token")
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "local-key")
                self.assertNotIn("EMBEDDING_API_KEY", os.environ)

    def test_scans_config_examples_and_data_without_exposing_values(self):
        secret = "sk-" + "x" * 32
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("config_runtime.py", ".env.example", "data/result.json", ".env", ".env.local", ".run/trace.jsonl"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(secret, encoding="utf-8")
            errors = []
            with patch.object(validate_delivery, "ROOT", root):
                validate_delivery.check_secret_sprawl(errors)
            self.assertEqual(len(errors), 3)
            for name in ("config_runtime.py", ".env.example", "data/result.json"):
                self.assertTrue(any(name in error for error in errors))
            self.assertTrue(all(secret not in error for error in errors))


if __name__ == "__main__":
    unittest.main()
