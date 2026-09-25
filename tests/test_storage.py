"""Test suite per la persistenza dello store e aggregazione delle evidenze."""

import sys
import tempfile
import unittest
from pathlib import Path

# Inclusione path del laboratorio
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from makima_lab.storage import (
    compute_knowledge_base_from_store,
    load_store,
)


class TestStorage(unittest.TestCase):
    """Verifica della serializzazione e aggregazione delle evidenze storiche."""

    def test_compute_knowledge_base_from_store(self):
        sample_store = {
            "version": "0.1.0",
            "observations": [
                {"id": 1, "target": "framework_release", "value": 1.0, "timestamp_sec": 1700000000},
                {"id": 2, "target": "framework_release", "value": 1.0, "timestamp_sec": 1700086400},
                {"id": 3, "target": "framework_release", "value": 0.0, "timestamp_sec": 1700172800},
                {"id": 4, "target": "custom_service", "value": 1.0, "timestamp_sec": 1700000000},
            ],
            "outcomes": [],
        }

        kb = compute_knowledge_base_from_store(sample_store)

        self.assertIn("framework_release", kb)
        self.assertEqual(kb["framework_release"]["successes"], 2)
        self.assertEqual(kb["framework_release"]["failures"], 1)

        self.assertIn("custom_service", kb)
        self.assertEqual(kb["custom_service"]["successes"], 1)
        self.assertEqual(kb["custom_service"]["failures"], 0)

    def test_load_non_existent_file_returns_empty_schema(self):
        non_existent = Path("non_existent_path_9999.json")
        data = load_store(non_existent)
        self.assertEqual(data["version"], "0.1.0")
        self.assertEqual(data["observations"], [])
        self.assertEqual(data["outcomes"], [])


if __name__ == "__main__":
    unittest.main()
