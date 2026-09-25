"""Test suite per il laboratorio scientifico makima_lab."""

import sys
import unittest
from pathlib import Path

# Permette l'import del modulo makima_lab durante l'esecuzione dei test
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from makima_lab import __version__, lab_status


class TestMakimaLab(unittest.TestCase):
    """Verifiche di base sui metadati e lo stato del package makima_lab."""

    def test_makima_lab_metadata(self):
        self.assertEqual(__version__, "0.1.0")

    def test_makima_lab_status(self):
        status = lab_status()
        self.assertEqual(status["lab"], "makima_lab")
        self.assertEqual(status["version"], "0.1.0")
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["role"], "Scientific Research & NLP Prototyping")


if __name__ == "__main__":
    unittest.main()
