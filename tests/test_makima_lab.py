"""Test iniziali per il laboratorio scientifico Makima."""

import sys
from pathlib import Path

# Permette l'import del modulo makima_lab durante l'esecuzione dei test
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from makima_lab import __version__, lab_status


def test_makima_lab_metadata():
    assert __version__ == "0.1.0"


def test_makima_lab_status():
    status = lab_status()
    assert status["lab"] == "makima_lab"
    assert status["version"] == "0.1.0"
    assert status["status"] == "ready"


if __name__ == "__main__":
    test_makima_lab_metadata()
    test_makima_lab_status()
    print("Tutti i test Python di base sono stati superati con successo.")
