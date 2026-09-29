"""Golden: Python StructuredIntent JSON deve deserializzare nel contratto Rust."""

from __future__ import annotations
import json
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from makima_lab.nlp.pipeline import SemanticForecastPipeline
from makima_lab.nlp.schemas.intent import Intent
from makima_lab.nlp.schemas.temporal import TemporalRelation


class TestPythonRustIntentContract(unittest.TestCase):
    """Verifica campi obbligatori e valori ammessi dal mirror Rust."""

    REQUIRED_KEYS = {
        "raw_query",
        "intent",
        "target",
        "temporal_window",
        "entities",
        "confidence",
        "confidence_breakdown",
        "is_valid_for_core",
        "validation_notes",
    }

    VALID_INTENTS = {i.value for i in Intent}
    VALID_TEMPORAL = {t.value for t in TemporalRelation}

    def setUp(self):
        self.pipeline = SemanticForecastPipeline(knowledge_base={})

    def _assert_contract(self, payload: dict) -> None:
        self.assertTrue(self.REQUIRED_KEYS.issubset(payload.keys()))
        self.assertIn(payload["intent"], self.VALID_INTENTS)
        tw = payload["temporal_window"]
        self.assertIn(tw["relation"], self.VALID_TEMPORAL)
        self.assertIsInstance(payload["confidence"], float)
        self.assertIsInstance(payload["is_valid_for_core"], bool)
        self.assertIsInstance(payload["entities"], list)
        self.assertIsInstance(payload["validation_notes"], list)
        # Deve essere JSON deserializzabile (compatibile serde)
        roundtrip = json.loads(json.dumps(payload, ensure_ascii=False))
        self.assertEqual(payload["intent"], roundtrip["intent"])

    def test_release_query_contract(self):
        structured = self.pipeline.process_intent("Quando rilascerò il prossimo framework?")
        payload = structured.to_dict()
        self._assert_contract(payload)
        self.assertEqual(payload["intent"], Intent.TEMPORAL_QUERY.value)
        self.assertEqual(payload["target"], "framework_release")
        self.assertEqual(payload["temporal_window"]["relation"], TemporalRelation.FUTURE.value)
        self.assertTrue(payload["is_valid_for_core"])

    def test_deploy_query_contract(self):
        structured = self.pipeline.process_intent("Qual è la probabilità del deploy?")
        payload = structured.to_dict()
        self._assert_contract(payload)
        self.assertEqual(payload["intent"], Intent.QUERY.value)
        self.assertEqual(payload["target"], "deploy")
        self.assertTrue(payload["is_valid_for_core"])

    def test_unknown_rejected(self):
        structured = self.pipeline.process_intent("ciao come stai")
        payload = structured.to_dict()
        self._assert_contract(payload)
        self.assertEqual(payload["intent"], Intent.UNKNOWN.value)
        self.assertFalse(payload["is_valid_for_core"])

    def test_compact_json_is_single_object(self):
        structured = self.pipeline.process_intent("Prevedi se la daily build passerà questa settimana")
        compact = structured.to_json_compact()
        self.assertTrue(compact.startswith("{"))
        self.assertNotIn("\n", compact)
        payload = json.loads(compact)
        self._assert_contract(payload)


if __name__ == "__main__":
    unittest.main()
