"""Test suite per il modulo NLP e Semantic Query Parsing di Makima Lab."""

import sys
import unittest
from pathlib import Path

# Inclusione path del laboratorio
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from makima_lab.nlp import (
    ForecastQuery,
    Intent,
    SemanticForecastPipeline,
    SemanticQueryParser,
    TemporalRelation,
    TemporalWindow,
)


class TestNLPParser(unittest.TestCase):
    """Test suite per la verifica del parsing semantico e del temporal reasoning."""

    def setUp(self):
        self.parser = SemanticQueryParser()
        self.pipeline = SemanticForecastPipeline(self.parser)

    def test_when_next_release(self):
        query = "Quando rilascerò il prossimo framework?"
        parsed = self.parser.parse(query)

        self.assertEqual(parsed.intent, Intent.RELEASE_PREDICTION)
        self.assertEqual(parsed.target, "framework_release")
        self.assertEqual(parsed.temporal_window.relation, TemporalRelation.NEXT)
        self.assertTrue(parsed.is_valid_forecast)

    def test_probability_before_month(self):
        query = "Qual è la probabilità che rilasci framework entro dicembre?"
        parsed = self.parser.parse(query)

        self.assertEqual(parsed.intent, Intent.RELEASE_PREDICTION)
        self.assertEqual(parsed.target, "framework_release")
        self.assertEqual(parsed.temporal_window.relation, TemporalRelation.BEFORE)
        self.assertEqual(parsed.temporal_window.boundary, "dicembre")
        self.assertTrue(parsed.is_valid_forecast)

    def test_within_days_forecast(self):
        query = "Riuscirò a rilasciare framework entro 30 giorni?"
        parsed = self.parser.parse(query)

        self.assertEqual(parsed.intent, Intent.RELEASE_PREDICTION)
        self.assertEqual(parsed.target, "framework_release")
        self.assertEqual(parsed.temporal_window.relation, TemporalRelation.WITHIN_DAYS)
        self.assertEqual(parsed.temporal_window.days, 30)
        self.assertTrue(parsed.is_valid_forecast)

    def test_this_week_likelihood(self):
        query = "Quanto è probabile che framework venga rilasciato questa settimana?"
        parsed = self.parser.parse(query)

        self.assertEqual(parsed.intent, Intent.RELEASE_PREDICTION)
        self.assertEqual(parsed.target, "framework_release")
        self.assertEqual(parsed.temporal_window.relation, TemporalRelation.THIS_WEEK)
        self.assertTrue(parsed.is_valid_forecast)

    def test_predict_prefix_with_custom_target(self):
        query = "Prevedi se api_gateway verrà rilasciato entro 14 giorni."
        parsed = self.parser.parse(query)

        self.assertEqual(parsed.intent, Intent.RELEASE_PREDICTION)
        self.assertEqual(parsed.temporal_window.relation, TemporalRelation.WITHIN_DAYS)
        self.assertEqual(parsed.temporal_window.days, 14)
        self.assertTrue(parsed.is_valid_forecast)

    def test_rejection_of_chitchat_and_compliments(self):
        # Domande estetiche / chitchat non devono MAI diventare FORECAST
        unsupported_queries = [
            "Quanto è bello il mio framework?",
            "Come stai?",
            "Chi sei?",
            "Raccontami una barzelletta",
            "Ciao Makima",
        ]
        for q in unsupported_queries:
            with self.subTest(query=q):
                parsed = self.parser.parse(q)
                self.assertEqual(parsed.intent, Intent.UNSUPPORTED)
                self.assertFalse(parsed.is_valid_forecast)

    def test_forecast_query_summary(self):
        query = "Quando rilascerò il prossimo framework?"
        parsed = self.parser.parse(query)
        summary = parsed.summary()

        self.assertIn("Query Originale:", summary)
        self.assertIn("RELEASE_PREDICTION", summary)
        self.assertIn("framework_release", summary)
        self.assertIn("Valida per Core:  Si", summary)

    def test_rilasceremo_nuova_feature_questa_settimana(self):
        query = "rilasceremo la nuova feature questa settimana?"
        parsed = self.parser.parse(query)

        self.assertEqual(parsed.intent, Intent.RELEASE_PREDICTION)
        self.assertEqual(parsed.target, "git:feature_ratio")
        self.assertEqual(parsed.temporal_window.relation, TemporalRelation.THIS_WEEK)
        self.assertTrue(parsed.is_valid_forecast)

    def test_unit_tests_discipline_query(self):
        query = "riusciremo a completare i test unitari entro 3 giorni?"
        parsed = self.parser.parse(query)

        self.assertIn(parsed.intent, (Intent.FORECAST, Intent.RELEASE_PREDICTION))
        self.assertEqual(parsed.target, "git:test_discipline")
        self.assertEqual(parsed.temporal_window.relation, TemporalRelation.WITHIN_DAYS)
        self.assertEqual(parsed.temporal_window.days, 3)
        self.assertTrue(parsed.is_valid_forecast)

    def test_pipeline_execution_valid_query(self):
        res = self.pipeline.execute("Qual è la probabilità che rilasci framework entro dicembre?")
        self.assertTrue(res.query.is_valid_forecast)
        self.assertIsNotNone(res.posterior)
        self.assertIsNotNone(res.temporal_probability)
        self.assertGreater(res.posterior.mean, 0.5)
        self.assertIn("MAKIMA SEMANTIC FORECAST PIPELINE", res.format_report())

    def test_pipeline_execution_real_feature_query(self):
        res = self.pipeline.execute("rilasceremo la nuova feature questa settimana?")
        self.assertTrue(res.query.is_valid_forecast)
        self.assertEqual(res.query.target, "git:feature_ratio")
        self.assertIsNotNone(res.posterior)
        self.assertIsNotNone(res.temporal_probability)
        self.assertIn("git:feature_ratio", res.format_report())

    def test_pipeline_execution_rejected_query(self):
        res = self.pipeline.execute("Quanto è bello il mio framework?")
        self.assertFalse(res.query.is_valid_forecast)
        self.assertIsNone(res.posterior)
        self.assertIn("RIFIUTATA", res.format_report())


if __name__ == "__main__":
    unittest.main()
