"""Suite di test esaustiva per la pipeline NLP multi-livello di Makima.

Verifica ogni livello della pipeline:
- Preprocessing & Tokenizzazione
- Rappresentazione Semantica Vettoriale (384-dim)
- Intent Classification & Rifiuto UNKNOWN
- Target & Entity Extraction (Canonico + Cosine Ranking)
- Comprensione Temporale (Passato, Presente, Futuro, Intervalli, Date)
- Memoria Conversazionale e Risoluzione Anafora (Pronomi/Riferimenti)
- Stima della Confidenza Composita
- Validazione Strutturata e Guardrails verso Rust
- Integrazione End-to-End e Serializzazione JSON
"""

import sys
import time
import unittest
from pathlib import Path

# Inclusione path di python
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from makima_lab.nlp.preprocessing.cleaner import TextCleaner
from makima_lab.nlp.preprocessing.tokenizer import SimpleTokenizer
from makima_lab.nlp.embeddings.representation import SemanticRepresentation
from makima_lab.nlp.intent.classifier import IntentClassifier
from makima_lab.nlp.entities.target_extractor import TargetExtractor
from makima_lab.nlp.temporal.analyzer import TemporalAnalyzer
from makima_lab.nlp.context.memory import ConversationContext, ConversationTurn
from makima_lab.nlp.confidence.estimator import ConfidenceEstimator
from makima_lab.nlp.validation.validator import IntentValidator
from makima_lab.nlp.schemas.intent import Intent
from makima_lab.nlp.schemas.temporal import TemporalRelation, TemporalWindow
from makima_lab.nlp.schemas.structured_intent import StructuredIntent
from makima_lab.nlp.pipeline import MakimaNLPPipeline, SemanticForecastPipeline


class TestPreprocessingAndTokenization(unittest.TestCase):
    """Verifica il livello di pulizia testo e tokenizzazione."""

    def test_text_cleaner_normalization(self):
        raw = "   QUANDO   rilascerò   l’app  ???   "
        cleaned = TextCleaner.clean(raw)
        self.assertEqual(cleaned, "quando rilascerò l app")

    def test_tokenizer_tokens_and_stopwords(self):
        text = "qual è la probabilità per il deploy in produzione"
        tokens = SimpleTokenizer.tokenize(text)
        self.assertIn("deploy", tokens)
        self.assertIn("produzione", tokens)

        filtered = SimpleTokenizer.filter_stopwords(tokens)
        self.assertNotIn("la", filtered)
        self.assertNotIn("il", filtered)
        self.assertIn("deploy", filtered)
        self.assertIn("produzione", filtered)

    def test_tokenizer_ngrams(self):
        tokens = ["daily", "build", "passera"]
        bigrams = SimpleTokenizer.extract_ngrams(tokens, n=2)
        self.assertEqual(bigrams, ["daily build", "build passera"])


class TestSemanticRepresentation(unittest.TestCase):
    """Verifica la dimensionalità, normalizzazione e similarità semantica."""

    def setUp(self):
        self.repr = SemanticRepresentation()

    def test_vector_dimension_and_norm(self):
        vec = self.repr.encode("rilascio nuova versione")
        self.assertEqual(vec.shape[0], self.repr.dimension)
        self.assertAlmostEqual(float(self.repr.dimension), 384.0, delta=1.0)
        norm = float((vec ** 2).sum() ** 0.5)
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_cosine_similarity(self):
        sim_close = self.repr.similarity("rilascio software", "distribuzione release")
        sim_far = self.repr.similarity("rilascio software", "partita di calcio stadio")
        self.assertIsInstance(sim_close, float)
        self.assertIsInstance(sim_far, float)

    def test_candidate_ranking(self):
        candidates = ["deploy", "git:commit_frequency", "daily_build"]
        ranked = self.repr.rank_candidates("quando facciamo il deploy?", candidates)
        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0][0], "deploy")


class TestIntentClassification(unittest.TestCase):
    """Verifica la corretta classificazione dell'intento e il rifiuto UNKNOWN."""

    def setUp(self):
        self.classifier = IntentClassifier()

    def test_query_intent(self):
        intent, conf, _ = self.classifier.classify("Qual è la probabilità di successo del deploy?")
        self.assertEqual(intent, Intent.QUERY)
        self.assertGreater(conf, 0.5)

    def test_temporal_query_intent(self):
        intent, conf, _ = self.classifier.classify("Quando rilascerò il prossimo framework?")
        self.assertEqual(intent, Intent.TEMPORAL_QUERY)
        self.assertGreater(conf, 0.6)

    def test_command_intent(self):
        intent, conf, _ = self.classifier.classify("Sincronizza subito i commit di git")
        self.assertEqual(intent, Intent.COMMAND)

    def test_information_intent(self):
        intent, conf, _ = self.classifier.classify("Spiegami come funziona il Brier Score")
        self.assertEqual(intent, Intent.INFORMATION)

    def test_observation_intent(self):
        intent, conf, _ = self.classifier.classify("Oggi ho completato i test con successo")
        self.assertEqual(intent, Intent.OBSERVATION)

    def test_status_intent(self):
        intent, conf, _ = self.classifier.classify("Qual è lo stato diagnostico del sistema?")
        self.assertEqual(intent, Intent.STATUS)

    def test_unknown_rejection_for_nonsense(self):
        intent, conf, _ = self.classifier.classify("xyz123 nonsensestring blabla")
        self.assertEqual(intent, Intent.UNKNOWN)
        self.assertLessEqual(conf, 0.5)

    def test_unknown_rejection_for_chitchat(self):
        intent, conf, _ = self.classifier.classify("Ciao, raccontami una barzelletta simpatica")
        self.assertEqual(intent, Intent.UNKNOWN)


class TestTargetAndEntityExtraction(unittest.TestCase):
    """Verifica l'estrazione deterministica ed embedding-ranked dei target."""

    def setUp(self):
        self.extractor = TargetExtractor()

    def test_exact_canonical_matching(self):
        t, entities, score, method = self.extractor.extract("qual è la probabilità per il deploy?")
        self.assertEqual(t, "deploy")
        self.assertGreaterEqual(score, 0.9)

    def test_keyword_descriptors(self):
        t1, _, _, _ = self.extractor.extract("quando rilascerò il prossimo framework?")
        self.assertEqual(t1, "framework_release")

        t2, _, _, _ = self.extractor.extract("cosa dice la daily build di oggi?")
        self.assertEqual(t2, "daily_build")

        t3, _, _, _ = self.extractor.extract("verifica i test di regressione")
        self.assertEqual(t3, "git:test_discipline")

    def test_empty_when_no_target_present(self):
        t, entities, score, method = self.extractor.extract("spiegami cosa significa la formula")
        self.assertIsNone(t)
        self.assertEqual(score, 0.0)


class TestTemporalUnderstanding(unittest.TestCase):
    """Verifica l'interpretazione temporale multi-livello."""

    def setUp(self):
        self.analyzer = TemporalAnalyzer()

    def test_future_relation(self):
        tw = self.analyzer.analyze("quando rilascerò la versione futura?")
        self.assertEqual(tw.relation, TemporalRelation.FUTURE)

    def test_past_relation(self):
        tw = self.analyzer.analyze("quando ho completato i test di ieri?")
        self.assertEqual(tw.relation, TemporalRelation.PAST)

    def test_present_relation(self):
        tw = self.analyzer.analyze("cosa sto facendo attualmente nel progetto?")
        self.assertEqual(tw.relation, TemporalRelation.PRESENT)

    def test_relative_interval_days(self):
        tw = self.analyzer.analyze("completeremo il deploy entro 7 giorni?")
        self.assertEqual(tw.relation, TemporalRelation.RELATIVE_INTERVAL)
        self.assertEqual(tw.days, 7)

    def test_relative_interval_week(self):
        tw = self.analyzer.analyze("cosa succederà questa settimana?")
        self.assertEqual(tw.relation, TemporalRelation.RELATIVE_INTERVAL)
        self.assertEqual(tw.days, 7)

    def test_specific_iso_date(self):
        tw = self.analyzer.analyze("prevedi il rilascio entro il 2026-12-31")
        self.assertEqual(tw.relation, TemporalRelation.SPECIFIC_DATE)
        self.assertEqual(tw.raw_expression, "2026-12-31")


class TestConversationContextAndAnaphora(unittest.TestCase):
    """Verifica la memoria conversazionale limitata e la risoluzione dei pronomi."""

    def setUp(self):
        self.context = ConversationContext(max_turns=5)

    def test_bounded_fifo(self):
        for i in range(10):
            intent = StructuredIntent(
                raw_query=f"Turn {i}",
                intent=Intent.QUERY,
                target=f"target_{i}",
                temporal_window=TemporalWindow(relation=TemporalRelation.FUTURE),
                entities=[],
                confidence=0.8,
                is_valid_for_core=True,
            )
            self.context.append_turn(ConversationTurn(raw_query=f"Turn {i}", structured_intent=intent, timestamp_sec=time.time()))

        self.assertEqual(len(self.context._turns), 5)
        self.assertEqual(self.context.last_target, "target_9")

    def test_pronoun_anaphora_resolution(self):
        first_intent = StructuredIntent(
            raw_query="Sto lavorando al framework",
            intent=Intent.OBSERVATION,
            target="framework_release",
            temporal_window=TemporalWindow(relation=TemporalRelation.PRESENT),
            entities=[],
            confidence=0.85,
            is_valid_for_core=True,
        )
        self.context.append_turn(ConversationTurn(raw_query="Sto lavorando al framework", structured_intent=first_intent, timestamp_sec=time.time()))

        resolved = self.context.resolve_anaphora("quando lo rilascio?", current_target=None)
        self.assertEqual(resolved, "framework_release")

    def test_previous_reference_resolution(self):
        intent1 = StructuredIntent(
            raw_query="deploy",
            intent=Intent.QUERY,
            target="deploy",
            temporal_window=TemporalWindow(relation=TemporalRelation.FUTURE),
            entities=[],
            confidence=0.9,
            is_valid_for_core=True,
        )
        intent2 = StructuredIntent(
            raw_query="daily build",
            intent=Intent.QUERY,
            target="daily_build",
            temporal_window=TemporalWindow(relation=TemporalRelation.FUTURE),
            entities=[],
            confidence=0.9,
            is_valid_for_core=True,
        )
        self.context.append_turn(ConversationTurn(raw_query="deploy", structured_intent=intent1, timestamp_sec=time.time()))
        self.context.append_turn(ConversationTurn(raw_query="daily build", structured_intent=intent2, timestamp_sec=time.time()))

        resolved_prev = self.context.resolve_anaphora("e per quello precedente?", current_target=None)
        self.assertEqual(resolved_prev, "deploy")


class TestConfidenceEstimation(unittest.TestCase):
    """Verifica il calcolo trasparente e composito della confidenza."""

    def test_high_confidence_case(self):
        tw = TemporalWindow(relation=TemporalRelation.RELATIVE_INTERVAL, days=7)
        score, diag = ConfidenceEstimator.estimate(
            intent=Intent.QUERY,
            intent_score=0.9,
            target="deploy",
            target_score=0.95,
            temporal_window=tw,
        )
        self.assertGreaterEqual(score, 0.8)
        self.assertLessEqual(score, 1.0)
        self.assertIn("intent_signal", diag)

    def test_penalty_for_unknown_intent(self):
        tw = TemporalWindow(relation=TemporalRelation.UNKNOWN)
        score, _ = ConfidenceEstimator.estimate(
            intent=Intent.UNKNOWN,
            intent_score=0.2,
            target=None,
            target_score=0.0,
            temporal_window=tw,
        )
        self.assertLessEqual(score, 0.25)


class TestIntentValidator(unittest.TestCase):
    """Verifica i guardrails e le regole di ammissione verso il core Rust."""

    def test_valid_query_passes(self):
        tw = TemporalWindow(relation=TemporalRelation.FUTURE)
        valid, notes = IntentValidator.validate(
            intent=Intent.TEMPORAL_QUERY,
            target="framework_release",
            temporal_window=tw,
            confidence=0.82,
        )
        self.assertTrue(valid)
        self.assertEqual(len(notes), 1)

    def test_query_without_target_fails_core_admission(self):
        tw = TemporalWindow(relation=TemporalRelation.UNKNOWN)
        valid, notes = IntentValidator.validate(
            intent=Intent.QUERY,
            target=None,
            temporal_window=tw,
            confidence=0.7,
        )
        self.assertFalse(valid)
        self.assertTrue(any("Nessun target" in n for n in notes))

    def test_unknown_fails_core_admission(self):
        tw = TemporalWindow(relation=TemporalRelation.UNKNOWN)
        valid, notes = IntentValidator.validate(
            intent=Intent.UNKNOWN,
            target=None,
            temporal_window=tw,
            confidence=0.1,
        )
        self.assertFalse(valid)


class TestEndToEndPipeline(unittest.TestCase):
    """Verifica l'integrazione end-to-end e la compatibilità JSON verso Rust Core."""

    def setUp(self):
        self.pipeline = MakimaNLPPipeline()

    def test_full_pipeline_query_release(self):
        structured = self.pipeline.process_intent("Quando rilascerò il prossimo framework?")
        self.assertEqual(structured.intent, Intent.TEMPORAL_QUERY)
        self.assertEqual(structured.target, "framework_release")
        self.assertEqual(structured.temporal_window.relation, TemporalRelation.FUTURE)
        self.assertTrue(structured.is_valid_for_core)
        self.assertGreater(structured.confidence, 0.6)

        # Verifica esportazione serializzabile in JSON per Rust Core
        json_dict = structured.to_dict()
        self.assertEqual(json_dict["intent"], "TEMPORAL_QUERY")
        self.assertEqual(json_dict["target"], "framework_release")
        self.assertEqual(json_dict["temporal_window"]["relation"], "FUTURE")
        self.assertIsInstance(json_dict["confidence"], float)

    def test_full_pipeline_legacy_compatibility(self):
        legacy_pipeline = SemanticForecastPipeline()
        result = legacy_pipeline.run("Qual è la probabilità per il deploy?")
        self.assertIsNotNone(result.query)
        self.assertEqual(result.query.target, "deploy")
        self.assertIn("deploy", result.explanation.lower())


if __name__ == "__main__":
    unittest.main()
