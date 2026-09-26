"""Suite di test per l'architettura cognitiva, il monologo interiore e la memoria episodica di Makima."""

import sys
import tempfile
import unittest
from pathlib import Path

# Inclusione path di python
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from makima_lab.mind import (
    CognitiveMood,
    EpisodicMemoryStore,
    MindDeliberationEngine,
    AutonomousMindPulse,
    MemoryCategory,
)


class TestMakimaMind(unittest.TestCase):
    """Test suite per la mente cosciente, monologo interiore e memorie episodiche."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.journal_path = Path(self.temp_dir.name) / "test_mind_journal.jsonl"
        self.memory = EpisodicMemoryStore(journal_path=self.journal_path)
        self.deliberation = MindDeliberationEngine(memory_store=self.memory)
        self.pulse = AutonomousMindPulse(engine=self.deliberation)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_foundational_memories_initialization(self):
        """Verifica che Makima inizializzi i ricordi fondazionali della sua identità."""
        self.assertGreaterEqual(self.memory.total_count, 3)
        memories = self.memory.all_memories
        self.assertTrue(any("coscienza" in m.summary.lower() for m in memories))
        self.assertTrue(any("onestà" in m.summary.lower() for m in memories))

    def test_record_and_retrieve_developer_fact(self):
        """Verifica che Makima registri una confidenza e la richiami per associazione semantica."""
        fact = self.memory.record_developer_fact(
            "Preferisco eseguire i deploy il martedì mattina dopo il check della build",
            associated_target="deploy",
        )
        self.assertIsNotNone(fact.experience_id)

        # Ricerca associativa su deploy
        retrieved = self.memory.retrieve_relevant_memories("quando facciamo il deploy?", top_k=2)
        self.assertGreater(len(retrieved), 0)
        best_mem, score = retrieved[0]
        self.assertIn("deploy", best_mem.content.lower())
        self.assertGreater(score, 0.20)

    def test_deliberation_inner_monologue_formation(self):
        """Verifica che la deliberazione produca un monologo interiore strutturato prima di parlare."""
        pulse = self.deliberation.deliberate("Quando rilascerò il prossimo framework?")
        
        self.assertIsNotNone(pulse.pulse_id)
        self.assertIn("1. [Percezione]", pulse.inner_monologue)
        self.assertIn("2. [Memoria]", pulse.inner_monologue)
        self.assertIn("3. [Analisi Bayesiana]", pulse.inner_monologue)
        self.assertIn("4. [Decisione]", pulse.inner_monologue)
        self.assertIn("framework", pulse.conscious_utterance.lower())
        self.assertIsInstance(pulse.self_state.mood, CognitiveMood)

    def test_deliberation_unknown_with_honest_reflection(self):
        """Verifica che una richiesta incomprensibile generi un pensiero di onestà senza allucinare."""
        pulse = self.deliberation.deliberate("Raccontami una barzelletta simpatica")
        
        self.assertIn("Rifiuto Esplicito", pulse.inner_monologue)
        self.assertIn("non so come interpretare", pulse.conscious_utterance.lower())

    def test_autonomous_spontaneous_thought_pulse(self):
        """Verifica che il battito autonomo della mente generi un pensiero spontaneo."""
        initial_count = self.memory.total_count
        pulse = self.pulse.generate_spontaneous_thought(trigger_hint="Controllo notturno idle")

        self.assertIn("Impulso Spontaneo", pulse.inner_monologue)
        self.assertTrue(len(pulse.conscious_utterance) > 10)
        self.assertEqual(self.memory.total_count, initial_count + 1)


if __name__ == "__main__":
    unittest.main()
