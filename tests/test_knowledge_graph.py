import unittest
import tempfile
import sys
from pathlib import Path

# Assicura importazione di makima_lab
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from makima_lab.mind.knowledge_graph import SecondBrainBuilder, KnowledgeNode, KnowledgeEdge


class TestKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.journal_file = Path(self.tmp_dir.name) / "mind_journal.jsonl"

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_second_brain_builder_initialization(self):
        builder = SecondBrainBuilder(journal_path=self.journal_file)
        self.assertGreaterEqual(builder.memory_store.total_count, 3)

    def test_second_brain_graph_construction(self):
        builder = SecondBrainBuilder(journal_path=self.journal_file)
        
        mock_targets = [
            {
                "target": "deploy_prod",
                "probability": 0.85,
                "observations_count": 20,
                "success_count": 17,
                "failure_count": 3,
                "uncertainty_variance": 0.0055,
            },
            {
                "target": "git:commits",
                "probability": 0.92,
                "observations_count": 45,
                "success_count": 41,
                "failure_count": 4,
                "uncertainty_variance": 0.0016,
            },
        ]

        graph = builder.build_graph(target_summaries=mock_targets)

        self.assertGreaterEqual(graph.stats["total_nodes"], 9)
        self.assertGreaterEqual(graph.stats["total_edges"], 10)
        self.assertEqual(graph.stats["targets_count"], 2)
        self.assertGreater(graph.stats["resonance_score"], 0)

        # Verifica esistenza nodi chiave
        node_ids = {n.id for n in graph.nodes}
        self.assertIn("concept_bayesian_core", node_ids)
        self.assertIn("concept_epistemic_calibration", node_ids)
        self.assertIn("concept_git_telemetry", node_ids)
        self.assertIn("target_deploy_prod", node_ids)
        self.assertIn("target_git:commits", node_ids)

        # Verifica serializzazione
        data = graph.to_dict()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertIn("stats", data)


if __name__ == "__main__":
    unittest.main()

