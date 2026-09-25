import sys
import unittest
from pathlib import Path

# Permette l'import del modulo makima_lab durante l'esecuzione dei test
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from makima_lab.git_observer import GitObserver, categorize_commit, GitCommit


class TestGitObserver(unittest.TestCase):

    def setUp(self):
        self.observer = GitObserver(".")

    def test_categorize_commit(self):
        self.assertEqual(categorize_commit("fix: risolto bug critico nel parser"), "bugfix")
        self.assertEqual(categorize_commit("feat: aggiunta nuova interfaccia utente"), "feature")
        self.assertEqual(categorize_commit("test: aggiunto test per sqlite storage"), "test")
        self.assertEqual(categorize_commit("docs: aggiornata architettura e readme"), "docs")
        self.assertEqual(categorize_commit("refactor: pulizia moduli"), "refactor")
        self.assertEqual(categorize_commit("chore: update deps"), "chore")

    def test_fetch_recent_commits(self):
        commits = self.observer.fetch_recent_commits(max_count=10)
        self.assertIsInstance(commits, list)
        if commits:
            first = commits[0]
            self.assertIsInstance(first, GitCommit)
            self.assertTrue(len(first.commit_hash) >= 7)
            self.assertIn(first.category, ["feature", "bugfix", "refactor", "docs", "test", "chore"])

    def test_sync_history_returns_valid_structure(self):
        res = self.observer.sync_history(max_count=15)
        self.assertIn("status", res)
        self.assertIn("synced_commits", res)
        self.assertIn("categories", res)
        self.assertIn("commit_rate_per_day", res)
        self.assertGreaterEqual(res["synced_commits"], 0)


if __name__ == "__main__":
    unittest.main()
