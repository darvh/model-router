"""Self-checks for the decision engine, classifier, and DeepSWE label derivation. Stdlib unittest."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_router.catalog import Model, tier_of_ref
from model_router.classify import Classifier
from model_router.deepswe import labels
from model_router.engine import Router, RunStore

CONFIG = {
    "tiers": {
        "utility": {"models": ["p/u1", "p/u2"], "directive": "U"},
        "balanced": {"models": ["p/b1"], "directive": "B"},
        "frontier": {"models": ["p/f1", "p/f2"], "directive": "F"},
    },
    "advisor": {"utility": "auto", "balanced": "auto", "frontier": "p/f2"},
    "cost_bands": {"utility": 1.5, "balanced": 15.0},
    "classifier": {"frontier_prob": 0.45, "utility_prob": 0.55},
    "instructions": {"base": "BASE", "escalation": "ESC {frm}->{to}", "advisor": "ADV"},
}


def make_models():
    return {
        r: Model(r, "p", r.split("/")[1], r, ci, co, 128000, True, True, "2026-01-01")
        for r, ci, co in [
            ("p/u1", 0.1, 0.4),
            ("p/u2", 0.1, 1.0),
            ("p/b1", 2.0, 10.0),
            ("p/f1", 5.0, 25.0),
            ("p/f2", 10.0, 50.0),
        ]
    }


class StubClassifier:
    def __init__(self, tier):
        self.tier = tier

    def predict(self, text):
        return self.tier, {"utility": 0.1, "balanced": 0.6, "frontier": 0.3}, "stub"


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.router = Router(CONFIG, make_models(), classifier=StubClassifier("balanced"))

    def test_sticky_same_model(self):
        d1 = self.router.decide("r1", "do a thing")
        d2 = self.router.decide("r1", "another turn")
        self.assertEqual(d1.model, "p/b1")
        self.assertEqual(d2.model, d1.model)
        self.assertTrue(d2.sticky)
        self.assertEqual(d2.turn, 2)

    def test_escalation(self):
        self.router.decide("r2", "x")
        d = self.router.decide("r2", signals={"prior_failure": True})
        self.assertEqual(d.tier, "frontier")
        self.assertEqual(d.model, "p/f1")
        self.assertEqual(d.escalation_count, 1)
        self.assertIn("ESC balanced->frontier", d.instructions)

    def test_floor_signal(self):
        r = Router(CONFIG, make_models(), classifier=StubClassifier("utility"))
        d = r.decide("r3", "x")
        self.assertEqual(d.tier, "utility")
        d = r.decide("r3", signals={"not_understood": True})
        self.assertEqual(d.tier, "balanced")

    def test_advisor_chain(self):
        r = Router(CONFIG, make_models(), classifier=StubClassifier("utility"))
        d = r.decide("r4", "x")
        self.assertEqual(d.advisor, "p/b1")  # utility auto -> first balanced
        d = r.decide("r4", signals={"prior_failure": True})
        self.assertEqual(d.tier, "balanced")
        self.assertEqual(d.advisor, "p/f1")  # balanced auto -> first frontier
        d = r.decide("r4", signals={"prior_failure": True})
        self.assertEqual(d.tier, "frontier")
        self.assertEqual(d.advisor, "p/f2")  # frontier -> configured advisor

    def test_instruction_stack(self):
        d = self.router.decide("r5", "x", instruction="first extra")
        self.assertEqual(d.instructions[0], "BASE")
        self.assertIn("B", d.instructions)
        self.assertIn("first extra", d.instructions)
        self.router.add_instruction("r5", "second extra")
        d = self.router.decide("r5")
        self.assertIn("second extra", d.instructions)
        self.assertLess(d.instructions.index("first extra"), d.instructions.index("second extra"))

    def test_complete_and_restart(self):
        self.router.decide("r6", "x")
        d = self.router.decide("r6", signals={"complete": True})
        self.assertTrue(d.closed)
        self.assertFalse(d.sticky)
        d = self.router.decide("r6", "x")
        self.assertFalse(d.closed)
        self.assertEqual(d.turn, 1)

    def test_needs_advisor_flag(self):
        self.router.decide("r7", "x")
        d = self.router.decide("r7", signals={"needs_advisor": True})
        self.assertTrue(d.advisor_required)
        self.assertTrue(d.advisor)

    def test_persistence_across_routers(self):
        with tempfile.TemporaryDirectory() as td:
            r = Router(CONFIG, make_models(), classifier=StubClassifier("utility"), store=RunStore(td))
            r.decide("run-a", "x")
            r2 = Router(CONFIG, make_models(), classifier=StubClassifier("frontier"), store=RunStore(td))
            d = r2.decide("run-a", "totally different text must stay sticky")
            self.assertEqual(d.tier, "utility")

    def test_missing_catalog_ref_fails_fast(self):
        with self.assertRaises(ValueError):
            Router(CONFIG, {"p/u1": make_models()["p/u1"]})


class ClassifierTests(unittest.TestCase):
    SIMPLE = "rename the variable and fix the typo in the small config file"
    COMPLEX = (
        "refactor the distributed transaction protocol and migrate the concurrency model "
        "across multiple services, redesigning the cache invalidation architecture under load"
    )

    def test_train_and_predict(self):
        examples = [("%s %d" % (self.SIMPLE, i), "utility") for i in range(8)]
        examples += [("%s %d" % (self.COMPLEX, i), "frontier") for i in range(8)]
        clf = Classifier().fit(examples)
        tier, _probs, source = clf.predict(self.SIMPLE)
        self.assertEqual(source, "trained")
        self.assertEqual(tier, "utility")
        tier2, _probs2, _ = clf.predict(self.COMPLEX)
        self.assertEqual(tier2, "frontier")

    def test_heuristic_fallback(self):
        tier, _probs, source = Classifier().predict("fix a typo")
        self.assertEqual(source, "heuristic")
        self.assertIn(tier, ("utility", "balanced", "frontier"))

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            examples = [("small quick fix %d" % i, "utility") for i in range(5)]
            examples += [("huge complex redesign %d" % i, "frontier") for i in range(5)]
            clf = Classifier().fit(examples)
            path = os.path.join(td, "classifier.json")
            clf.save(path)
            clf2 = Classifier.load(path)
            self.assertEqual(clf.predict("small quick fix")[0], clf2.predict("small quick fix")[0])
            self.assertEqual(clf.predict("huge complex redesign")[0], clf2.predict("huge complex redesign")[0])


class LabelsTests(unittest.TestCase):
    @staticmethod
    def trial(task, model, passed, errored=False, cost=0.1):
        return {
            "task_name": task,
            "model": model,
            "provider": "p",
            "passed": passed,
            "errored": errored,
            "cost_usd": cost,
            "source": "deep-swe",
        }

    def test_labels_pick_cheapest_passing_tier(self):
        tasks = [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}]
        trials = [
            self.trial("t1", "u1", True),
            self.trial("t2", "u1", False),
            self.trial("t2", "f1", True),
            self.trial("t3", "u1", False),
            self.trial("t3", "b1", False),
        ]
        out = labels(tasks, trials, lambda ref: tier_of_ref(ref, CONFIG, make_models()))
        self.assertEqual(out["t1"], "utility")
        self.assertEqual(out["t2"], "frontier")
        self.assertEqual(out["t3"], "frontier")

    def test_cost_band_fallback(self):
        # p/unknown not in any tier list and not in catalog -> conservative frontier
        out = labels([{"id": "t1"}], [self.trial("t1", "unknown", True)], lambda ref: None)
        self.assertEqual(out["t1"], "frontier")


if __name__ == "__main__":
    unittest.main(verbosity=2)
