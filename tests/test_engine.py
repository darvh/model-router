"""Self-checks for the decision engine, classifier, and DeepSWE label derivation. Stdlib unittest."""
import json as _json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_router.catalog import AA_INDICES_CACHE, Model, apply_tier_ranking, load_aa_indices, rank_candidates, tier_of_ref
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
    def __init__(self, tier, probs=None):
        self.tier = tier
        self.probs = probs or {"utility": 0.1, "balanced": 0.6, "frontier": 0.3}

    def predict(self, text):
        return self.tier, dict(self.probs), "stub"


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
        self.assertFalse(d2.advisor_required)
        self.assertEqual(d2.advisor_reason, "")

    def test_escalation(self):
        self.router.decide("r2", "x")
        d = self.router.decide("r2", signals={"prior_failure": True})
        self.assertEqual(d.tier, "frontier")
        self.assertEqual(d.model, "p/f1")
        self.assertEqual(d.escalation_count, 1)
        self.assertIn("ESC balanced->frontier", d.instructions)
        self.assertTrue(d.advisor_required)
        self.assertEqual(d.advisor_reason, "escalation")

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
        self.assertEqual(d.advisor_reason, "needs_advisor")
        self.assertTrue(d.advisor)

    def test_persistence_across_routers(self):
        with tempfile.TemporaryDirectory() as td:
            r = Router(CONFIG, make_models(), classifier=StubClassifier("utility"), store=RunStore(td))
            r.decide("run-a", "x")
            r2 = Router(CONFIG, make_models(), classifier=StubClassifier("frontier"), store=RunStore(td))
            d = r2.decide("run-a", "totally different text must stay sticky")
            self.assertEqual(d.tier, "utility")

    def test_expected_cost_rule(self):
        import json as _json

        cfg = _json.loads(_json.dumps(CONFIG))
        cfg["decision"] = {
            "rule": "expected_cost",
            "confidence_cap": 1.0,
            "tier_costs": {"utility": 0.1, "balanced": 1.8, "frontier": 3.84},
        }
        r = Router(cfg, make_models(), classifier=StubClassifier("balanced", {"utility": 0.9, "balanced": 0.1, "frontier": 0.0}))
        self.assertEqual(r.decide("rc1", "x").tier, "utility")
        r2 = Router(cfg, make_models(), classifier=StubClassifier("utility", {"utility": 0.05, "balanced": 0.05, "frontier": 0.9}))
        self.assertEqual(r2.decide("rc2", "x").tier, "frontier")
        r3 = Router(cfg, make_models(), classifier=StubClassifier("frontier", {"utility": 0.0, "balanced": 0.9, "frontier": 0.1}))
        self.assertEqual(r3.decide("rc3", "x").tier, "balanced")

    def test_confidence_cap_guards_overconfident_classifier(self):
        import json as _json

        cfg = _json.loads(_json.dumps(CONFIG))
        cfg["decision"] = {
            "rule": "expected_cost",
            "confidence_cap": 0.9,
            "tier_costs": {"utility": 0.1, "balanced": 1.8, "frontier": 3.84},
        }
        r = Router(cfg, make_models(), classifier=StubClassifier("balanced", {"utility": 0.0, "balanced": 1.0, "frontier": 0.0}))
        self.assertEqual(r.decide("rc4", "x").tier, "utility")

    def test_escalation_caps_at_top(self):
        r = Router(CONFIG, make_models(), classifier=StubClassifier("balanced"))
        r.decide("rt", "x")
        d = r.decide("rt", signals={"prior_failure": True})  # -> frontier p/f1
        self.assertEqual(d.model, "p/f1")
        d = r.decide("rt", signals={"prior_failure": True})  # -> frontier alternate p/f2
        self.assertEqual(d.model, "p/f2")
        d = r.decide("rt", signals={"prior_failure": True})  # top: stays, no extra escalation
        self.assertEqual(d.model, "p/f2")
        self.assertEqual(d.escalation_count, 2)

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


    def test_modes_cost_balanced_quality(self):
        base = _json.loads(_json.dumps(CONFIG))
        base["decision"] = {"rule": "expected_cost", "tier_costs": {"utility": 0.1, "balanced": 1.8, "frontier": 3.84}}
        sure_balanced = {"utility": 0.0, "balanced": 1.0, "frontier": 0.0}

        r = Router(base, make_models(), classifier=StubClassifier("balanced", sure_balanced))
        self.assertEqual(r.decide("rm1", "x").tier, "utility")  # cost: cap 0.9

        cfg = _json.loads(_json.dumps(base))
        cfg["decision"]["mode"] = "balanced"
        r = Router(cfg, make_models(), classifier=StubClassifier("balanced", sure_balanced))
        self.assertEqual(r.decide("rm2", "x").tier, "balanced")  # balanced: cap 0.98

        cfg = _json.loads(_json.dumps(base))
        cfg["decision"]["mode"] = "quality"
        r = Router(cfg, make_models(), classifier=StubClassifier("utility", {"utility": 1.0, "balanced": 0.0, "frontier": 0.0}))
        self.assertEqual(r.decide("rm3", "x").tier, "balanced")  # quality: floor balanced

        cfg = _json.loads(_json.dumps(base))
        cfg["decision"]["mode"] = "quality"
        r = Router(cfg, make_models(), classifier=StubClassifier("frontier", {"utility": 0.05, "balanced": 0.05, "frontier": 0.9}))
        self.assertEqual(r.decide("rm4", "x").tier, "frontier")  # quality: very sure frontier


class RankingTests(unittest.TestCase):
    def test_value_ranking_quality_per_cost(self):
        models = make_models()
        quality = {"p/u1": 0.1, "p/u2": 0.8}
        costs = {"p/u1": 0.1, "p/u2": 0.5}
        refs = [ref for ref, _c, _q in rank_candidates(["p/u1", "p/u2"], models, quality, costs)]
        self.assertEqual(refs, ["p/u2", "p/u1"])  # 1.6 vs 1.0 value

    def test_unmeasured_sinks_and_config_untouched(self):
        models = make_models()
        quality = {"p/u1": 0.1, "p/u2": 0.8}
        costs = {"p/u1": 0.1, "p/u2": 0.5}
        ranked = apply_tier_ranking(CONFIG, models, quality, costs)
        self.assertEqual(ranked["tiers"]["utility"]["models"][0], "p/u2")
        self.assertEqual(CONFIG["tiers"]["utility"]["models"], ["p/u1", "p/u2"])
        ghost = _json.loads(_json.dumps(CONFIG))
        ghost["tiers"]["utility"]["models"].append("p/ghost")
        ranked2 = apply_tier_ranking(ghost, models, quality, costs)
        self.assertEqual(ranked2["tiers"]["utility"]["models"][-1], "p/ghost")


class AaCacheTests(unittest.TestCase):
    def test_cached_indices_returned_without_key(self):
        with tempfile.TemporaryDirectory() as td:
            data = {"gpt6": {"intelligence": 50.0}}
            with open(os.path.join(td, AA_INDICES_CACHE), "w", encoding="utf-8") as f:
                _json.dump(data, f)
            self.assertEqual(load_aa_indices(td), data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
