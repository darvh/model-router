"""Self-checks for the decision engine, classifier, and DeepSWE label derivation. Stdlib unittest."""
import json as _json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_router.catalog import AA_INDICES_CACHE, Model, load_aa_indices, quality_from_indices, rank_candidates, tier_of_ref
from model_router.classify import Classifier, frustration_score
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


MEASURED = {"p/u1": 0.10, "p/u2": 0.30, "p/b1": 2.00, "p/f1": 5.00, "p/f2": 9.00}


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
        self.assertEqual(d1.model, "p/u1")  # cost mode: cheapest candidate
        self.assertEqual(d2.model, d1.model)
        self.assertTrue(d2.sticky)
        self.assertEqual(d2.turn, 2)
        self.assertFalse(d2.advisor_required)
        self.assertEqual(d2.advisor_reason, "")

    def test_escalation(self):
        self.router.decide("r2", "x")
        d = self.router.decide("r2", signals={"prior_failure": True})
        self.assertEqual(d.tier, "balanced")
        self.assertEqual(d.model, "p/b1")
        self.assertEqual(d.escalation_count, 1)
        self.assertIn("ESC utility->balanced", d.instructions)
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
        self.assertIn("U", d.instructions)
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

    def test_modes_optimize_cost_quality_balanced(self):
        quality = {"p/u1": 0.2, "p/u2": 0.7, "p/b1": 0.95, "p/f1": 0.9, "p/f2": 0.9}
        probs = {"utility": 0.0, "balanced": 1.0, "frontier": 0.0}  # required = 0.6
        base = _json.loads(_json.dumps(CONFIG))
        base["decision"] = {"mode": "cost", "tier_requirement": {"utility": 0.35, "balanced": 0.6, "frontier": 0.8}}

        r = Router(base, make_models(), classifier=StubClassifier("balanced", probs), quality=quality)
        d = r.decide("rm1", "x")
        self.assertEqual(d.model, "p/u2")  # cheapest feasible (u1 gated out by Q < r)
        self.assertAlmostEqual(d.rationale["required_quality"], 0.6)

        cfg = _json.loads(_json.dumps(base))
        cfg["decision"]["mode"] = "quality"
        r = Router(cfg, make_models(), classifier=StubClassifier("balanced", probs), quality=quality)
        self.assertEqual(r.decide("rm2", "x").model, "p/b1")  # max quality

        cfg = _json.loads(_json.dumps(base))
        cfg["decision"]["mode"] = "balanced"
        r = Router(cfg, make_models(), classifier=StubClassifier("balanced", probs), quality=quality)
        self.assertEqual(r.decide("rm3", "x").model, "p/b1")  # max arithmetic mean of Q and cost score

    def test_numeric_signals_escalate_and_advise(self):
        self.router.decide("rs1", "x")  # u1
        d = self.router.decide("rs1", signals={"user_frustration": 0.8})
        self.assertEqual(d.tier, "balanced")
        self.assertTrue(d.advisor_required)
        self.assertEqual(d.advisor_reason, "user_frustration")
        self.assertIn("user_frustration", d.rationale["signals_applied"])

    def test_numeric_signals_below_threshold_noop(self):
        self.router.decide("rs2", "x")
        d = self.router.decide("rs2", signals={"tool_error_rate": 0.1})
        self.assertEqual(d.tier, "utility")
        self.assertEqual(d.escalation_count, 0)
        self.assertEqual(d.rationale["signals_applied"], [])

    def test_signal_stacking_caps_escalation(self):
        self.router.decide("rs3", "x")
        d = self.router.decide(
            "rs3",
            signals={"prior_failure": True, "tool_error_rate": 0.9, "user_frustration": 0.9},
        )
        self.assertEqual(d.escalation_count, 1)  # at most one tier per decision
        self.assertTrue(d.advisor_required)

    def test_signal_rules_can_be_disabled(self):
        cfg = _json.loads(_json.dumps(CONFIG))
        cfg["decision"] = {"mode": "cost", "signal_rules": {"user_frustration": {"enabled": False}}}
        r = Router(cfg, make_models(), classifier=StubClassifier("balanced"))
        r.decide("rs4", "x")
        d = r.decide("rs4", signals={"user_frustration": 0.9})
        self.assertEqual(d.escalation_count, 0)

    def test_tool_error_rate_escalates(self):
        self.router.decide("rs5", "x")
        d = self.router.decide("rs5", signals={"tool_error_rate": 0.5})
        self.assertEqual(d.tier, "balanced")
        self.assertEqual(d.escalation_count, 1)
        self.assertEqual(d.advisor_reason, "tool_error_rate")

    def test_floor_and_failure_capped_at_one_tier(self):
        r = Router(CONFIG, make_models(), classifier=StubClassifier("utility"))
        r.decide("rcap", "x")  # u1
        d = r.decide("rcap", signals={"not_understood": True, "prior_failure": True})
        self.assertEqual(d.tier, "balanced")  # floor applied; failure only flags advisor
        self.assertEqual(d.escalation_count, 1)
        self.assertTrue(d.advisor_required)

    def test_run_store_corrupt_and_old_schema(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "bad.json"), "w", encoding="utf-8") as f:
                f.write("{not json")
            store = RunStore(td)
            self.assertIsNone(store.get("bad"))
            old = {"run_id": "old", "ladder_index": 1, "turn": 3, "advisor_required": True, "bogus": 1}
            with open(os.path.join(td, "old.json"), "w", encoding="utf-8") as f:
                _json.dump(old, f)
            state = store.get("old")
            self.assertEqual(state.turn, 3)
            self.assertEqual(state.signals_applied, [])

    def test_quality_modes_ignore_unmeasured_models(self):
        base = _json.loads(_json.dumps(CONFIG))
        base["decision"] = {"mode": "quality"}
        quality = {"p/b1": 0.3}  # only b1 measured; unknown models would score 0.5 and win
        r = Router(base, make_models(), classifier=StubClassifier("balanced"), quality=quality)
        self.assertEqual(r.decide("rq1", "x").model, "p/b1")

    def test_effort_defaults_and_overrides(self):
        r = Router(CONFIG, make_models(), classifier=StubClassifier("balanced"))
        d = r.decide("re1", "x")
        self.assertEqual(d.effort, "low")  # utility default
        self.assertEqual(d.effort_params, {})  # no reasoning options in fixture
        cfg = _json.loads(_json.dumps(CONFIG))
        cfg["efforts"] = {"p/u1": "high"}
        r2 = Router(cfg, make_models(), classifier=StubClassifier("balanced"))
        self.assertEqual(r2.decide("re2", "x").effort, "high")  # ref override wins
        cfg["efforts"] = {"utility": "medium"}
        r3 = Router(cfg, make_models(), classifier=StubClassifier("balanced"))
        self.assertEqual(r3.decide("re3", "x").effort, "medium")

    def test_effort_params_normalization(self):
        models = make_models()
        models["p/u1"] = Model("p/u1", "p", "u1", "U1", 0.1, 0.4, 128000, True, True, "2026-01-01",
                               reasoning_options=[{"type": "effort", "values": ["low", "high", "max"]}])
        models["p/u2"] = Model("p/u2", "p", "u2", "U2", 0.1, 1.0, 128000, True, True, "2026-01-01",
                               reasoning_options=[{"type": "budget_tokens", "min": 2048}])
        models["p/b1"] = Model("p/b1", "p", "b1", "B1", 2.0, 10.0, 128000, True, True, "2026-01-01",
                               reasoning_options=[{"type": "toggle"}])
        r = Router(CONFIG, models, classifier=StubClassifier("balanced"))
        d = r.decide("rp1", "x")  # cost mode picks p/u1
        self.assertEqual(d.effort_params, {"reasoning_effort": "low"})
        self.assertEqual(r._resolve_effort_params("p/u1", "medium"), {"reasoning_effort": "high"})  # rounds up
        self.assertEqual(r._resolve_effort_params("p/u2", "medium"), {"thinking_budget_tokens": 8192})
        cfg = _json.loads(_json.dumps(CONFIG))
        cfg["effort_budgets"] = {"medium": 5000}
        r2 = Router(cfg, models, classifier=StubClassifier("balanced"))
        self.assertEqual(r2._resolve_effort_params("p/u2", "medium"), {"thinking_budget_tokens": 5000})
        self.assertEqual(r._resolve_effort_params("p/b1", "medium"), {"reasoning": True})
        self.assertEqual(r._resolve_effort_params("p/b1", "none"), {"reasoning": False})

    def test_anchors_resolution(self):
        base = _json.loads(_json.dumps(CONFIG))
        base["decision"] = {"mode": "cost"}
        probs = {"utility": 0.0, "balanced": 1.0, "frontier": 0.0}
        anchors = {"utility": 0.2, "balanced": 0.4, "frontier": 0.6}
        r = Router(base, make_models(), classifier=StubClassifier("balanced", probs), anchors=anchors)
        self.assertAlmostEqual(r.decide("ra1", "x").rationale["required_quality"], 0.4)
        base["decision"]["tier_requirement"] = {"utility": 0.3, "balanced": 0.9, "frontier": 0.95}
        r2 = Router(base, make_models(), classifier=StubClassifier("balanced", probs), anchors=anchors)
        self.assertAlmostEqual(r2.decide("ra2", "x").rationale["required_quality"], 0.9)

    def test_measured_costs_override_blended(self):
        # u1 has a lower token price but expensive measured runs; u2 is cheaper measured
        r = Router(
            CONFIG, make_models(), classifier=StubClassifier("balanced"),
            costs={"p/u1": 5.0, "p/u2": 0.01},
        )
        self.assertEqual(r.decide("rcost", "x").model, "p/u2")

    def test_domain_quality_selection(self):
        r = Router(CONFIG, make_models(), classifier=StubClassifier("balanced"))
        r.quality["p/u1"] = {
            "artificial_analysis_coding_index": 30.0,
            "artificial_analysis_intelligence_index": 90.0,
        }
        self.assertAlmostEqual(r._quality_of("p/u1", "coding"), 0.3)
        self.assertAlmostEqual(r._quality_of("p/u1", "math"), 0.9)
        r.quality["p/u2"] = 0.55  # flat float still works
        self.assertAlmostEqual(r._quality_of("p/u2", "coding"), 0.55)

    def test_outcome_blending_and_recording(self):
        outcomes = {}
        r = Router(
            CONFIG, make_models(), classifier=StubClassifier("balanced"),
            quality={"p/u1": 0.5}, outcomes=outcomes,
        )
        self.assertAlmostEqual(r._quality_of("p/u1", "general"), 0.5)
        outcomes["p/u1|general"] = {"n": 9, "successes": 9, "cost_sum": 0.0, "cost_n": 0}
        self.assertGreater(r._quality_of("p/u1", "general"), 0.7)  # observations beat the prior

        r.decide("ro1", "x")
        info = r.record_outcome("ro1", success=True, cost_usd=0.12)
        self.assertEqual(info["ref"], "p/u1")
        self.assertEqual(outcomes["p/u1|general"]["n"], 10)  # 9 pre-seeded + 1 recorded
        self.assertEqual(outcomes["p/u1|general"]["successes"], 10)
        self.assertEqual(outcomes["p/u1|general"]["cost_n"], 1)
        self.assertIsNone(r.record_outcome("missing", success=True))

    def test_budget_blocks_escalation(self):
        cfg = _json.loads(_json.dumps(CONFIG))
        cfg["decision"] = {"mode": "cost", "budget_usd": 1.0}
        r = Router(cfg, make_models(), classifier=StubClassifier("balanced"), costs=MEASURED)
        r.decide("rb1", "x")  # u1
        d = r.decide("rb1", signals={"prior_failure": True})  # b1 costs more than the budget
        self.assertEqual(d.tier, "utility")
        self.assertEqual(d.escalation_count, 0)
        self.assertTrue(d.advisor_required)
        self.assertEqual(d.advisor_reason, "budget")

    def test_budget_uses_spent(self):
        r = Router(CONFIG, make_models(), classifier=StubClassifier("balanced"), costs=MEASURED)
        r.decide("rb2", "x", budget_usd=0.5)
        d = r.decide("rb2", signals={"prior_failure": True})
        self.assertEqual(d.tier, "utility")
        r.record_outcome("rb2", success=True, cost_usd=0.4)  # spend accumulates
        d = r.decide("rb2", signals={"prior_failure": True})
        self.assertEqual(d.escalation_count, 0)

    def test_estimated_cost_reported(self):
        r = Router(CONFIG, make_models(), classifier=StubClassifier("balanced"), costs=MEASURED)
        self.assertGreater(r.decide("rb3", "x").estimated_cost_usd, 0.0)

    def test_cost_source_reported(self):
        r = Router(CONFIG, make_models(), classifier=StubClassifier("balanced"), costs=MEASURED)
        self.assertEqual(r.decide("rc1", "x").cost_source, "measured")
        bare = Router(CONFIG, make_models(), classifier=StubClassifier("balanced"))
        self.assertEqual(bare.decide("rc2", "x").cost_source, "unknown")

    def test_cost_units_never_mix(self):
        """Catalog price is per 1M tokens: it may order models, never pose as $/task."""
        bare = Router(CONFIG, make_models(), classifier=StubClassifier("balanced"))
        self.assertIsNotNone(bare._ordering_cost("p/u1"))  # ranking works with no trials
        self.assertIsNone(bare._task_cost("p/u1"))  # ...but no dollar figure exists
        self.assertEqual(bare.decide("rc3", "x").estimated_cost_usd, 0.0)
        self.assertEqual(bare.decide("rc3b", "x").cost_source, "unknown")

    def test_config_cost_override(self):
        cfg = _json.loads(_json.dumps(CONFIG))
        cfg["costs"] = {"p/b1": 4.0}
        r = Router(cfg, make_models(), classifier=StubClassifier("balanced"))
        self.assertEqual(r._task_cost("p/b1"), 4.0)
        self.assertEqual(r._cost_source("p/b1"), "measured")

    def test_budget_downgrades_initial_pick(self):
        """A budget below the chosen model must move the first pick down, not fail."""
        cfg = _json.loads(_json.dumps(CONFIG))
        cfg["decision"] = {"mode": "quality"}
        r = Router(cfg, make_models(), classifier=StubClassifier("frontier"), costs=MEASURED)
        d = r.decide("rb5", "x", budget_usd=1.0)
        self.assertLess(d.estimated_cost_usd, 100.0)
        self.assertNotEqual(d.model, MEASURED and "p/f2")

    def test_budget_unaffordable_flags_advisor(self):
        cfg = _json.loads(_json.dumps(CONFIG))
        cfg["decision"] = {"mode": "quality"}
        r = Router(cfg, make_models(), classifier=StubClassifier("frontier"), costs=MEASURED)
        d = r.decide("rb6", "x", budget_usd=0.0)
        self.assertTrue(d.advisor_required)
        self.assertEqual(d.advisor_reason, "budget")

    def test_unpriced_model_never_blocks_a_budget(self):
        """No $/task figure means no budget claim either way: route, don't refuse."""
        r = Router(CONFIG, make_models(), classifier=StubClassifier("balanced"))
        d = r.decide("rb7", "x", budget_usd=0.01)
        self.assertFalse(d.advisor_required)
        self.assertEqual(d.cost_source, "unknown")
        self.assertEqual(d.estimated_cost_usd, 0.0)

    def test_budget_flags_unpriced_escalation(self):
        """Escalating to a candidate with no $/task must not imply the cap held."""
        r = Router(CONFIG, make_models(), classifier=StubClassifier("balanced"), costs={"p/u1": 0.1})
        d = r.decide("rb8", "x", budget_usd=5.0, signals={"prior_failure": True})
        self.assertEqual(d.tier, "balanced")
        self.assertEqual(d.cost_source, "unknown")
        self.assertEqual(d.advisor_reason, "budget_unpriced")

    def test_decision_log_written(self):
        tmp = tempfile.mkdtemp()
        store = RunStore(os.path.join(tmp, "runs"))
        r = Router(
            CONFIG, make_models(), classifier=StubClassifier("balanced"), store=store, costs=MEASURED
        )
        r.decide("rl1", "fix the parser")
        log = os.path.join(tmp, "decisions.jsonl")
        self.assertTrue(os.path.exists(log))
        row = _json.loads(open(log).readline())
        self.assertEqual(row["model"], "p/u1")
        self.assertEqual(row["tier"], "utility")
        self.assertEqual(row["cost_source"], "measured")

    def test_cache_age_hours(self):
        from model_router.catalog import cache_age_hours

        self.assertIsNone(cache_age_hours(os.path.join(tempfile.mkdtemp(), "nope.json")))
        fresh = os.path.join(tempfile.mkdtemp(), "fresh.json")
        open(fresh, "w").close()
        self.assertLess(cache_age_hours(fresh), 1.0)

    def test_latency_reported(self):
        r = Router(
            CONFIG, make_models(), classifier=StubClassifier("balanced"),
            latency={"p/u1": {"median_time_to_first_token_seconds": 1.5}},
        )
        d = r.decide("rl1", "x")
        self.assertEqual(d.latency["median_time_to_first_token_seconds"], 1.5)

    def test_performance_extraction(self):
        from model_router.catalog import performance_from_entry

        entry = {
            "performance": {
                "median_time_to_first_token_seconds": 15.31,
                "median_output_tokens_per_second": 239.12,
                "notes": "x",
            }
        }
        perf = performance_from_entry(entry)
        self.assertEqual(perf["median_time_to_first_token_seconds"], 15.31)
        self.assertNotIn("notes", perf)
        self.assertEqual(performance_from_entry(None), {})

    def test_external_tier_override(self):
        """A harness classifier can supply the tier; the rest of the pipeline still holds."""
        r = Router(CONFIG, make_models(), classifier=StubClassifier("utility"), costs=MEASURED)
        d = r.decide("ro1", "x", tier="balanced")
        self.assertEqual(d.tier, "balanced")
        self.assertEqual(d.rationale["classifier"], "external")
        self.assertEqual(d.rationale["probs"]["balanced"], 1.0)
        with self.assertRaises(ValueError):
            r.decide("ro2", "x", tier="nonsense")

    def test_ab_arm_assignment_is_deterministic(self):
        cfg = _json.loads(_json.dumps(CONFIG))
        cfg["ab"] = {"enabled": True, "split": 0.5, "baseline_tier": "frontier"}
        r = Router(cfg, make_models(), classifier=StubClassifier("utility"), costs=MEASURED)
        arms = {r._assign_arm("run-%d" % i) for i in range(200)}
        self.assertEqual(arms, {"router", "baseline"})  # both arms get traffic
        self.assertEqual(r._assign_arm("run-7"), r._assign_arm("run-7"))  # stable per run

    def test_ab_baseline_is_fixed_control(self):
        cfg = _json.loads(_json.dumps(CONFIG))
        cfg["ab"] = {"enabled": True, "split": 0.0, "baseline_tier": "frontier"}  # everyone baseline
        r = Router(cfg, make_models(), classifier=StubClassifier("utility"), costs=MEASURED)
        d = r.decide("rab1", "x", signals={"prior_failure": True, "user_frustration": 0.9})
        self.assertEqual(d.arm, "baseline")
        self.assertEqual(d.tier, "frontier")
        self.assertEqual(d.model, "p/f1")
        self.assertEqual(d.escalation_count, 0)  # no escalation in the control
        d2 = r.decide("rab1", "x", signals={"prior_failure": True})
        self.assertEqual(d2.model, "p/f1")  # sticky

    def test_ab_router_arm_escalates_normally(self):
        cfg = _json.loads(_json.dumps(CONFIG))
        cfg["ab"] = {"enabled": True, "split": 1.0, "baseline_tier": "frontier"}  # everyone router
        r = Router(cfg, make_models(), classifier=StubClassifier("utility"), costs=MEASURED)
        d = r.decide("rab2", "x", signals={"prior_failure": True})
        self.assertEqual(d.arm, "router")
        self.assertEqual(d.escalation_count, 1)

    def test_record_outcome_counts_successes_failures(self):
        r = Router(CONFIG, make_models(), classifier=StubClassifier("balanced"))
        r.decide("rc5", "x")
        r.record_outcome("rc5", True, cost_usd=0.2)
        r.record_outcome("rc5", False)
        state = r.store.get("rc5")
        self.assertEqual((state.successes, state.failures), (1, 1))

    def test_two_proportion_p(self):
        from model_router.cli import _two_proportion_p

        self.assertLess(_two_proportion_p(90, 100, 50, 100), 0.001)
        self.assertGreater(_two_proportion_p(50, 100, 52, 100), 0.5)
        self.assertEqual(_two_proportion_p(0, 0, 5, 10), 1.0)

    def test_escalation_caps_at_top(self):
        r = Router(CONFIG, make_models(), classifier=StubClassifier("balanced"))
        r.decide("rt", "x")  # u1
        d = r.decide("rt", signals={"prior_failure": True})  # -> balanced b1
        self.assertEqual(d.model, "p/b1")
        d = r.decide("rt", signals={"prior_failure": True})  # -> frontier f1
        self.assertEqual(d.model, "p/f1")
        d = r.decide("rt", signals={"prior_failure": True})  # -> frontier alternate f2
        self.assertEqual(d.model, "p/f2")
        d = r.decide("rt", signals={"prior_failure": True})  # top: stays, no extra escalation
        self.assertEqual(d.model, "p/f2")
        self.assertEqual(d.escalation_count, 3)

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

    def test_task_domain(self):
        from model_router.classify import task_domain

        self.assertEqual(task_domain("Refactor the auth middleware and add tests"), "coding")
        self.assertEqual(task_domain("Prove the theorem and derive the integral"), "math")
        self.assertEqual(task_domain("Write a blog article and summarize the tone"), "writing")
        self.assertEqual(task_domain(""), "general")

    def test_frustration_score(self):
        self.assertGreaterEqual(frustration_score(["still not working!!", "fix it again"]), 0.5)
        self.assertGreaterEqual(
            frustration_score(["the pagination helper is broken", "the pagination helper is still broken"]),
            0.5,
        )
        self.assertGreater(frustration_score(["FIX THE PAGINATION NOW"]), 0.0)
        self.assertEqual(frustration_score(["add a unit test"]), 0.0)
        self.assertEqual(frustration_score([]), 0.0)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            examples = [("small quick fix %d" % i, "utility") for i in range(5)]
            examples += [("huge complex redesign %d" % i, "frontier") for i in range(5)]
            clf = Classifier().fit(examples)
            clf.temperature = 2.0
            path = os.path.join(td, "classifier.json")
            clf.save(path)
            clf2 = Classifier.load(path)
            self.assertEqual(clf2.temperature, 2.0)
            self.assertEqual(clf.predict("small quick fix")[0], clf2.predict("small quick fix")[0])
            self.assertEqual(clf.predict("huge complex redesign")[0], clf2.predict("huge complex redesign")[0])

    def test_temperature_softens(self):
        from model_router.classify import apply_temperature

        probs = {"utility": 0.98, "balanced": 0.02}
        softened = apply_temperature(probs, 4.0)
        self.assertLess(softened["utility"], 0.98)
        self.assertAlmostEqual(sum(softened.values()), 1.0)


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


class RankingTests(unittest.TestCase):
    def test_value_ranking_quality_per_cost(self):
        models = make_models()
        quality = {"p/u1": 0.1, "p/u2": 0.8}
        costs = {"p/u1": 0.1, "p/u2": 0.5}
        refs = [ref for ref, _c, _q in rank_candidates(["p/u1", "p/u2"], models, quality, costs)]
        self.assertEqual(refs, ["p/u2", "p/u1"])  # 1.6 vs 1.0 value

    def test_quality_from_indices(self):
        self.assertAlmostEqual(quality_from_indices({"artificial_analysis_coding_index": 81.0}), 0.81)
        self.assertAlmostEqual(quality_from_indices({"intelligence": 0.9}), 0.009)
        self.assertAlmostEqual(quality_from_indices({"coding": 120.0}), 1.0)
        self.assertEqual(
            quality_from_indices(
                {"artificial_analysis_intelligence_index": 50.0, "artificial_analysis_coding_index": 80.0}
            ),
            0.8,
        )
        self.assertIsNone(quality_from_indices(None))
        self.assertIsNone(quality_from_indices({"math": 70.0}))
        self.assertIsNone(quality_from_indices({"artificial_analysis_intelligence_index_cost": 12.0}))

    def test_extract_aa_entries(self):
        from model_router.catalog import _extract_aa_entries, canon

        entries = [
            {
                "slug": "gpt-6-astra",
                "name": "GPT-6 Astra",
                "evaluations": {
                    "artificial_analysis_coding_index": 76.9,
                    "artificial_analysis_intelligence_index": 52.7,
                    "artificial_analysis_intelligence_index_cost": 4.2,
                    "notes": "x",
                },
            },
            {"name": "No Slug Model", "evaluations": {"artificial_analysis_intelligence_index": 30.0}},
            {"slug": "no-evals"},
            "junk",
        ]
        out = _extract_aa_entries(entries)
        self.assertIn(canon("gpt-6-astra"), out)
        self.assertNotIn("artificial_analysis_intelligence_index_cost", out[canon("gpt-6-astra")])
        self.assertIn(canon("No Slug Model"), out)
        self.assertEqual(len(out), 2)

    def test_build_quality_aa_overrides(self):
        from model_router.catalog import AA_INDICES_CACHE, canon
        from model_router.cli import build_quality

        with tempfile.TemporaryDirectory() as td:
            idx = {canon("gpt-x"): {"artificial_analysis_coding_index": 80.0}}
            with open(os.path.join(td, AA_INDICES_CACHE), "w", encoding="utf-8") as f:
                _json.dump(idx, f)
            models = {"p/gpt-x": Model("p/gpt-x", "p", "gpt-x", "GPT-X", 1, 2, 128000, True, True, "2026-01-01")}
            quality = build_quality(models, td)
            self.assertAlmostEqual(quality_from_indices(quality["p/gpt-x"]), 0.8)


class AaCacheTests(unittest.TestCase):
    def test_cached_indices_returned_without_key(self):
        with tempfile.TemporaryDirectory() as td:
            data = {"gpt6": {"intelligence": 50.0}}
            with open(os.path.join(td, AA_INDICES_CACHE), "w", encoding="utf-8") as f:
                _json.dump(data, f)
            self.assertEqual(load_aa_indices(td), data)


class CliHelpersTests(unittest.TestCase):
    def test_dotenv_parsing(self):
        from model_router.cli import _load_dotenv

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, ".env")
            with open(path, "w", encoding="utf-8") as f:
                f.write('# comment\nexport AA_API_KEY="secret-1"  # inline\nPLAIN=plain-value\n')
            saved = {k: os.environ.pop(k, None) for k in ("AA_API_KEY", "PLAIN")}
            try:
                _load_dotenv(path)
                self.assertEqual(os.environ["AA_API_KEY"], "secret-1")
                self.assertEqual(os.environ["PLAIN"], "plain-value")
                os.environ["PLAIN"] = "existing"
                _load_dotenv(path)
                self.assertEqual(os.environ["PLAIN"], "existing")  # never overrides
            finally:
                for k, v in saved.items():
                    os.environ.pop(k, None)
                    if v is not None:
                        os.environ[k] = v

    def test_parse_signals(self):
        from model_router.cli import _parse_signals

        signals = _parse_signals(["tool_error_rate=0.7"], ["still not working!!", "fix it again"], {})
        self.assertEqual(signals["tool_error_rate"], 0.7)
        self.assertGreaterEqual(signals["user_frustration"], 0.5)
        with self.assertRaises(SystemExit):
            _parse_signals(["nope"], [], {})
        with self.assertRaises(SystemExit):
            _parse_signals(["tool_error_rate=abc"], [], {})

    def test_calibrate_anchors(self):
        from model_router.cli import _calibrate_anchors

        labels_map = {"t1": "utility", "t2": "balanced"}
        stats = {
            ("t1", "p/u1"): {"pass_rate": 1.0},
            ("t1", "p/b1"): {"pass_rate": 1.0},
            ("t2", "p/u1"): {"pass_rate": 0.0},
            ("t2", "p/b1"): {"pass_rate": 0.8},
        }
        quality = {"p/u1": 0.5, "p/b1": 0.9}
        anchors = _calibrate_anchors(labels_map, stats, quality)
        self.assertEqual(anchors["utility"], 0.5)
        self.assertEqual(anchors["balanced"], 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
