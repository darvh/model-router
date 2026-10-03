"""model-router CLI.

refresh     refresh models.dev catalog (and AA indices when AA_API_KEY is set)
calibrate   train the classifier on DeepSWE task instructions + trial labels
evaluate    held-out eval of the router's decisions against DeepSWE outcomes
route       decide for a run (sticky, escalates on --signal)
rank        candidate models by real cost/quality signal
tiers       show the configured combos
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

from . import __version__
from ._net import read_json
from .catalog import load_aa_indices, load_models, normalize_name, rank_candidates, tier_of_ref
from .classify import Classifier, kfold_accuracy
from .deepswe import (
    DEFAULT_VERSION,
    fetch_instructions,
    fetch_tasks,
    fetch_trials,
    labels,
    model_stats,
    task_model_stats,
)
from .engine import RunStore, Router, TIERS

DEFAULT_CACHE = os.environ.get("MODEL_ROUTER_CACHE") or os.path.join(
    os.path.expanduser("~"), ".cache", "model-router"
)


def load_config(path: str) -> dict:
    cfg = read_json(path)
    if not cfg:
        sys.exit("config not found: %s" % path)
    return cfg


def load_classifier(cache_dir: str):
    path = os.path.join(cache_dir, "classifier.json")
    if os.path.exists(path):
        return Classifier.load(path)
    return None


def cmd_refresh(args) -> None:
    models = load_models(args.cache_dir, force=args.force)
    print("models.dev: %d models cached" % len(models))
    indices = load_aa_indices(args.cache_dir, api_key=args.aa_key, force=args.force)
    if indices:
        print("AA indices: %d models" % len(indices))
    else:
        print("AA indices: none (set AA_API_KEY - free tier is 100 req/day)")


def cmd_calibrate(args) -> None:
    cfg = load_config(args.config)
    models = load_models(args.cache_dir)
    tasks = fetch_tasks(args.cache_dir, force=args.force)
    trials = fetch_trials(args.cache_dir, force=args.force)
    print("deepswe: %d tasks, %d trials" % (len(tasks), len(trials)))

    ids = [t["id"] for t in tasks]
    seen = [0]

    def progress(done, total):
        seen[0] = done
        print("\rinstructions: %d/%d" % (done, total), end="", flush=True)

    instructions = fetch_instructions(args.cache_dir, ids, force=args.force, progress=progress)
    print()

    def tier_of(ref):
        return tier_of_ref(ref, cfg, models)

    labs = labels(tasks, trials, tier_of)
    print("label distribution:", dict(Counter(labs.values())))

    examples = [(instructions[tid], lab) for tid, lab in labs.items() if instructions.get(tid)]
    if not examples:
        sys.exit("no instructions fetched; nothing to train on")

    clf = Classifier(cfg.get("classifier")).fit(examples)
    acc, confusion = kfold_accuracy(examples, k=5, thresholds=cfg.get("classifier"))
    print("5-fold CV accuracy: %.1f%%" % (acc * 100))
    for (y, p), c in sorted(confusion.items()):
        print("  %-8s -> %-8s %3d" % (y, p, c))

    clf.meta["cv_accuracy"] = acc
    clf.meta["trained_on"] = len(examples)
    path = os.path.join(args.cache_dir, "classifier.json")
    clf.save(path)
    print("saved:", path)


def cmd_evaluate(args) -> None:
    cfg = load_config(args.config)
    models = load_models(args.cache_dir)
    tasks = fetch_tasks(args.cache_dir)
    trials = fetch_trials(args.cache_dir)
    ids = [t["id"] for t in tasks]
    instructions = fetch_instructions(args.cache_dir, ids)
    stats = task_model_stats(trials)

    def tier_of(ref):
        return tier_of_ref(ref, cfg, models)

    labs = labels(tasks, trials, tier_of)
    examples = [(t["id"], instructions[t["id"]], labs[t["id"]]) for t in tasks if instructions.get(t["id"])]

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    cut = int(len(examples) * (1.0 - args.test_frac))
    train, test = examples[:cut], examples[cut:]
    clf = Classifier(cfg.get("classifier")).fit([(text, lab) for _tid, text, lab in train])
    router = Router(cfg, models, clf)

    tier_match = 0
    routed_with_stats = 0
    regrets = []
    pass_rates = []
    rows = []
    for tid, text, lab in test:
        d = router.decide("eval:%s" % tid, text)
        routed = stats.get((tid, d.model))
        if routed is None:
            for sel_tier, ref in router.ladder:
                if sel_tier == d.tier and (tid, ref) in stats:
                    routed = stats[(tid, ref)]
                    break

        reliable = [
            (ref, s)
            for (task, ref), s in stats.items()
            if task == tid and (s["pass_rate"] or 0.0) >= 0.5 and s["avg_cost_usd"] is not None
        ]
        any_pass = [
            (ref, s)
            for (task, ref), s in stats.items()
            if task == tid and (s["pass_rate"] or 0.0) > 0.0 and s["avg_cost_usd"] is not None
        ]
        pool = reliable or any_pass
        oracle = min(pool, key=lambda x: x[1]["avg_cost_usd"]) if pool else None

        tier_match += int(d.tier == lab)
        regret = None
        if routed and routed.get("avg_cost_usd") is not None:
            routed_with_stats += 1
            if routed.get("pass_rate") is not None:
                pass_rates.append(routed["pass_rate"])
            if oracle and oracle[1]["avg_cost_usd"] is not None:
                regret = routed["avg_cost_usd"] - oracle[1]["avg_cost_usd"]
                regrets.append(regret)
        rows.append(
            {
                "task": tid,
                "label": lab,
                "predicted_tier": d.tier,
                "model": d.model,
                "routed_pass_rate": routed.get("pass_rate") if routed else None,
                "routed_cost": routed.get("avg_cost_usd") if routed else None,
                "oracle_model": oracle[0] if oracle else None,
                "oracle_cost": oracle[1]["avg_cost_usd"] if oracle else None,
                "regret": regret,
            }
        )

    n = len(test)
    print("held-out tasks: %d (train %d)" % (n, len(train)))
    print("tier accuracy: %d/%d = %.1f%%" % (tier_match, n, 100.0 * tier_match / max(1, n)))
    print("routed choices with trial data: %d/%d" % (routed_with_stats, n))
    if pass_rates:
        print("mean pass rate of routed choice: %.1f%%" % (100.0 * sum(pass_rates) / len(pass_rates)))
    if regrets:
        regrets_sorted = sorted(regrets)
        print("cost regret vs cheapest-passing (USD/task): mean %.3f, median %.3f, total %.2f"
              % (sum(regrets) / len(regrets), regrets_sorted[len(regrets) // 2], sum(regrets)))
    if args.json_out:
        from ._net import write_json

        write_json(args.json_out, {"summary": {"n": n, "tier_match": tier_match}, "rows": rows})
        print("wrote:", args.json_out)


def cmd_route(args) -> None:
    cfg = load_config(args.config)
    models = load_models(args.cache_dir)
    clf = load_classifier(args.cache_dir)
    store = RunStore(os.path.join(args.cache_dir, "runs"))
    router = Router(cfg, models, clf, store)

    task = args.task or ""
    if not task and not sys.stdin.isatty():
        task = sys.stdin.read()

    signals = {s: True for s in (args.signal or [])}
    instructions = args.instruction or []
    decision = router.decide(
        args.run_id, task, signals=signals, instruction=instructions[0] if instructions else None
    )
    for extra in instructions[1:]:
        decision = router.add_instruction(args.run_id, extra) or decision
    print(json.dumps(decision.to_dict(), indent=2))


def cmd_rank(args) -> None:
    cfg = load_config(args.config)
    models = load_models(args.cache_dir)
    if args.tier:
        refs = list(cfg["tiers"][args.tier]["models"])
    else:
        refs = [ref for spec in cfg["tiers"].values() for ref in spec["models"]]

    quality = {}
    trials_path = os.path.join(args.cache_dir, "deepswe", DEFAULT_VERSION, "trials.json")
    cached = read_json(trials_path)
    if cached:
        for ref, s in model_stats(cached["rows"]).items():
            if s.get("pass_rate") is not None:
                quality[ref] = s["pass_rate"]
    aa = load_aa_indices(args.cache_dir)
    aa_by_norm = {normalize_name(m.name): m for m in models.values() if m.name}

    print("%-38s %9s %9s %8s %s" % ("model", "in$/M", "out$/M", "pass", "AA indices"))
    for ref, _blended, _q in rank_candidates(refs, models, quality):
        m = models.get(ref)
        if not m:
            print("%-38s %9s" % (ref, "(not in catalog)"))
            continue
        q = quality.get(ref)
        idx = aa_by_norm.get(normalize_name(m.name)) and aa.get(normalize_name(m.name))
        idx_str = ", ".join("%s=%.1f" % (k, v) for k, v in (idx or {}).items()) if idx else ""
        print(
            "%-38s %9.2f %9.2f %7s %s"
            % (ref, m.cost_in, m.cost_out, ("%.0f%%" % (100 * q)) if q is not None else "-", idx_str)
        )


def cmd_tiers(args) -> None:
    cfg = load_config(args.config)
    for tier in TIERS:
        spec = cfg["tiers"].get(tier)
        if not spec:
            continue
        advisor = (cfg.get("advisor") or {}).get(tier, "auto")
        print("%s: %s  (advisor: %s)" % (tier, ", ".join(spec["models"]), advisor))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="model-router", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def base(p):
        p.add_argument("--config", default="config.json", help="config JSON (default: ./config.json)")
        p.add_argument("--cache-dir", default=DEFAULT_CACHE)

    p = sub.add_parser("refresh", help="refresh catalog caches")
    base(p)
    p.add_argument("--aa-key", default=None, help="Artificial Analysis API key (or AA_API_KEY env)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_refresh)

    p = sub.add_parser("calibrate", help="train classifier on DeepSWE")
    base(p)
    p.add_argument("--force", action="store_true", help="re-download deepswe data")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("evaluate", help="held-out eval against DeepSWE outcomes")
    base(p)
    p.add_argument("--test-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("route", help="decide for a run")
    base(p)
    p.add_argument("--run-id", default="default")
    p.add_argument("--task", default=None, help="task text (else read stdin)")
    p.add_argument("--signal", action="append", default=[], help="repeatable: prior_failure, verification_divergence, not_understood, planning_needs_more_tools, needs_exploration, needs_advisor, complete")
    p.add_argument("--instruction", action="append", default=[], help="append to the run instruction stack")
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("rank", help="rank candidate models by cost + real quality signal")
    base(p)
    p.add_argument("--tier", choices=list(TIERS), default=None)
    p.set_defaults(func=cmd_rank)

    p = sub.add_parser("tiers", help="show configured combos")
    base(p)
    p.set_defaults(func=cmd_tiers)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
