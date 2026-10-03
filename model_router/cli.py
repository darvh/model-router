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
from ._net import read_json, write_json
from .catalog import (
    build_model_index,
    canon,
    load_aa_indices,
    load_models,
    quality_from_indices,
    rank_candidates,
    resolve_ref,
    tier_of_ref,
)
from .classify import Classifier, calibrate_temperature, frustration_score, kfold_accuracy
from .deepswe import (
    DEFAULT_VERSION,
    fetch_instructions,
    fetch_tasks,
    fetch_trials,
    labels,
    model_stats,
    task_model_stats,
)
from .engine import ADVISOR_SIGNAL, DEFAULT_SIGNAL_RULES, ESCALATING_SIGNALS, FLOOR_SIGNALS, RunStore, Router, TIERS

DEFAULT_CACHE = os.environ.get("MODEL_ROUTER_CACHE") or os.path.join(
    os.path.expanduser("~"), ".cache", "model-router"
)


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, never overrides existing env, never logs values."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _sep, value = line.partition("=")
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            value = value.strip()
            if " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            value = value.strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_config(path: str) -> dict:
    cfg = read_json(path)
    if not cfg:
        sys.exit("config not found: %s" % path)
    return cfg


def load_classifier(cache_dir: str):
    path = os.path.join(cache_dir, "classifier.json")
    if os.path.exists(path):
        try:
            return Classifier.load(path)
        except Exception:
            return Classifier()  # corrupt classifier: cold heuristic start
    return Classifier()  # heuristic cold start until `calibrate` runs


def _parse_signals(items, messages, cfg) -> dict:
    signals = {}
    rules = (cfg.get("decision") or {}).get("signal_rules")
    known = set(FLOOR_SIGNALS) | set(ESCALATING_SIGNALS) | {ADVISOR_SIGNAL, "complete"}
    known |= set(rules if rules is not None else DEFAULT_SIGNAL_RULES)
    for item in items or []:
        name, _sep, raw = item.partition("=")
        if name not in known:
            sys.exit("unknown signal: %s (known: %s)" % (name, ", ".join(sorted(known))))
        if raw:
            try:
                signals[name] = float(raw)
            except ValueError:
                sys.exit("bad signal value: %s=%s" % (name, raw))
        else:
            signals[name] = True
    if messages:
        score = frustration_score(messages)
        if score > 0:
            previous = signals.get("user_frustration", 0.0)
            previous = 1.0 if previous is True else float(previous)
            signals["user_frustration"] = max(previous, score)
    return signals


def make_ref_of(models):
    """Trial (provider, model) -> models.dev ref, with fallback to the raw pair."""
    index = build_model_index(models)

    def ref_of(t):
        return resolve_ref(t.get("provider"), t.get("model"), models, index) or (
            "%s/%s" % (t.get("provider"), t.get("model"))
        )

    return ref_of


def build_quality(models, cache_dir):
    """ref -> 0..1 capability: DeepSWE pass rate, overridden by AA indices when available."""
    quality = {}
    cached = read_json(os.path.join(cache_dir, "deepswe", DEFAULT_VERSION, "trials.json"))
    if cached:
        for ref, s in model_stats(cached["rows"], ref_of=make_ref_of(models)).items():
            if s.get("pass_rate") is not None:
                quality[ref] = s["pass_rate"]
    aa = load_aa_indices(cache_dir)
    if aa:
        for ref, m in models.items():
            entry = aa.get(canon(m.id)) or aa.get(canon(m.name))
            q = quality_from_indices(entry)
            if q is not None:
                quality[ref] = q
    return quality


def _load_anchors(cache_dir):
    return read_json(os.path.join(cache_dir, "anchors.json")) or {}


def _calibrate_anchors(labels_map, task_stats, quality):
    """Median observed required quality per tier: min Q among models that reliably passed."""
    per_tier = {t: [] for t in TIERS}
    for tid, label in labels_map.items():
        qs = [
            quality[ref]
            for (task, ref), s in task_stats.items()
            if task == tid and (s.get("pass_rate") or 0.0) >= 0.5 and ref in quality and quality[ref] is not None
        ]
        if qs:
            per_tier.setdefault(label, []).append(min(qs))
    anchors = {}
    for tier in TIERS:
        vals = sorted(per_tier.get(tier, []))
        if vals:
            anchors[tier] = round(vals[len(vals) // 2], 3)
    prev = 0.0
    for tier in TIERS:  # enforce monotonicity
        if tier in anchors:
            anchors[tier] = max(anchors[tier], prev)
            prev = anchors[tier]
    return anchors


def cmd_refresh(args) -> None:
    models = load_models(args.cache_dir, force=args.force)
    print("models.dev: %d models cached" % len(models))
    try:
        indices = load_aa_indices(args.cache_dir, api_key=args.aa_key, force=args.force, raise_errors=True)
    except Exception as exc:
        print("AA fetch failed: %s" % exc)
        indices = load_aa_indices(args.cache_dir, api_key=args.aa_key)
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

    labs = labels(tasks, trials, tier_of, ref_of=make_ref_of(models))
    print("label distribution:", dict(Counter(labs.values())))

    stats = task_model_stats(trials, ref_of=make_ref_of(models))
    quality = build_quality(models, args.cache_dir)
    anchors = _calibrate_anchors(labs, stats, quality)
    if anchors:
        write_json(os.path.join(args.cache_dir, "anchors.json"), anchors)
        print("calibrated anchors:", anchors)
    else:
        print("calibrated anchors: none (falling back to defaults)")

    examples = [(instructions[tid], lab) for tid, lab in labs.items() if instructions.get(tid)]
    if not examples:
        sys.exit("no instructions fetched; nothing to train on")

    ls = float((cfg.get("classifier") or {}).get("label_smoothing", 0.0))
    clf = Classifier(cfg.get("classifier")).fit(examples, label_smoothing=ls)
    acc, confusion = kfold_accuracy(examples, k=5, thresholds=cfg.get("classifier"), label_smoothing=ls)
    print("5-fold CV accuracy: %.1f%%" % (acc * 100))
    for (y, p), c in sorted(confusion.items()):
        print("  %-8s -> %-8s %3d" % (y, p, c))

    temp, nll = calibrate_temperature(examples)
    clf.temperature = temp
    print("probability temperature: %.2f (out-of-fold NLL %.3f)" % (temp, nll))

    clf.meta["cv_accuracy"] = acc
    clf.meta["trained_on"] = len(examples)
    path = os.path.join(args.cache_dir, "classifier.json")
    clf.save(path)
    print("saved:", path)


def cmd_evaluate(args) -> None:
    cfg = load_config(args.config)
    models = load_models(args.cache_dir)
    quality = build_quality(models, args.cache_dir)
    tasks = fetch_tasks(args.cache_dir)
    trials = fetch_trials(args.cache_dir)
    ids = [t["id"] for t in tasks]
    instructions = fetch_instructions(args.cache_dir, ids)
    ref_of = make_ref_of(models)
    stats = task_model_stats(trials, ref_of=ref_of)

    def tier_of(ref):
        return tier_of_ref(ref, cfg, models)

    labs = labels(tasks, trials, tier_of, ref_of=ref_of)
    examples = [(t["id"], instructions[t["id"]], labs[t["id"]]) for t in tasks if instructions.get(t["id"])]

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    cut = int(len(examples) * (1.0 - args.test_frac))
    train, test = examples[:cut], examples[cut:]
    clf = Classifier(cfg.get("classifier")).fit(
        [(text, lab) for _tid, text, lab in train],
        label_smoothing=float((cfg.get("classifier") or {}).get("label_smoothing", 0.0)),
    )
    router = Router(cfg, models, clf, quality=quality, anchors=_load_anchors(args.cache_dir))

    tier_match = 0
    routed_with_stats = 0
    regrets = []
    pass_rates = []
    unsolved = 0
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
            if (routed.get("pass_rate") or 0.0) > 0.0:
                if oracle and oracle[1]["avg_cost_usd"] is not None:
                    regret = routed["avg_cost_usd"] - oracle[1]["avg_cost_usd"]
                    regrets.append(regret)
            else:
                unsolved += 1
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

    # --- policy simulation on held-out tasks: real trial costs, escalation on failure
    def chain(reps, start, max_steps):
        i = TIERS.index(start)
        fail, cost, succ, used = 1.0, 0.0, 0.0, 0
        for tier in TIERS[i : i + max_steps]:
            rep = reps.get(tier)
            if not rep:
                break
            p, c = rep[1]["pass_rate"], rep[1]["avg_cost_usd"]
            if p is None or c is None:
                break
            cost += fail * c
            succ += fail * p
            fail *= 1.0 - p
            used += 1
        return (cost, succ) if used else None

    def reps_for(tid):
        reps = {}
        for tier in TIERS:
            for ref in cfg["tiers"][tier]["models"]:
                s = stats.get((tid, ref))
                if s and s["pass_rate"] is not None and s["avg_cost_usd"] is not None:
                    reps[tier] = (ref, s)
                    break
        return reps

    policies = {
        "always-utility": ("utility", 1),
        "always-balanced": ("balanced", 1),
        "always-frontier": ("frontier", 1),
        "balanced->frontier": ("balanced", 2),
        "utility->balanced->frontier": ("utility", 3),
    }
    agg = {name: {"n": 0, "cost": 0.0, "succ": 0.0} for name in list(policies) + ["router"]}
    for row in rows:
        reps = reps_for(row["task"])
        for name, (start, steps) in policies.items():
            out = chain(reps, start, steps)
            if out:
                agg[name]["n"] += 1
                agg[name]["cost"] += out[0]
                agg[name]["succ"] += out[1]
        out = chain(reps, row["predicted_tier"], 3)
        if out:
            agg["router"]["n"] += 1
            agg["router"]["cost"] += out[0]
            agg["router"]["succ"] += out[1]

    print("policy simulation (cost USD/task, success = pass rate):")
    print("  %-30s %5s %8s %8s %12s" % ("policy", "n", "cost", "success", "cost/success"))
    for name, a in agg.items():
        if a["n"] == 0:
            continue
        c = a["cost"] / a["n"]
        s = a["succ"] / a["n"]
        cps = (c / s) if s > 0 else float("inf")
        print("  %-30s %5d %8.3f %7.1f%% %12.2f" % (name, a["n"], c, 100.0 * s, cps))

    n = len(test)
    print("held-out tasks: %d (train %d)" % (n, len(train)))
    print("tier accuracy: %d/%d = %.1f%%" % (tier_match, n, 100.0 * tier_match / max(1, n)))
    print("routed choices with trial data: %d/%d" % (routed_with_stats, n))
    if pass_rates:
        print("mean pass rate of routed choice: %.1f%%" % (100.0 * sum(pass_rates) / len(pass_rates)))
    print("routed choices that failed outright: %d/%d" % (unsolved, n))
    if regrets:
        regrets_sorted = sorted(regrets)
        print("cost regret vs cheapest-passing (USD/task, passes only): mean %.3f, median %.3f, total %.2f"
              % (sum(regrets) / len(regrets), regrets_sorted[len(regrets) // 2], sum(regrets)))
    if args.json_out:
        from ._net import write_json

        write_json(
            args.json_out,
            {
                "summary": {"n": n, "tier_match": tier_match},
                "policies": {k: dict(v) for k, v in agg.items()},
                "rows": rows,
            },
        )
        print("wrote:", args.json_out)


def cmd_route(args) -> None:
    cfg = load_config(args.config)
    models = load_models(args.cache_dir)
    quality = build_quality(models, args.cache_dir)
    clf = load_classifier(args.cache_dir)
    store = RunStore(os.path.join(args.cache_dir, "runs"))
    anchors = _load_anchors(args.cache_dir)
    router = Router(cfg, models, clf, store, quality=quality, anchors=anchors)

    task = args.task or ""
    if not task and not sys.stdin.isatty():
        task = sys.stdin.read()

    signals = _parse_signals(args.signal, args.message, cfg)
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
        for ref, s in model_stats(cached["rows"], ref_of=make_ref_of(models)).items():
            if s.get("pass_rate") is not None:
                quality[ref] = s["pass_rate"]
    aa = load_aa_indices(args.cache_dir)

    print("%-38s %9s %9s %8s %s" % ("model", "in$/M", "out$/M", "pass", "AA indices"))
    for ref, _blended, _q in rank_candidates(refs, models, quality):
        m = models.get(ref)
        if not m:
            print("%-38s %9s" % (ref, "(not in catalog)"))
            continue
        q = quality.get(ref)
        idx = aa.get(canon(m.id)) or aa.get(canon(m.name))
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
    _load_dotenv()
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
    p.add_argument("--signal", action="append", default=[], help="name or name=0..1 (repeatable): prior_failure, verification_divergence, not_understood, planning_needs_more_tools, needs_exploration, needs_advisor, user_frustration, tool_error_rate, complete")
    p.add_argument("--message", action="append", default=[], help="recent user message (repeatable); auto-scores user_frustration")
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
