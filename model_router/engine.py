"""Decision engine: classify -> sticky run -> escalate -> advisor -> instruction stack.

This module only decides. It never calls a model and never executes a task.
The caller (a coding agent) consumes a Decision and does the calling.

Semantics:
- A run locks (sticky) to one model at its first decision.
- Escalation signals jump to the next tier's first model; at the frontier they walk
  the remaining frontier candidates, then stay.
- Every decision includes the advisor model (explicit config, or the next tier up).
- The instruction stack accumulates per run: base + current tier directive +
  caller additions + escalation notes. The caller re-sends the whole stack each call,
  like a regular session.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Tuple

from ._net import read_json, write_json
from .catalog import DOMAIN_QUALITY_PRIORITY, quality_from_indices
from .classify import task_domain

TIERS = ("utility", "balanced", "frontier")

# Signals mapped onto engine behavior (failure/divergence escalates, etc.).
FLOOR_SIGNALS = {
    "not_understood": "balanced",
    "planning_needs_more_tools": "balanced",
    "needs_exploration": "frontier",
}
ESCALATING_SIGNALS = ("prior_failure", "verification_divergence")
ADVISOR_SIGNAL = "needs_advisor"

# Default capability anchors per tier (0..1 quality scale), used to turn the
# classifier's tier probabilities into a required-quality number for the task.
DEFAULT_TIER_REQUIREMENT = {"utility": 0.35, "balanced": 0.6, "frontier": 0.8}

# Default reasoning effort per tier: cost-conscious (low/medium), never benchmark-max.
DEFAULT_EFFORTS = {"utility": "low", "balanced": "medium", "frontier": "medium"}

# Canonical effort ordering for mapping abstract effort onto model-specific vocabularies.
EFFORT_RANK = {"none": 0, "minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}


def _pick_effort_value(values, effort: str):
    """Nearest model-understood effort value: exact match, else closest rank (ties round up)."""
    lower = [str(v).lower() for v in values]
    if effort.lower() in lower:
        return values[lower.index(effort.lower())]
    target = EFFORT_RANK.get(effort.lower(), 2)

    def key(v):
        rank = EFFORT_RANK.get(str(v).lower(), 2)
        return (abs(rank - target), 0 if rank >= target else 1)

    return min(values, key=key)

# Optional numeric signal rules. Fires only when the caller passes the signal;
# config decision.signal_rules replaces this map entirely.
DEFAULT_SIGNAL_RULES = {
    "user_frustration": {"threshold": 0.5, "action": "escalate_advise"},
    "tool_error_rate": {"threshold": 0.3, "action": "escalate"},
}


@dataclass
class Decision:
    run_id: str
    turn: int
    tier: str
    model: str
    effort: str
    effort_params: Dict
    advisor: str
    advisor_instruction: str
    advisor_required: bool
    advisor_reason: str
    escalated: bool
    escalation_count: int
    estimated_cost_usd: float
    cost_source: str  # measured | outcomes | estimated | unknown
    latency: Dict
    sticky: bool
    closed: bool
    instructions: List[str]
    signals: Dict[str, bool]
    rationale: Dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunState:
    run_id: str
    ladder_index: int = 0
    turn: int = 0
    escalated: bool = False
    escalation_count: int = 0
    advisor_required: bool = False
    advisor_reason: str = ""
    required_quality: float = 0.0
    domain: str = "general"
    spent_usd: float = 0.0
    budget_usd: Optional[float] = None
    signals_applied: List[str] = field(default_factory=list)
    runtime_instructions: List[str] = field(default_factory=list)
    classifier_source: str = "none"
    classifier_probs: Dict[str, float] = field(default_factory=dict)
    closed: bool = False


class RunStore:
    """Run state store. Pass a directory to persist runs as JSON (CLI); else in-memory.

    Single writer per run_id: writes are atomic (tmp + replace) but there is no locking,
    so two concurrent writers on one run_id will lose updates. Distinct run_ids are
    independent files and are safe to write in parallel.
    """

    def __init__(self, directory: Optional[str] = None):
        self.directory = directory
        self._mem: Dict[str, RunState] = {}

    def _path(self, run_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", run_id)
        return os.path.join(self.directory, safe + ".json")

    def get(self, run_id: str) -> Optional[RunState]:
        if run_id in self._mem:
            return self._mem[run_id]
        if self.directory:
            d = read_json(self._path(run_id))
            if isinstance(d, dict):
                names = {f.name for f in fields(RunState)}
                try:
                    state = RunState(**{k: v for k, v in d.items() if k in names})
                except (TypeError, ValueError):
                    state = None  # corrupt state: start the run fresh rather than crash
                if state is not None:
                    self._mem[run_id] = state
                    return state
        return None

    def put(self, state: RunState) -> None:
        self._mem[state.run_id] = state
        if self.directory:
            write_json(self._path(state.run_id), asdict(state))

    def clear(self, run_id: str) -> None:
        self._mem.pop(run_id, None)
        if self.directory and os.path.exists(self._path(run_id)):
            os.remove(self._path(run_id))


class Router:
    """Sticky-with-escalation routing decision engine."""

    def __init__(
        self,
        config: dict,
        models: Optional[dict] = None,
        classifier=None,
        store: Optional[RunStore] = None,
        quality: Optional[Dict[str, float]] = None,
        anchors: Optional[Dict[str, float]] = None,
        efforts: Optional[Dict[str, str]] = None,
        costs: Optional[Dict[str, float]] = None,
        outcomes: Optional[Dict[str, dict]] = None,
        latency: Optional[Dict[str, dict]] = None,
    ):
        self.config = config
        self.models = models or {}
        self.classifier = classifier
        self.store = store or RunStore()
        self.quality = dict(quality or {})  # ref -> 0..1 capability (AA index, DeepSWE pass)
        self.anchors = dict(anchors or {})  # calibrated tier -> required quality
        self.costs = dict(costs or {})  # measured $/task where available
        self.costs.update(config.get("costs") or {})  # config per-task overrides, if any
        # Observed outcomes, keyed "ref|domain": {"n", "successes", "cost_sum", "cost_n"}.
        # Mutated in place by record_outcome so callers can persist the same dict.
        self.outcomes = outcomes if outcomes is not None else {}
        self.prior_weight = float((config.get("outcomes") or {}).get("prior_weight", 5.0))
        self.latency = dict(latency or {})  # ref -> median performance metrics (AA)
        self.efforts = dict(DEFAULT_EFFORTS)
        self.efforts.update(efforts or {})
        self.efforts.update(config.get("efforts") or {})  # config wins over defaults/injected
        self.ladder: List[Tuple[str, str]] = []
        for tier, spec in config["tiers"].items():
            for ref in spec["models"]:
                self.ladder.append((tier, ref))
        if not self.ladder:
            raise ValueError("config.tiers has no models")
        self._costs = self._normalized_costs()
        self._check_refs()

    def _check_refs(self) -> None:
        if not self.models:
            return
        missing = [ref for _t, ref in self.ladder if ref not in self.models]
        if missing:
            raise ValueError("model refs not found in catalog: %s" % ", ".join(sorted(set(missing))))

    # ------------------------------------------------------------------ helpers

    def _tier_start(self, tier: str) -> int:
        for i, (t, _ref) in enumerate(self.ladder):
            if t == tier:
                return i
        raise ValueError("no models for tier %r" % tier)

    def _tier_start_or_none(self, tier: str) -> Optional[int]:
        for i, (t, _ref) in enumerate(self.ladder):
            if t == tier:
                return i
        return None

    def _resolve_advisor(self, tier: str, model: str) -> str:
        declared = (self.config.get("advisor") or {}).get(tier, "auto")
        if declared != "auto":
            return declared
        idx = TIERS.index(tier) if tier in TIERS else len(TIERS) - 1
        for higher in TIERS[idx + 1 :]:
            start = self._tier_start_or_none(higher)
            if start is not None:
                return self.ladder[start][1]
        for t, ref in self.ladder:  # same tier: alternate candidate if one exists
            if t == tier and ref != model:
                return ref
        return model

    def _escalation_target(self, state: RunState) -> int:
        """Next tier's first model; at the frontier, the next frontier candidate; else stay."""
        tier = self.ladder[state.ladder_index][0]
        idx = TIERS.index(tier) if tier in TIERS else len(TIERS) - 1
        for higher in TIERS[idx + 1 :]:
            start = self._tier_start_or_none(higher)
            if start is not None:
                return start
        for i, (t, _ref) in enumerate(self.ladder):
            if t == tier and i > state.ladder_index:
                return i
        return state.ladder_index

    def _cost_source(self, ref: str) -> str:
        """Where the $/task for a ref came from: config/trials, run outcomes, or unknown."""
        if ref in self.costs:
            return "measured"
        observed, count = self._observed_cost(ref)
        if observed is not None and count > 0:
            return "outcomes"
        return "unknown"

    def _expected_chain_cost(self, index: int) -> float:
        """Expected USD from here to the top of the ladder, 0.0 if unpriced.

        Stops at the first unpriced step rather than guessing it, and uses capability as
        the success proxy, so this is an order-of-magnitude figure for the measured
        ladders, not a quote.
        """
        total = 0.0
        fail = 1.0
        for i in range(index, min(index + 3, len(self.ladder))):
            c = self._task_cost(self.ladder[i][1])
            if c is None:
                break
            total += fail * c
            q = self._quality_of(self.ladder[i][1]) or 0.5
            fail *= 1.0 - max(0.0, min(1.0, q))
            if fail <= 0.01:
                break
        return round(total, 4)

    def _affordable_feasible(self, state: RunState, required: float) -> Optional[str]:
        """Cheapest ladder ref the budget allows and the quality gate accepts, if any."""
        affordable = [
            ref
            for i, (_tier, ref) in enumerate(self.ladder)
            if self._can_afford(state, i)
            and (self._quality_of(ref) is None or float(self._quality_of(ref)) >= required)
        ]
        if not affordable:
            return None
        return min(affordable, key=lambda ref: self._costs.get(ref) if self._costs.get(ref) is not None else 1.0)

    def _can_afford(self, state: RunState, target_index: int) -> bool:
        if state.budget_usd is None:
            return True
        step = self._task_cost(self.ladder[target_index][1])
        if step is None:
            return True
        return (state.spent_usd + step) <= float(state.budget_usd)

    def _move(self, state: RunState, new_index: int, reason: str) -> bool:
        new_index = max(0, min(new_index, len(self.ladder) - 1))
        if new_index == state.ladder_index:
            return False
        if not self._can_afford(state, new_index):
            state.advisor_required = True
            state.advisor_reason = "budget"
            return False
        unpriced = state.budget_usd is not None and self._task_cost(self.ladder[new_index][1]) is None
        frm = self.ladder[state.ladder_index][0]
        to = self.ladder[new_index][0]
        tmpl = (self.config.get("instructions") or {}).get("escalation", "Escalation: {frm} -> {to}.")
        state.runtime_instructions.append(tmpl.format(frm=frm, to=to))
        state.ladder_index = new_index
        state.escalated = True
        state.escalation_count += 1
        state.advisor_required = True  # failure/divergence also routes to advisor review
        # No $/task for the new model means a set budget cannot be enforced from here on;
        # say so rather than implying the cap held.
        state.advisor_reason = "budget_unpriced" if unpriced else (reason or "escalation")
        return True

    def _snapshot(self, state: RunState, signals: Dict[str, bool]) -> Decision:
        tier, model = self.ladder[state.ladder_index]
        instructions = self.config.get("instructions") or {}
        stack = [instructions.get("base", "")]
        stack.append(self.config["tiers"][tier].get("directive", ""))
        stack.extend(state.runtime_instructions)
        stack = [s for s in stack if s]
        effort = self._resolve_effort(tier, model)
        decision = Decision(
            run_id=state.run_id,
            turn=state.turn,
            tier=tier,
            model=model,
            effort=effort,
            effort_params=self._resolve_effort_params(model, effort),
            advisor=self._resolve_advisor(tier, model),
            advisor_instruction=instructions.get("advisor", ""),
            advisor_required=state.advisor_required,
            advisor_reason=state.advisor_reason,
            escalated=state.escalated,
            escalation_count=state.escalation_count,
            estimated_cost_usd=self._expected_chain_cost(state.ladder_index),
            cost_source=self._cost_source(model),
            latency=dict(self.latency.get(model, {})),
            sticky=not state.closed,
            closed=state.closed,
            instructions=stack,
            signals=dict(signals),
            rationale={
                "classifier": state.classifier_source,
                "probs": state.classifier_probs,
                "mode": (self.config.get("decision") or {}).get("mode", "cost"),
                "required_quality": state.required_quality,
                "domain": state.domain,
                "signals_applied": list(state.signals_applied),
            },
        )
        self._log_decision(decision)
        return decision

    def _log_decision(self, d: Decision) -> None:
        """Append one JSON line to <cache>/decisions.jsonl so a consumer can audit what it
        was told and why. Logging must never break a decision, hence the bare except."""
        if not self.store.directory:
            return
        path = os.path.join(os.path.dirname(self.store.directory), "decisions.jsonl")
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": d.run_id,
            "turn": d.turn,
            "tier": d.tier,
            "model": d.model,
            "effort": d.effort,
            "advisor": d.advisor,
            "advisor_required": d.advisor_required,
            "advisor_reason": d.advisor_reason,
            "escalated": d.escalated,
            "estimated_cost_usd": d.estimated_cost_usd,
            "cost_source": d.cost_source,
            "rationale": d.rationale,
            "signals": d.signals,
        }
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _task_cost(self, ref: str) -> Optional[float]:
        """Expected $/task for one ref, or None if we have no per-task price for it.

        Only two sources, both genuinely per-task: measured $/task from trials
        (`costs`), and $/task averaged over this system's own recorded outcomes. Catalog
        token price is per 1M tokens and never converts here: scaling it by any cohort
        token count was tried and is off by 5-50x, because tokens per task varies far
        more than price does. Unmeasured models stay unpriced; supply a per-task number
        via config `costs` if you need one.
        """
        c = self.costs.get(ref)
        if c is not None:
            return float(c)
        observed, count = self._observed_cost(ref)
        if observed is not None and count > 0:
            return float(observed)
        return None

    def _ordering_cost(self, ref: str) -> Optional[float]:
        """Cost used to rank candidates: $/task when known, else catalog token price.

        The price fallback only orders models; it is never used as a dollar figure for
        budgets or cost estimates, so the two unit systems cannot mix.
        """
        c = self._task_cost(ref)
        if c is not None:
            return c
        m = self.models.get(ref)
        return m.blended_cost if m else None

    def _normalized_costs(self) -> Dict[str, Optional[float]]:
        """Candidate cost normalized to 0..1 (cheapest -> 0) for score blending."""
        refs = [ref for _t, ref in self.ladder]
        values = []
        for ref in refs:
            c = self._ordering_cost(ref)
            observed, count = self._observed_cost(ref)
            if c is not None and observed is not None and count > 0:
                w = self.prior_weight
                c = (c * w + observed * count) / (w + count)
            values.append(c)
        known = [v for v in values if v is not None]
        lo, hi = (min(known), max(known)) if known else (0.0, 1.0)
        out: Dict[str, Optional[float]] = {}
        for ref, v in zip(refs, values):
            if v is None:
                out[ref] = None
            elif hi <= lo:
                out[ref] = 0.0
            else:
                out[ref] = (v - lo) / (hi - lo)
        return out

    def _required_quality(self, probs: Dict[str, float]) -> float:
        """Task -> required capability number: sum(tier prob * tier anchor).

        Anchor priority: config decision.tier_requirement > calibrated anchors > defaults.
        """
        explicit = (self.config.get("decision") or {}).get("tier_requirement")
        anchors = explicit if explicit else (self.anchors or DEFAULT_TIER_REQUIREMENT)
        return sum(float(probs.get(t, 0.0)) * float(anchors.get(t, 0.0)) for t in TIERS)

    def _resolve_effort(self, tier: str, model: str) -> str:
        return self.efforts.get(model) or self.efforts.get(tier) or "medium"

    def _resolve_effort_params(self, ref: str, effort: str) -> Dict:
        """Abstract effort -> the model's own parameter vocabulary (from models.dev options)."""
        m = self.models.get(ref)
        options = (m.reasoning_options or []) if m else []
        for opt in options:
            if isinstance(opt, dict) and opt.get("type") == "effort" and opt.get("values"):
                return {"reasoning_effort": _pick_effort_value(list(opt["values"]), effort)}
        for opt in options:
            if isinstance(opt, dict) and opt.get("type") == "budget_tokens":
                budgets = self.config.get("effort_budgets") or {"low": 1024, "medium": 8192, "high": 32768}
                budget = int(budgets.get(effort, budgets.get("medium", 8192)))
                minimum = opt.get("min")
                if isinstance(minimum, (int, float)):
                    budget = max(budget, int(minimum))
                return {"thinking_budget_tokens": budget}
        for opt in options:
            if isinstance(opt, dict) and opt.get("type") == "toggle":
                return {"reasoning": effort.lower() not in ("none", "minimal")}
        return {}

    def _quality_of(self, ref: str, domain: str = "general") -> Optional[float]:
        """Capability 0..1 for a candidate in the task's domain.

        Blends the offline prior (AA/DeepSWE or injected float) with observed run
        outcomes (empirical Bayes, prior_weight pseudo-counts).
        """
        entry = self.quality.get(ref)
        prior = None
        if entry is not None:
            if isinstance(entry, (int, float)) and not isinstance(entry, bool):
                prior = float(entry)
            else:
                prior = quality_from_indices(entry, priority=DOMAIN_QUALITY_PRIORITY.get(domain))
        obs = self.outcomes.get("%s|%s" % (ref, domain))
        if obs and obs.get("n", 0) > 0:
            n = float(obs["n"])
            successes = float(obs.get("successes", 0))
            if prior is None:
                return successes / n
            w = self.prior_weight
            return (successes + w * prior) / (n + w)
        return prior

    def _observed_cost(self, ref: str) -> Tuple[Optional[float], float]:
        """Average observed $/task for a ref across domains, and its sample count."""
        total = count = 0.0
        for key, obs in self.outcomes.items():
            if key.split("|", 1)[0] == ref and obs.get("cost_n"):
                total += float(obs.get("cost_sum", 0.0))
                count += float(obs["cost_n"])
        return ((total / count) if count else None), count

    def _score(self, ref: str, mode: str, domain: str = "general") -> float:
        """Objective score (higher better). Q in 0..1; cost normalized 0=cheap..1=pricey."""
        q = self._quality_of(ref, domain)
        qn = 0.5 if q is None else max(0.0, min(1.0, float(q)))
        c = self._costs.get(ref)
        cn = 0.5 if c is None else float(c)
        if mode == "quality":
            return qn
        if mode == "balanced":
            return (qn + (1.0 - cn)) / 2.0  # arithmetic mean of quality and cost score
        return 1.0 - cn  # cost

    def _select_initial(self, probs: Dict[str, float], domain: str = "general") -> Tuple[str, float]:
        """Pick the start model per objective mode.

        required r = sum(tier prob * tier anchor); feasible set = Q >= r.
        cost -> cheapest feasible; quality -> max Q; balanced -> max mean(Q, 1-C_norm).
        Quality-based modes ignore unmeasured models when any measured candidate exists.
        """
        mode = (self.config.get("decision") or {}).get("mode", "cost")
        required = self._required_quality(probs)
        refs = [ref for _t, ref in self.ladder]
        known = [ref for ref in refs if self._quality_of(ref, domain) is not None]
        if mode in ("quality", "balanced") and known:
            refs = known
        feasible = [
            ref for ref in refs
            if self._quality_of(ref, domain) is None or float(self._quality_of(ref, domain)) >= required
        ]
        pool = feasible or refs
        best = max(pool, key=lambda ref: (self._score(ref, mode, domain), -(self._costs.get(ref) or 0.0)))
        return best, required

    def _ladder_index(self, ref: str) -> int:
        for i, (_t, r) in enumerate(self.ladder):
            if r == ref:
                return i
        raise ValueError("model not in ladder: %r" % ref)

    def _signal_rules(self) -> Dict[str, dict]:
        """Optional numeric signal rules; decision.signal_rules replaces the defaults."""
        rules = (self.config.get("decision") or {}).get("signal_rules")
        if rules is None:
            return DEFAULT_SIGNAL_RULES
        return rules or {}

    def _apply_signal_rules(self, state: RunState, signals: Dict, applied: List[str], escalated: bool) -> None:
        """Numeric/boolean signals -> escalate (at most one tier per decision) or advise.

        Examples: user_frustration >= 0.5 -> escalate + advisor; tool_error_rate >= 0.3
        -> escalate. All optional: nothing fires unless the caller passes the signal.
        """
        for name, rule in self._signal_rules().items():
            if not rule or rule.get("enabled") is False or name not in signals:
                continue
            raw = signals[name]
            value = 1.0 if raw is True else (0.0 if raw is False else float(raw))
            if value < float(rule.get("threshold", 0.5)):
                continue
            action = str(rule.get("action", "escalate"))
            applied.append(name)
            if "escalate" in action:
                if escalated:
                    state.advisor_required = True
                    state.advisor_reason = state.advisor_reason or name
                else:
                    escalated = self._move(state, self._escalation_target(state), name)
                    if not escalated:
                        state.advisor_required = True
                        state.advisor_reason = state.advisor_reason or name
            if "advise" in action:
                state.advisor_required = True
                state.advisor_reason = state.advisor_reason or name

    # ------------------------------------------------------------------- public

    def decide(
        self,
        run_id: str,
        task: str = "",
        signals: Optional[Dict[str, bool]] = None,
        instruction: Optional[str] = None,
        budget_usd: Optional[float] = None,
    ) -> Decision:
        signals = {k: v for k, v in (signals or {}).items() if v not in (None, False)}

        state = self.store.get(run_id)
        if state is not None and state.closed:
            state = None

        if signals.get("complete"):
            if state is None:
                base = (self.config.get("instructions") or {}).get("base", "")
                return Decision(
                    run_id=run_id,
                    turn=0,
                    tier="",
                    model="",
                    effort="",
                    effort_params={},
                    advisor="",
                    advisor_instruction="",
                    advisor_required=False,
                    advisor_reason="",
                    escalated=False,
                    escalation_count=0,
                    estimated_cost_usd=0.0,
                    cost_source="unknown",
                    latency={},
                    sticky=False,
                    closed=True,
                    instructions=[base] if base else [],
                    signals=signals,
                    rationale={"classifier": "none", "probs": {}, "mode": (self.config.get("decision") or {}).get("mode", "cost"), "required_quality": 0.0, "domain": "general", "signals_applied": []},
                )
            state.closed = True
            state.turn += 1
            self.store.put(state)
            return self._snapshot(state, signals)

        if state is None:
            probs: Dict[str, float] = {}
            source = "none"
            if self.classifier is not None and task:
                _tier, probs, source = self.classifier.predict(task)
            domain = task_domain(task) if task else "general"
            model, required = self._select_initial(probs, domain)
            state = RunState(
                run_id=run_id,
                ladder_index=self._ladder_index(model),
                classifier_source=source,
                classifier_probs=probs,
                required_quality=required,
                domain=domain,
            )

        if budget_usd is None:
            budget_usd = (self.config.get("decision") or {}).get("budget_usd")
        if budget_usd is not None:
            state.budget_usd = float(budget_usd)
        # A budget smaller than the chosen model: take the cheapest affordable feasible
        # candidate instead. If nothing fits, keep the pick and say so (advisor decides).
        if state.budget_usd is not None and not self._can_afford(state, state.ladder_index):
            cheaper = self._affordable_feasible(state, state.required_quality)
            if cheaper is not None:
                state.ladder_index = self._ladder_index(cheaper)
            else:
                state.advisor_required = True
                state.advisor_reason = "budget"

        if instruction:
            state.runtime_instructions.append(instruction)

        escalated_this_turn = False
        current_tier = self.ladder[state.ladder_index][0]
        for sig, floor in FLOOR_SIGNALS.items():
            if signals.get(sig) and TIERS.index(floor) > TIERS.index(current_tier):
                if self._move(state, self._tier_start(floor), "floor:" + sig):
                    escalated_this_turn = True
                    current_tier = self.ladder[state.ladder_index][0]

        if any(signals.get(s) for s in ESCALATING_SIGNALS):
            if escalated_this_turn:
                state.advisor_required = True
                state.advisor_reason = state.advisor_reason or "escalation"
            else:
                escalated_this_turn = self._move(state, self._escalation_target(state), "escalation")

        applied: List[str] = []
        self._apply_signal_rules(state, signals, applied, escalated_this_turn)
        if applied:
            state.signals_applied.extend(applied)

        if signals.get(ADVISOR_SIGNAL):
            state.advisor_required = True
            state.advisor_reason = state.advisor_reason or "needs_advisor"

        state.turn += 1
        self.store.put(state)
        return self._snapshot(state, signals)

    def add_instruction(self, run_id: str, text: str) -> Optional[Decision]:
        state = self.store.get(run_id)
        if state is None or state.closed:
            return None
        state.runtime_instructions.append(text)
        self.store.put(state)
        return self._snapshot(state, {})

    def escalate(self, run_id: str, reason: str = "manual") -> Optional[Decision]:
        state = self.store.get(run_id)
        if state is None or state.closed:
            return None
        self._move(state, self._escalation_target(state), reason)
        self.store.put(state)
        return self._snapshot(state, {})

    def close(self, run_id: str) -> None:
        state = self.store.get(run_id)
        if state is not None:
            state.closed = True
            self.store.put(state)

    def record_outcome(self, run_id: str, success: bool, cost_usd: Optional[float] = None) -> Optional[dict]:
        """Feed a finished run's outcome back in; adapts future priors for (ref, domain).

        The outcomes dict is mutated in place - callers persist it (CLI writes
        outcomes.json). Close the run separately when the sticky run ends.
        """
        state = self.store.get(run_id)
        if state is None:
            return None
        ref = self.ladder[state.ladder_index][1]
        key = "%s|%s" % (ref, state.domain)
        obs = self.outcomes.setdefault(key, {"n": 0, "successes": 0, "cost_sum": 0.0, "cost_n": 0})
        obs["n"] = int(obs.get("n", 0)) + 1
        if success:
            obs["successes"] = int(obs.get("successes", 0)) + 1
        if cost_usd is not None:
            obs["cost_sum"] = float(obs.get("cost_sum", 0.0)) + float(cost_usd)
            obs["cost_n"] = int(obs.get("cost_n", 0)) + 1
            state.spent_usd = float(state.spent_usd or 0.0) + float(cost_usd)
            self.store.put(state)
        return {"ref": ref, "domain": state.domain, "n": obs["n"], "successes": obs["successes"]}

    def state(self, run_id: str) -> Optional[RunState]:
        return self.store.get(run_id)
