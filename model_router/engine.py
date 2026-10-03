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

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from ._net import read_json, write_json

TIERS = ("utility", "balanced", "frontier")

# Signals mapped onto engine behavior (failure/divergence escalates, etc.).
FLOOR_SIGNALS = {
    "not_understood": "balanced",
    "planning_needs_more_tools": "balanced",
    "needs_exploration": "frontier",
}
ESCALATING_SIGNALS = ("prior_failure", "verification_divergence")
ADVISOR_SIGNAL = "needs_advisor"

# Objective presets over the expected-cost rule. Explicit decision.* keys win.
MODE_DEFAULTS = {
    "cost": {"confidence_cap": 0.9},                          # cheap starts, escalate on failure
    "balanced": {"confidence_cap": 0.98},                     # trust the classifier when it is sure
    "quality": {"confidence_cap": 0.98, "min_tier": "balanced"},  # never start below balanced
}


@dataclass
class Decision:
    run_id: str
    turn: int
    tier: str
    model: str
    advisor: str
    advisor_instruction: str
    advisor_required: bool
    advisor_reason: str
    escalated: bool
    escalation_count: int
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
    runtime_instructions: List[str] = field(default_factory=list)
    classifier_source: str = "none"
    classifier_probs: Dict[str, float] = field(default_factory=dict)
    closed: bool = False


class RunStore:
    """Run state store. Pass a directory to persist runs as JSON (CLI); else in-memory."""

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
            if d:
                state = RunState(**d)
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

    def __init__(self, config: dict, models: Optional[dict] = None, classifier=None, store: Optional[RunStore] = None):
        self.config = config
        self.models = models or {}
        self.classifier = classifier
        self.store = store or RunStore()
        self.ladder: List[Tuple[str, str]] = []
        for tier, spec in config["tiers"].items():
            for ref in spec["models"]:
                self.ladder.append((tier, ref))
        if not self.ladder:
            raise ValueError("config.tiers has no models")
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

    def _move(self, state: RunState, new_index: int, reason: str) -> bool:
        new_index = max(0, min(new_index, len(self.ladder) - 1))
        if new_index == state.ladder_index:
            return False
        frm = self.ladder[state.ladder_index][0]
        to = self.ladder[new_index][0]
        tmpl = (self.config.get("instructions") or {}).get("escalation", "Escalation: {frm} -> {to}.")
        state.runtime_instructions.append(tmpl.format(frm=frm, to=to))
        state.ladder_index = new_index
        state.escalated = True
        state.escalation_count += 1
        state.advisor_required = True  # failure/divergence also routes to advisor review
        state.advisor_reason = "escalation"
        return True

    def _snapshot(self, state: RunState, signals: Dict[str, bool]) -> Decision:
        tier, model = self.ladder[state.ladder_index]
        instructions = self.config.get("instructions") or {}
        stack = [instructions.get("base", "")]
        stack.append(self.config["tiers"][tier].get("directive", ""))
        stack.extend(state.runtime_instructions)
        stack = [s for s in stack if s]
        return Decision(
            run_id=state.run_id,
            turn=state.turn,
            tier=tier,
            model=model,
            advisor=self._resolve_advisor(tier, model),
            advisor_instruction=instructions.get("advisor", ""),
            advisor_required=state.advisor_required,
            advisor_reason=state.advisor_reason,
            escalated=state.escalated,
            escalation_count=state.escalation_count,
            sticky=not state.closed,
            closed=state.closed,
            instructions=stack,
            signals=dict(signals),
            rationale={"classifier": state.classifier_source, "probs": state.classifier_probs},
        )

    def _decision_cfg(self) -> dict:
        d = dict(self.config.get("decision") or {})
        preset = MODE_DEFAULTS.get(d.get("mode", "cost"), {})
        for k, v in preset.items():
            d.setdefault(k, v)
        return d

    def _expected_cost_tier(self, probs: Dict[str, float]) -> Optional[str]:
        """Cheapest expected start: E[cost | start t] with escalation only on failure.

        E[utility] = cu + (pb+pf)*cb + pf*cf
        E[balanced] = cb + pf*cf
        E[frontier] = cf
        """
        cfg = self._decision_cfg()
        costs = cfg.get("tier_costs") or {}
        c = [float(costs.get(t, 0.0)) for t in TIERS]
        if not any(c):
            return None
        cap = float(cfg.get("confidence_cap", 0.9))
        p = [min(float(probs.get(t, 0.0)), cap) for t in TIERS]
        expected = [
            c[0] + (p[1] + p[2]) * c[1] + p[2] * c[2],
            c[1] + p[2] * c[2],
            c[2],
        ]
        return TIERS[expected.index(min(expected))]

    def _initial_tier(self, task: str) -> Tuple[str, str, Dict[str, float]]:
        if self.classifier is not None and task:
            tier, probs, source = self.classifier.predict(task)
            cfg = self._decision_cfg()
            if cfg.get("rule", "thresholds") == "expected_cost":
                override = self._expected_cost_tier(probs)
                if override is not None:
                    tier = override
            floor = cfg.get("min_tier")
            if floor and TIERS.index(tier) < TIERS.index(floor):
                tier = floor
            return tier, source, probs
        default = "balanced" if self._tier_start_or_none("balanced") is not None else self.ladder[0][0]
        return default, "none", {}

    # ------------------------------------------------------------------- public

    def decide(
        self,
        run_id: str,
        task: str = "",
        signals: Optional[Dict[str, bool]] = None,
        instruction: Optional[str] = None,
    ) -> Decision:
        signals = {k: bool(v) for k, v in (signals or {}).items() if v}

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
                    advisor="",
                    advisor_instruction="",
                    advisor_required=False,
                    advisor_reason="",
                    escalated=False,
                    escalation_count=0,
                    sticky=False,
                    closed=True,
                    instructions=[base] if base else [],
                    signals=signals,
                    rationale={"classifier": "none", "probs": {}},
                )
            state.closed = True
            state.turn += 1
            self.store.put(state)
            return self._snapshot(state, signals)

        if state is None:
            tier, source, probs = self._initial_tier(task)
            state = RunState(
                run_id=run_id,
                ladder_index=self._tier_start(tier),
                classifier_source=source,
                classifier_probs=probs,
            )

        if instruction:
            state.runtime_instructions.append(instruction)

        current_tier = self.ladder[state.ladder_index][0]
        for sig, floor in FLOOR_SIGNALS.items():
            if signals.get(sig) and TIERS.index(floor) > TIERS.index(current_tier):
                if self._move(state, self._tier_start(floor), "floor:" + sig):
                    current_tier = self.ladder[state.ladder_index][0]

        if any(signals.get(s) for s in ESCALATING_SIGNALS):
            self._move(state, self._escalation_target(state), "signal")

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

    def state(self, run_id: str) -> Optional[RunState]:
        return self.store.get(run_id)
