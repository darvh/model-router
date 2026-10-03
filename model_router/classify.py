"""Task complexity classifier: TF-IDF (words + bigrams) + softmax logistic regression.

Trained on DeepSWE task instructions labeled with the cheapest tier whose model
reliably solved the task (see deepswe.labels). Falls back to a small heuristic when
untrained so routing works cold; `model-router calibrate` replaces it.

ponytail: bag-of-ngrams, not embeddings - a 3-class problem over 113 calibration docs
does not justify a transformer dependency. Upgrade path: optional embedding backend
behind the same predict() signature.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

TIERS = ("utility", "balanced", "frontier")

WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_-]*")
EXT_RE = re.compile(
    r"\b[\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|rb|cs|c|cc|cpp|h|hpp|md|toml|json|ya?ml|sh|sql|html|css)\b",
    re.I,
)
COMPLEX_RE = re.compile(
    r"(refactor|migrat|architect|redesign|concurren|race condition|deadlock|performance|"
    r"optimi[sz]|security|vulnerab|protocol|parser|compil|distribut|transaction|schema|"
    r"asynchronous|async|stream|recursive|index|cache)",
    re.I,
)
ACTION_RE = re.compile(
    r"\b(add|fix|implement|ensure|make|support|update|remove|handle|harden|expose|"
    r"prevent|avoid|allow|enable|generate|attach)\b",
    re.I,
)


def tokenize(text: str) -> List[str]:
    words = WORD_RE.findall(text.lower())
    bigrams = ["%s_%s" % (a, b) for a, b in zip(words, words[1:])]
    return words + bigrams


def dense_features(text: str) -> Dict[str, float]:
    words = WORD_RE.findall(text.lower())
    return {
        "@len": math.log1p(len(text)) / 7.0,
        "@fences": min(1.0, text.count("```") / 6.0),
        "@paths": min(1.0, len(EXT_RE.findall(text)) / 6.0),
        "@backticks": min(1.0, text.count("`") / 24.0),
        "@complex": min(1.0, len(COMPLEX_RE.findall(text)) / 4.0),
        "@actions": min(1.0, len(ACTION_RE.findall(text)) / 5.0),
        "@bullets": min(1.0, len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", text)) / 8.0),
        "@questions": min(1.0, text.count("?") / 3.0),
        "@avgword": min(1.0, (sum(len(w) for w in words) / max(1, len(words))) / 9.0),
    }


def heuristic_probs(text: str) -> Dict[str, float]:
    d = dense_features(text)
    risk = 0.4 * d["@complex"] + 0.3 * d["@paths"] + 0.2 * d["@fences"] + 0.1 * d["@len"]
    if risk >= 0.5:
        return {"utility": 0.12, "balanced": 0.33, "frontier": 0.55}
    if risk <= 0.22:
        return {"utility": 0.55, "balanced": 0.35, "frontier": 0.10}
    return {"utility": 0.20, "balanced": 0.60, "frontier": 0.20}


FRUSTRATION_RE = re.compile(
    r"(still (not|broken|failing|wrong)|not working|doesn'?t work|didn'?t work|"
    r"\bagain\b|why (did|is|does|are)|i said|no[,!]? (that|this)|\bwrong\b|"
    r"\bundo\b|\brevert\b|\bstop\b|\bbroken\b|failed again|fix it|seriously)",
    re.I,
)

DOMAIN_PATTERNS = (
    ("coding", re.compile(
        r"(```|\bdef \b|\bclass \b|\bfunction\b|\brefactor\b|\bapi\b|\bbug\b|\btests?\b|"
        r"\bcompile\b|\brepositor|\bcommit\b|\bmerge\b|typescript|javascript|python|rust|golang|"
        r"\bsql\b|docker|kubernetes|library|module|dependency|exception|stack trace)", re.I)),
    ("math", re.compile(
        r"(theorem|prove|integral|derivative|probability|equation|matrix|algebra|geometry|"
        r"combinatorics|\bmath\b|\bproof\b)", re.I)),
    ("research", re.compile(
        r"(research|literature|\bsources?\b|\bcite\b|citation|survey|state of the art|"
        r"\bpaper\b|compare .{0,24}approaches)", re.I)),
    ("data", re.compile(
        r"(dataset|dataframe|pandas|\bcsv\b|\betl\b|aggregate|statistics|regression|chart|"
        r"dashboard|warehouse|sql query)", re.I)),
    ("writing", re.compile(
        r"(\bwrite\b|\bdraft\b|article|blog|readme|documentation|summarize|rewrite|\btone\b)", re.I)),
)


def task_domain(text: str) -> str:
    """Cheap semantic domain tag: which capability index should judge this task.

    Heuristic on purpose - used to pick the matching AA index for quality Q, not to
    classify task difficulty. Falls back to 'general'.
    """
    scores = {name: len(rx.findall(text or "")) for name, rx in DOMAIN_PATTERNS}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def _caps_score(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 8:
        return 0.0
    return 1.0 if sum(1 for c in letters if c.isupper()) / len(letters) >= 0.6 else 0.0


def _repetition_score(a: str, b: str) -> float:
    wa = {w for w in re.findall(r"[a-z0-9]+", (a or "").lower()) if len(w) >= 3}
    wb = {w for w in re.findall(r"[a-z0-9]+", (b or "").lower()) if len(w) >= 3}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def frustration_score(messages: Sequence[str]) -> float:
    """Heuristic 0..1 user-frustration score from the most recent messages.

    Cheap cues only: frustration phrases, shouting, ALL CAPS, and repetition of the
    previous message (users repeating themselves is the strongest tell). Optional input:
    the caller can forward this as the `user_frustration` signal; callers with richer
    context (rejections, retry counts) should pass their own number instead.
    """
    recent = [m for m in (messages or []) if m][-3:]
    if not recent:
        return 0.0
    score = 0.0
    for text in recent:
        hits = len(FRUSTRATION_RE.findall(text))
        shout = 1.0 if text.count("!") >= 2 or text.count("?") >= 3 else 0.0
        score += min(1.0, 0.4 * hits + 0.3 * shout + 0.3 * _caps_score(text))
    if len(recent) >= 2:
        score += 0.5 * _repetition_score(recent[-1], recent[-2])
    return min(1.0, score / 2.0)


def apply_temperature(probs: Dict[str, float], temperature: float) -> Dict[str, float]:
    """Soften (T>1) or sharpen (T<1) a probability dict. Used for calibration."""
    if temperature <= 0 or abs(temperature - 1.0) < 1e-9:
        return dict(probs)
    powered = {k: max(float(v), 1e-12) ** (1.0 / temperature) for k, v in probs.items()}
    z = sum(powered.values())
    return {k: v / z for k, v in powered.items()}


class Vectorizer:
    def __init__(self, max_features: int = 1200, min_df: int = 2, max_df: float = 0.8):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, texts: Sequence[str]) -> "Vectorizer":
        df = Counter()
        for t in texts:
            for tok in set(tokenize(t)):
                df[tok] += 1
        n = len(texts)
        keep = [(tok, c) for tok, c in df.items() if c >= self.min_df and c <= self.max_df * n]
        keep.sort(key=lambda x: (-x[1], x[0]))
        keep = keep[: self.max_features]
        self.vocab = {tok: i for i, (tok, _) in enumerate(keep)}
        self.idf = {tok: math.log((n + 1) / (c + 1)) + 1.0 for tok, c in keep}
        return self

    def transform(self, text: str) -> Dict[str, float]:
        counts = Counter(tok for tok in tokenize(text) if tok in self.vocab)
        out = {tok: (1.0 + math.log(c)) * self.idf[tok] for tok, c in counts.items()}
        for k, v in dense_features(text).items():
            out[k] = v * 2.0
        return out

    def to_dict(self) -> dict:
        return {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "vocab": self.vocab,
            "idf": self.idf,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Vectorizer":
        v = cls(d["max_features"], d["min_df"], d["max_df"])
        v.vocab = d["vocab"]
        v.idf = d["idf"]
        return v


class SoftmaxRegression:
    def __init__(self, n_classes: int, l2: float = 1e-3, lr: float = 0.3, epochs: int = 100, seed: int = 0):
        self.n_classes = n_classes
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.w: List[Dict[str, float]] = [defaultdict(float) for _ in range(n_classes)]
        self.b: List[float] = [0.0] * n_classes

    def _scores(self, x: Dict[str, float]) -> List[float]:
        return [sum(self.w[c].get(f, 0.0) * v for f, v in x.items()) + self.b[c] for c in range(self.n_classes)]

    @staticmethod
    def _softmax(scores: List[float]) -> List[float]:
        m = max(scores)
        exps = [math.exp(v - m) for v in scores]
        z = sum(exps)
        return [v / z for v in exps]

    def fit(
        self,
        X: Sequence[Dict[str, float]],
        y: Sequence[int],
        class_weight: Optional[Dict[int, float]] = None,
        label_smoothing: float = 0.0,
    ):
        rng = random.Random(self.seed)
        order = list(range(len(X)))
        eps = max(0.0, min(label_smoothing, 0.5))
        for _epoch in range(self.epochs):
            rng.shuffle(order)
            for i in order:
                x, yi = X[i], y[i]
                p = self._softmax(self._scores(x))
                cw = (class_weight or {}).get(yi, 1.0)
                for c in range(self.n_classes):
                    target = (1.0 - eps) if c == yi else (eps / max(1, self.n_classes - 1))
                    g = (p[c] - target) * cw
                    self.b[c] -= self.lr * (g + self.l2 * self.b[c])
                    wc = self.w[c]
                    for f, v in x.items():
                        wc[f] -= self.lr * (g * v + self.l2 * wc[f])
        return self

    def probs(self, x: Dict[str, float]) -> List[float]:
        return self._softmax(self._scores(x))


class Classifier:
    def __init__(self, thresholds: Optional[dict] = None):
        t = {"frontier": 0.45, "utility": 0.55}
        src = dict(thresholds or {})
        if "frontier_prob" in src:
            src["frontier"] = src.pop("frontier_prob")
        if "utility_prob" in src:
            src["utility"] = src.pop("utility_prob")
        t.update(src)
        self.thresholds = t
        self.temperature = 1.0
        self.vec: Optional[Vectorizer] = None
        self.model: Optional[SoftmaxRegression] = None
        self.meta: dict = {}

    def fit(
        self,
        examples: Sequence[Tuple[str, str]],
        epochs: int = 100,
        lr: float = 0.3,
        l2: float = 1e-3,
        label_smoothing: float = 0.0,
    ):
        texts = [t for t, _ in examples]
        ys = [y for _, y in examples]
        classes = [c for c in TIERS if c in set(ys)]
        self.vec = Vectorizer().fit(texts)
        X = [self.vec.transform(t) for t in texts]
        counts = Counter(ys)
        n = len(ys)
        class_weight = {classes.index(c): n / (len(classes) * counts[c]) for c in counts}
        self.model = SoftmaxRegression(len(classes), l2=l2, lr=lr, epochs=epochs).fit(
            X,
            [classes.index(y) for y in ys],
            class_weight,
            label_smoothing=label_smoothing,
        )
        self.meta["classes"] = classes
        self.meta["trained_on"] = n
        return self

    def _probs(self, text: str) -> Tuple[Dict[str, float], str]:
        if self.model is None or self.vec is None:
            return heuristic_probs(text), "heuristic"
        p = self.model.probs(self.vec.transform(text))
        probs = {c: p[i] for i, c in enumerate(self.meta["classes"])}
        return apply_temperature(probs, self.temperature), "trained"

    def predict(self, text: str) -> Tuple[str, Dict[str, float], str]:
        probs, source = self._probs(text)
        if probs.get("frontier", 0.0) >= self.thresholds["frontier"]:
            return "frontier", probs, source
        if probs.get("utility", 0.0) >= self.thresholds["utility"]:
            return "utility", probs, source
        return "balanced", probs, source

    def save(self, path: str) -> None:
        from ._net import write_json

        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str) -> "Classifier":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return {
            "thresholds": self.thresholds,
            "temperature": self.temperature,
            "vectorizer": self.vec.to_dict() if self.vec else None,
            "weights": [dict(w) for w in self.model.w] if self.model else None,
            "bias": self.model.b if self.model else None,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Classifier":
        c = cls(d.get("thresholds"))
        c.temperature = float(d.get("temperature", 1.0))
        if d.get("vectorizer"):
            c.vec = Vectorizer.from_dict(d["vectorizer"])
        if d.get("weights") is not None:
            c.model = SoftmaxRegression(len(d["weights"]))
            c.model.w = [defaultdict(float, w) for w in d["weights"]]
            c.model.b = list(d["bias"])
        c.meta = d.get("meta") or {}
        return c


def kfold_accuracy(
    examples: Sequence[Tuple[str, str]],
    k: int = 5,
    seed: int = 0,
    epochs: int = 100,
    lr: float = 0.3,
    l2: float = 1e-3,
    thresholds: Optional[dict] = None,
    label_smoothing: float = 0.0,
) -> Tuple[float, Counter]:
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    correct = 0
    total = 0
    confusion: Counter = Counter()
    for f in range(k):
        test_idx = set(folds[f])
        train = [examples[i] for i in idx if i not in test_idx]
        if not train:
            continue
        clf = Classifier(thresholds).fit(train, epochs=epochs, lr=lr, l2=l2, label_smoothing=label_smoothing)
        for i in folds[f]:
            text, y = examples[i]
            pred, _probs, _src = clf.predict(text)
            correct += int(pred == y)
            total += 1
            confusion[(y, pred)] += 1
    return (correct / total if total else 0.0), confusion


def calibrate_temperature(
    examples: Sequence[Tuple[str, str]],
    k: int = 5,
    seed: int = 0,
    epochs: int = 100,
    lr: float = 0.3,
    l2: float = 1e-3,
) -> Tuple[float, float]:
    """Fit a probability temperature on out-of-fold predictions (min NLL).

    Returns (temperature, nll). T > 1 means the raw probabilities were overconfident.
    """
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    oof: List[Tuple[Dict[str, float], str]] = []
    for f in range(k):
        test_idx = set(folds[f])
        train = [examples[i] for i in idx if i not in test_idx]
        if not train:
            continue
        clf = Classifier().fit(train, epochs=epochs, lr=lr, l2=l2)
        for i in folds[f]:
            text, y = examples[i]
            probs, _src = clf._probs(text)
            oof.append((probs, y))
    if not oof:
        return 1.0, 0.0
    best_t, best_nll = 1.0, None
    for t in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        nll = 0.0
        for probs, y in oof:
            calibrated = apply_temperature(probs, t)
            nll -= math.log(max(calibrated.get(y, 1e-12), 1e-12))
        nll /= len(oof)
        if best_nll is None or nll < best_nll:
            best_t, best_nll = t, nll
    return best_t, (best_nll or 0.0)
