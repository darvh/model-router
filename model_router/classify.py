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

    def fit(self, X: Sequence[Dict[str, float]], y: Sequence[int], class_weight: Optional[Dict[int, float]] = None):
        rng = random.Random(self.seed)
        order = list(range(len(X)))
        for _epoch in range(self.epochs):
            rng.shuffle(order)
            for i in order:
                x, yi = X[i], y[i]
                p = self._softmax(self._scores(x))
                cw = (class_weight or {}).get(yi, 1.0)
                for c in range(self.n_classes):
                    g = (p[c] - (1.0 if c == yi else 0.0)) * cw
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
        t.update(thresholds or {})
        self.thresholds = t
        self.vec: Optional[Vectorizer] = None
        self.model: Optional[SoftmaxRegression] = None
        self.meta: dict = {}

    def fit(self, examples: Sequence[Tuple[str, str]], epochs: int = 100, lr: float = 0.3, l2: float = 1e-3):
        texts = [t for t, _ in examples]
        ys = [y for _, y in examples]
        classes = [c for c in TIERS if c in set(ys)]
        self.vec = Vectorizer().fit(texts)
        X = [self.vec.transform(t) for t in texts]
        counts = Counter(ys)
        n = len(ys)
        class_weight = {classes.index(c): n / (len(classes) * counts[c]) for c in counts}
        self.model = SoftmaxRegression(len(classes), l2=l2, lr=lr, epochs=epochs).fit(
            X, [classes.index(y) for y in ys], class_weight
        )
        self.meta["classes"] = classes
        self.meta["trained_on"] = n
        return self

    def predict(self, text: str) -> Tuple[str, Dict[str, float], str]:
        if self.model is None or self.vec is None:
            probs = heuristic_probs(text)
            source = "heuristic"
        else:
            p = self.model.probs(self.vec.transform(text))
            probs = {c: p[i] for i, c in enumerate(self.meta["classes"])}
            source = "trained"
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
            "vectorizer": self.vec.to_dict() if self.vec else None,
            "weights": [dict(w) for w in self.model.w] if self.model else None,
            "bias": self.model.b if self.model else None,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Classifier":
        c = cls(d.get("thresholds"))
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
        clf = Classifier(thresholds).fit(train, epochs=epochs, lr=lr, l2=l2)
        for i in folds[f]:
            text, y = examples[i]
            pred, _probs, _src = clf.predict(text)
            correct += int(pred == y)
            total += 1
            confusion[(y, pred)] += 1
    return (correct / total if total else 0.0), confusion
