"""model_router - sticky-with-escalation routing decision engine.

Decides only: tier + model + advisor + instruction stack for a run.
Never calls a model, never executes a task.
"""
from .catalog import Model, load_models, tier_of_ref
from .classify import Classifier
from .engine import Decision, Router, RunStore

__version__ = "0.1.0"
__all__ = ["Router", "Decision", "RunStore", "Classifier", "Model", "load_models", "tier_of_ref"]
