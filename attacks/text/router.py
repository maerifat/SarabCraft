"""
Text attack router: dispatches to the correct attack function.

Mirrors attacks/image/router.py — handles model wrapping, cancellation,
constraint setup, and dispatch via TEXT_ATTACK_DISPATCH table.
"""

import logging

from attacks.text.types import AttackCancelledError

from attacks.text.deepwordbug import run_deepwordbug
from attacks.text.textbugger import run_textbugger
from attacks.text.hotflip import run_hotflip
from attacks.text.textfooler import run_textfooler
from attacks.text.bert_attack import run_bert_attack
from attacks.text.bae import run_bae
from attacks.text.pwws import run_pwws
from attacks.text.alzantot_ga import run_alzantot_ga
from attacks.text.faster_alzantot_ga import run_faster_alzantot_ga
from attacks.text.iga import run_iga
from attacks.text.pso import run_pso
from attacks.text.clare import run_clare
from attacks.text.back_translation import run_back_translation
from attacks.text.pruthi2019 import run_pruthi2019

logger = logging.getLogger("textattack.router")


# ── Model wrappers ───────────────────────────────────────────────────────────

class _TextModelWrapper:
    """Wraps HF model to provide clean predict() interface + query counting.

    Mirrors _PixelModelWrapper in attacks/image/router.py — normalises the model
    interface so attack functions don't touch HF internals directly.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.query_count = 0

    def predict(self, text: str) -> tuple[str, float, int]:
        """Returns (label, confidence, label_index)."""
        self.query_count += 1
        from models.text_loader import get_label_and_confidence
        return get_label_and_confidence(self.model, self.tokenizer, text)

    def predict_probs(self, text: str):
        """Returns full probability vector as a list."""
        self.query_count += 1
        import torch
        from models.text_loader import device

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        return probs.cpu().tolist()

    def predict_probs_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Batched probability prediction — tokenizes and runs forward pass
        in chunks of ``batch_size`` for much higher throughput."""
        import torch
        from models.text_loader import device

        all_probs: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            self.query_count += len(chunk)
            inputs = self.tokenizer(
                chunk, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            all_probs.extend(probs.cpu().tolist())
        return all_probs

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float, int]]:
        """Batch prediction. Each query counts."""
        return [self.predict(t) for t in texts]


class _CancelableTextModelWrapper:
    """Checks cancellation on each predict() call.

    Mirrors _CancelableModelWrapper in attacks/image/router.py.
    """

    def __init__(self, wrapped: _TextModelWrapper, should_cancel):
        self.wrapped = wrapped
        self.should_cancel = should_cancel

    @property
    def model(self):
        return self.wrapped.model

    @property
    def tokenizer(self):
        return self.wrapped.tokenizer

    @property
    def query_count(self):
        return self.wrapped.query_count

    def predict(self, text: str) -> tuple[str, float, int]:
        if self.should_cancel and self.should_cancel():
            raise AttackCancelledError("Attack cancelled")
        return self.wrapped.predict(text)

    def predict_probs(self, text: str):
        if self.should_cancel and self.should_cancel():
            raise AttackCancelledError("Attack cancelled")
        return self.wrapped.predict_probs(text)

    def predict_probs_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if self.should_cancel and self.should_cancel():
            raise AttackCancelledError("Attack cancelled")
        return self.wrapped.predict_probs_batch(texts, batch_size)

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float, int]]:
        if self.should_cancel and self.should_cancel():
            raise AttackCancelledError("Attack cancelled")
        return self.wrapped.predict_batch(texts)


def _wrap_cancelable(wrapper, should_cancel):
    if not should_cancel:
        return wrapper
    return _CancelableTextModelWrapper(wrapper, should_cancel)


# ── Dispatch registry ────────────────────────────────────────────────────────
# Each entry: attack_name → lambda(model_wrapper, tokenizer, text, target_label, params)
# All return: adversarial text (str) — raw output, like image attacks return tensor.

def _p(params, key, default, cast=float):
    return cast(params.get(key, default))


TEXT_ATTACK_DISPATCH = {
    "DeepWordBug": lambda w, tok, txt, tgt, p:
        run_deepwordbug(w, tok, txt, tgt,
            _p(p, "max_perturbations", 5, int),
            str(p.get("scoring_method", "combined")),
            str(p.get("transformer", "homoglyph"))),

    "TextBugger": lambda w, tok, txt, tgt, p:
        run_textbugger(w, tok, txt, tgt,
            _p(p, "max_perturbations", 5, int),
            str(p.get("mode", "black-box")),
            str(p.get("strategy", "combined")),
            _p(p, "similarity_threshold", 0.8),
            _p(p, "max_queries", 5000, int),
            seed=p.get("seed", None)),

    "HotFlip": lambda w, tok, txt, tgt, p:
        run_hotflip(w, tok, txt, tgt,
            _p(p, "max_flips", 5, int),
            _p(p, "beam_width", 10, int),
            _p(p, "max_perturbed", 2, int),
            _p(p, "similarity_threshold", 0.8)),

    "Pruthi2019": lambda w, tok, txt, tgt, p:
        run_pruthi2019(w, tok, txt, tgt,
            _p(p, "max_perturbations", 1, int)),

    "TextFooler": lambda w, tok, txt, tgt, p:
        run_textfooler(w, tok, txt, tgt,
            _p(p, "max_candidates", 50, int),
            _p(p, "similarity_threshold", 0.840845057),
            _p(p, "max_perturbation_ratio", 0.3),
            _p(p, "embedding_cos_threshold", 0.5)),

    "BERT-Attack": lambda w, tok, txt, tgt, p:
        run_bert_attack(w, tok, txt, tgt,
            _p(p, "max_candidates", 48, int),
            _p(p, "max_perturbation_ratio", 0.4),
            _p(p, "threshold_pred_score", 0.0),
            bool(p.get("use_bpe", True))),

    "BAE": lambda w, tok, txt, tgt, p:
        run_bae(w, tok, txt, tgt,
            str(p.get("strategy", "R")),
            _p(p, "max_candidates", 50, int),
            _p(p, "similarity_threshold", 0.936338023),
            _p(p, "max_perturbation_ratio", 0.5)),

    "PWWS": lambda w, tok, txt, tgt, p:
        run_pwws(w, tok, txt, tgt,
            _p(p, "max_candidates", 50, int),
            bool(p.get("use_named_entities", False))),

    "Alzantot GA": lambda w, tok, txt, tgt, p:
        run_alzantot_ga(w, tok, txt, tgt,
            _p(p, "population_size", 60, int),
            _p(p, "max_generations", 20, int),
            _p(p, "mutation_rate", 1.0),
            _p(p, "similarity_threshold", 0.8),
            _p(p, "require_embeddings", True, bool)),

    "Faster Alzantot GA": lambda w, tok, txt, tgt, p:
        run_faster_alzantot_ga(w, tok, txt, tgt,
            _p(p, "population_size", 60, int),
            _p(p, "max_generations", 40, int),
            _p(p, "mutation_rate", 1.0),
            _p(p, "similarity_threshold", 0.8),
            _p(p, "require_embeddings", True, bool)),

    "IGA": lambda w, tok, txt, tgt, p:
        run_iga(w, tok, txt, tgt,
            _p(p, "population_size", 60, int),
            _p(p, "max_generations", 20, int),
            _p(p, "max_replace_times_per_index", 5, int),
            _p(p, "max_perturbation_ratio", 0.2)),

    "PSO": lambda w, tok, txt, tgt, p:
        run_pso(w, tok, txt, tgt,
            _p(p, "pop_size", 60, int),
            _p(p, "max_iters", 20, int),
            _p(p, "max_perturbation_ratio", 0.2),
            _p(p, "max_queries", 5000, int),
            seed=p.get("seed", None)),

    "Clare": lambda w, tok, txt, tgt, p:
        run_clare(w, tok, txt, tgt,
            _p(p, "max_perturbations", 5, int),
            _p(p, "similarity_threshold", 0.7)),

    "Back-Translation": lambda w, tok, txt, tgt, p:
        run_back_translation(w, tok, txt, tgt,
            _p(p, "num_paraphrases", 5, int),
            _p(p, "similarity_threshold", 0.6),
            _p(p, "chained_back_translation", 0, int),
            str(p.get("target_lang", "es"))),
}


# ── Main entry point ─────────────────────────────────────────────────────────

def run_text_attack(
    attack_name: str,
    model,
    tokenizer,
    text: str,
    *,
    target_label: str = None,
    params: dict = None,
    should_cancel=None,
) -> tuple[str, int, str | None]:
    """Route to appropriate text attack function.

    Mirrors run_attack_method() in attacks/image/router.py:
      1. Wraps model (_TextModelWrapper + cancellation)
      2. Resolves target_label to model's actual label names
      3. Dispatches to attack function
      4. Returns adversarial text + query count

    Args:
        attack_name: key in TEXT_ATTACK_DISPATCH
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        text: input text to attack
        target_label: target class name (None = untargeted)
        params: attack-specific parameters
        should_cancel: callable returning bool for cooperative cancellation

    Returns:
        Tuple of (adversarial_text, query_count, resolved_target_label).
    """
    if params is None:
        params = {}

    # Resolve user-friendly target label (e.g. 'POSITIVE') to model label (e.g. 'LABEL_1')
    if target_label is not None:
        from models.text_loader import resolve_target_label
        resolved = resolve_target_label(model, target_label)
        if resolved is not None:
            logger.info("Resolved target label '%s' → '%s'", target_label, resolved)
            target_label = resolved

    # Wrap model (like _build_pixel_model)
    base_wrapper = _TextModelWrapper(model, tokenizer)
    wrapped = _wrap_cancelable(base_wrapper, should_cancel)

    dispatch_fn = TEXT_ATTACK_DISPATCH.get(attack_name)
    if dispatch_fn is None:
        available = ", ".join(sorted(TEXT_ATTACK_DISPATCH.keys()))
        raise ValueError(f"Unknown text attack: '{attack_name}'. Available: {available}")

    logger.info("Running text attack: %s (target=%s)", attack_name, target_label)
    adversarial_text = dispatch_fn(wrapped, tokenizer, text, target_label, params)
    return adversarial_text, base_wrapper.query_count, target_label

