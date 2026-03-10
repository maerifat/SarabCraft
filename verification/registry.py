"""
Verifier discovery and orchestration.

Automatically discovers all Verifier subclasses and provides
a single run_verification() entry point for the UI.
"""

import time
from typing import Optional
from PIL import Image

from backend.models.registry import (
    BACKEND_HF_API,
    SERVICE_LOCAL_MODELS,
    TASK_IMAGE_CLASSIFICATION,
    resolve_source_model,
)
from verification.base import Verifier, Prediction, VerificationResult


_REGISTRY: list = []
_ALL_LOADED = False


def _public_error_message(exc: Exception, fallback: str) -> str:
    if isinstance(exc, RuntimeError):
        return str(exc)
    return fallback


def register(verifier_class):
    """Decorator that registers a Verifier subclass."""
    if verifier_class not in _REGISTRY:
        _REGISTRY.append(verifier_class)
    return verifier_class


def _ensure_loaded():
    """Import submodules so @register decorators fire."""
    global _ALL_LOADED
    if _ALL_LOADED:
        return
    _ALL_LOADED = True
    import verification.local_models      # noqa: F401
    import verification.huggingface_api    # noqa: F401
    import verification.aws_rekognition   # noqa: F401
    import verification.azure_vision      # noqa: F401
    import verification.gcp_vision        # noqa: F401


def get_all_verifiers() -> list:
    """Return instances of every registered verifier."""
    _ensure_loaded()
    return [cls() for cls in _REGISTRY]


def get_available_verifiers() -> list:
    """Return only verifiers that are currently configured and ready."""
    return [v for v in get_all_verifiers() if v.is_available()]


def _run_single(
    verifier,
    adversarial_image,
    original_image,
    target_label,
    original_label,
    display_name,
    *,
    local_model_value=None,
    remote_model_ref=None,
    exact_preprocess=True,
):
    """Run verification for one service/model and return VerificationResult."""
    if hasattr(verifier, "set_model_names"):
        verifier.set_model_names([local_model_value or display_name])
    if hasattr(verifier, "set_model") and remote_model_ref:
        verifier.set_model(remote_model_ref)
    if hasattr(verifier, "set_labels"):
        verifier.set_labels(target_label=target_label, original_label=original_label)
    if hasattr(verifier, "set_exact_preprocess"):
        verifier.set_exact_preprocess(exact_preprocess)

    t0 = time.time()
    adv_preds = verifier.classify(adversarial_image)
    elapsed = (time.time() - t0) * 1000

    orig_preds = []
    if original_image is not None:
        orig_preds = verifier.classify(original_image)

    matched = False
    orig_gone = False
    conf_drop = 0.0

    if target_label and adv_preds:
        tl = target_label.lower()
        matched = any(tl in p.label.lower() for p in adv_preds[:5])

    if original_label and adv_preds:
        ol = original_label.lower()
        orig_gone = not any(ol in p.label.lower() for p in adv_preds[:3])

    if original_label and orig_preds and adv_preds:
        ol = original_label.lower()
        orig_conf = max(
            (p.confidence for p in orig_preds if ol in p.label.lower()),
            default=0.0,
        )
        adv_conf = max(
            (p.confidence for p in adv_preds if ol in p.label.lower()),
            default=0.0,
        )
        conf_drop = max(0.0, orig_conf - adv_conf)

    return VerificationResult(
        verifier_name=display_name,
        service_type=verifier.service_type,
        predictions=adv_preds,
        original_predictions=orig_preds,
        target_label=target_label,
        original_label=original_label,
        matched_target=matched,
        original_label_gone=orig_gone,
        confidence_drop=conf_drop,
        elapsed_ms=elapsed,
    )


def run_verification(
    adversarial_image: Image.Image,
    original_image: Optional[Image.Image],
    target_label: Optional[str],
    original_label: Optional[str],
    local_model_names: Optional[list] = None,
    remote_targets: Optional[list[dict]] = None,
    local_exact_preprocess: bool = True,
    progress_callback=None,
) -> list:
    """
    Run verification against all selected services.

    Each service/model gets its own VerificationResult showing the effect on that one.
    Local models expand to one result per selected model.
    Remote targets are registry snapshots resolved by the backend routes.
    """
    _ensure_loaded()
    verifiers = get_all_verifiers()
    results = []

    remote_targets = list(remote_targets or [])

    tasks = []
    local_names = list(local_model_names or [])
    for model_value in local_names:
        entry = resolve_source_model(model_value, domain="image", task=TASK_IMAGE_CLASSIFICATION)
        display = (entry or {}).get("display_name") or str(model_value)
        resolved_value = (entry or {}).get("id") or (entry or {}).get("model_ref") or str(model_value)
        tasks.append(
            {
                "display_name": display,
                "svc_key": SERVICE_LOCAL_MODELS,
                "remote_model_ref": None,
                "local_model_value": resolved_value,
            }
        )

    for target in remote_targets:
        service_name = (
            (target.get("settings") or {}).get("service_name")
            or ("HuggingFace API" if target.get("backend") == BACKEND_HF_API else target.get("display_name"))
        )
        tasks.append(
            {
                "display_name": target.get("display_name") or service_name,
                "svc_key": service_name,
                "remote_model_ref": target.get("model_ref") if target.get("backend") == BACKEND_HF_API else None,
                "local_model_value": None,
            }
        )

    total = max(len(tasks), 1)

    for idx, task in enumerate(tasks):
        display_name = task["display_name"]
        svc_key = task["svc_key"]
        remote_model_ref = task["remote_model_ref"]
        if progress_callback:
            progress_callback(idx / total, f"Testing: {display_name}...")

        verifier = next((v for v in verifiers if v.name == svc_key), None)
        if verifier is None:
            results.append(VerificationResult(
                verifier_name=display_name,
                service_type="unknown",
                predictions=[],
                original_predictions=[],
                error=f"Verifier '{svc_key}' not found",
            ))
            continue

        if not verifier.is_available():
            results.append(VerificationResult(
                verifier_name=display_name,
                service_type=verifier.service_type,
                predictions=[],
                original_predictions=[],
                error=verifier.status_message(),
            ))
            continue

        try:
            use_exact = local_exact_preprocess if task["local_model_value"] else False
            r = _run_single(
                verifier, adversarial_image, original_image,
                target_label, original_label, display_name,
                local_model_value=task["local_model_value"],
                remote_model_ref=remote_model_ref,
                exact_preprocess=use_exact,
            )
            results.append(r)
        except Exception as e:
            results.append(VerificationResult(
                verifier_name=display_name,
                service_type=verifier.service_type,
                predictions=[],
                original_predictions=[],
                error=_public_error_message(e, "Verification failed for this target"),
                elapsed_ms=0,
            ))

    if progress_callback:
        progress_callback(1.0, "Verification complete")

    return results
