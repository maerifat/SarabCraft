"""
Model loading, caching, and input size detection.
"""

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

from config import device, PRELOADED_MODEL

model_cache = {}

# Models that are 22K-only (no ImageNet-1K head) — warn users to use the fine-tuned variant
_22K_ONLY_VARIANTS = {
    "microsoft/swin-large-patch4-window7-224-22k": "microsoft/swin-large-patch4-window7-224",
    "microsoft/swin-base-patch4-window7-224-22k": "microsoft/swin-base-patch4-window7-224",
    "microsoft/beit-large-patch16-224-pt22k": "microsoft/beit-large-patch16-224-pt22k-ft22k",
}


def load_model(model_name="microsoft/resnet-50", progress=None):
    """Load pretrained model from Hugging Face (downloads on first use)."""
    global model_cache

    def update_progress(value, desc):
        if progress is not None:
            try:
                progress(value, desc=desc)
            except Exception:
                pass

    if model_name in model_cache:
        update_progress(1.0, f"✅ {model_name} (cached)")
        return model_cache[model_name]

    # Check for known 22K-only models and suggest the correct variant
    if model_name in _22K_ONLY_VARIANTS:
        suggestion = _22K_ONLY_VARIANTS[model_name]
        raise ValueError(
            f"{model_name} is a 22K-pretrained model without an ImageNet-1K classification head. "
            f"Use '{suggestion}' instead (the 1K fine-tuned version)."
        )

    is_new = model_name != PRELOADED_MODEL

    if is_new:
        print(f"⬇️ Downloading {model_name}...", flush=True)
        update_progress(0.1, f"⬇️ Downloading {model_name}...")
    else:
        print(f"Loading {model_name} (pre-downloaded)...", flush=True)
        update_progress(0.3, f"Loading {model_name}...")

    # Load processor
    if is_new:
        update_progress(0.2, "📦 Downloading processor...")
    processor = AutoImageProcessor.from_pretrained(model_name)

    # Load model — auto-retry with trust_remote_code if needed
    if is_new:
        update_progress(0.5, "📦 Downloading model weights...")
    try:
        model = AutoModelForImageClassification.from_pretrained(model_name)
    except ValueError as e:
        if "trust_remote_code" in str(e):
            print(f"  Model requires trust_remote_code, retrying...", flush=True)
            model = AutoModelForImageClassification.from_pretrained(
                model_name, trust_remote_code=True
            )
        else:
            raise

    update_progress(0.8, f"🔧 Loading to {device}...")
    model = model.to(device)
    model.eval()

    model_cache[model_name] = (model, processor)

    update_progress(1.0, f"✅ {model_name} ready!")
    print(f"✅ Model {model_name} loaded on {device}", flush=True)

    return model_cache[model_name]


def get_model_input_size(proc):
    """Get the expected input size for a model's processor by testing it."""
    try:
        dummy_img = Image.new('RGB', (512, 512), color='white')
        outputs = proc(images=dummy_img, return_tensors="pt")
        tensor = outputs["pixel_values"]
        h, w = tensor.shape[2], tensor.shape[3]
        print(f"[DEBUG] Detected model input size via test: {w}x{h}", flush=True)
        return h, w
    except Exception as e:
        print(f"[DEBUG] Size detection failed: {e}, using default 224x224", flush=True)
        return 224, 224
