"""Static Hugging Face preset metadata used across registry and verification."""

HF_DEFAULT_MODEL = "google/vit-base-patch16-224"

HF_PRESET_MODELS = [
    ("google/vit-base-patch16-224", "ViT-Base (Google)"),
    ("google/vit-large-patch16-224", "ViT-Large (Google)"),
    ("microsoft/resnet-50", "ResNet-50 (Microsoft)"),
    ("facebook/deit-base-distilled-patch16-224", "DeiT-Base (Facebook)"),
    ("facebook/deit-small-patch16-224", "DeiT-Small (Facebook)"),
    ("nateraw/vit-age-classifier", "Age Classifier (ViT)"),
    ("openai/clip-vit-base-patch32", "[zero-shot] CLIP ViT-B/32 (OpenAI)"),
    ("openai/clip-vit-large-patch14", "[zero-shot] CLIP ViT-L/14 (OpenAI)"),
    ("google/siglip-base-patch16-224", "[zero-shot] SigLIP-Base (Google)"),
    ("facebook/detr-resnet-50", "[detection] DETR ResNet-50"),
    ("nvidia/segformer-b0-finetuned-ade-512-512", "[segment] SegFormer-B0 (NVIDIA)"),
]

HF_IMAGE_CLASSIFICATION_MODELS = HF_PRESET_MODELS
