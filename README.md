<div align="center">

# SarabCraft

### Crafting illusions that machines believe.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Black Hat Arsenal](https://img.shields.io/badge/Black%20Hat-Arsenal%202026-red.svg)](https://www.blackhat.com/arsenal.html)

**SarabCraft is a multimodal adversarial AI research framework for crafting believable image and audio attacks, validating transfer across real targets, and turning the results into evidence security teams can act on.**

[Quick Start](#quick-start) · [Attacks](#supported-attacks) · [Verification](#transfer-verification) · [Model Management](#model-management) · [Workflows](#typical-workflows) · [Plugins](#plugin-system)

</div>

---

## What It Does

SarabCraft gives red teams, defenders, and researchers a single platform to demonstrate, measure, and communicate adversarial failure across image and audio systems.

- **Recreate realistic failure cases** across image models, audio models, and speech systems with 32+ image attacks and 8 audio attack types
- **Measure more than a label flip** with perturbation metrics, GradCAM overlays, confidence shifts, side-by-side outputs, and transfer results
- **Prove transfer risk across real targets** including cloud APIs, local models, and custom plugins rather than stopping at a single lab checkpoint
- **Bring custom models and targets into the workflow** from the UI so teams can test the systems they actually care about
- **Scale from one sample to full studies** with batch runs, robustness sweeps, benchmark campaigns, jobs, artifacts, and history
- **Export evidence that travels well** in demos, briefings, write-ups, and disclosures through HTML reports and structured JSON

---

## Why Security Teams Care

- **For red teams** — turn abstract model risk into concrete demonstrations that engineers and leadership cannot dismiss
- **For defenders** — see which attacks transfer, which models break first, and where robustness claims fail under pressure
- **For researchers** — move from isolated proofs of concept to repeatable experiments across local and remote targets
- **For conference and client work** — produce outputs, artifacts, dashboards, and reports that are ready for demos, briefings, and reviews

---

## Quick Start

### Docker

```bash
git clone https://github.com/maerifat/SarabCraft.git
cd SarabCraft
cp .env.example .env
# Edit .env with local credentials for Postgres and MinIO
docker compose up --build -d
```

Open **http://localhost:7860**

For the full product experience, use the Docker flow above.

### Local

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000

# In another terminal, from the repo root:
cd frontend && npm install && npm run dev
```

Frontend at http://localhost:5173, API at http://localhost:8000

Use Docker when you want the full analyst workflow, including long-running jobs, resumable runs, artifact persistence, centralized model management, and the integrated Jobs view.

### Recommended First Steps

1. Open **Settings > Models** to add custom local image, audio, or ASR models and any remote verification targets you want to test.
2. Open **Settings > Credentials** to configure provider profiles for Hugging Face, AWS, Azure, GCP, ElevenLabs, and other supported services.
3. Run **Image Attack** or **Audio Attack** on a single sample, then verify transfer against local or remote targets.
4. Use **Jobs** for long-running attacks, robustness comparisons, and benchmark runs.
5. Review **Dashboard** and **History**, then export HTML or JSON evidence when you are ready to share results.

---

## Supported Attacks

### Image (32+ methods)

| Category | Methods |
|---|---|
| **Gradient (L∞)** | FGSM, I-FGSM, PGD, APGD, MI-FGSM, DI-FGSM, TI-FGSM, NI-FGSM, SI-NI-FGSM, VMI-FGSM, VNI-FGSM, PI-FGSM, Jitter |
| **Optimization** | DeepFool (L2), C&W (L2), FAB, JSMA (L0), EAD (L1+L2), SparseFool |
| **Transfer** | SSA, Admix, BSR, CFM (CVPR 2023), TA-Bench (NeurIPS 2023) |
| **Black-Box** | Square Attack, SPSA, One Pixel, Boundary Attack, HopSkipJump |
| **Physical** | Adversarial Patch, UAP |
| **Research** | SarabCraft R1 — SarabCraft's first in-house transfer-focused image attack with standard and multi-image transfer modes |
| **Ensemble** | AutoAttack (APGD-CE + APGD-DLR + FAB + Square) |

### SarabCraft Research

- **SarabCraft R1** is SarabCraft's first in-house transfer-focused image attack.
- **Standard mode** is built for strong single-model transfer studies.
- **Multi-image transfer mode** unlocks broader bank-building strategies for harder evaluations.
- **Cloud verification stays in the same workflow** through AWS Rekognition, Azure Computer Vision, and Google Cloud Vision integrations.

### Audio Attacks (8 types)

| Attack | What it does |
|---|---|
| **Targeted Transcription** | Forces Whisper to output attacker-chosen text |
| **Hidden Command** | Embeds voice commands in music or noise |
| **Universal Muting** | Prepends universal noise that silences ASR |
| **Psychoacoustic** | Perturbations masked below human hearing threshold |
| **Over-the-Air** | Robust to speaker → air → microphone playback |
| **Speech Jamming** | Denial-of-service — degrades ASR to gibberish |
| **UA3** | Universal perturbation across Whisper + Wav2Vec2 + HuBERT |
| **Audio Classification** | FGSM/PGD/C&W/MI-FGSM/DeepFool on audio classifiers |

---

## Transfer Verification

Test adversarial examples against external services to measure real-world impact beyond a single offline checkpoint:

| Target | Domain | How |
|---|---|---|
| **AWS Rekognition** | Image | `DetectLabels` API |
| **Azure Computer Vision** | Image | `tag_image_in_stream` API |
| **Google Cloud Vision** | Image | `label_detection` API |
| **HuggingFace Inference** | Image | Any HF pipeline (classification, CLIP, detection, segmentation, VQA) |
| **AWS Transcribe** | Audio | Batch transcription via S3 |
| **ElevenLabs STT** | Audio | Scribe v2 API |
| **Local Models (30+)** | Image | Direct inference with exact or resized preprocessing |
| **Custom Plugins** | Both | User-written Python classifiers |

Results include: target match status, confidence drop, original label removal, per-service timing, and WER for audio.

---

## Model Management

SarabCraft includes a centralized model management workflow under **Settings > Models**.

- **Domain-first management** with separate **Image** and **Audio** tabs
- **Custom local model onboarding** for image classifiers, audio classifiers, and ASR models
- **Remote target management** for Hugging Face API targets and supported cloud verification services
- **Workflow-aware enablement** so one local model can participate in classification, attacks, robustness, benchmarks, and local verification where it belongs
- **Operational controls** to test, duplicate, edit, disable, or archive built-in and custom entries
- **Stable experiment history** so long-running jobs and prior results stay reproducible even after later edits

This means you can add a custom checkpoint or remote target once, activate the workflows it should support, and reuse it consistently across the platform.

---

## Supported Models

**Image** — 30+ built-in architectures including ResNet, ConvNeXt, ViT, DeiT, BEiT, Swin, SwinV2, DINOv2, MobileViT, EfficientNet, RegNet, and more, plus custom local image models you register in **Settings > Models**.

**Audio** — built-in audio classifiers such as AST (Speech Commands, AudioSet), Wav2Vec2 (Emotion, Language ID), and HuBERT, plus custom local audio classifiers.

**ASR** — built-in OpenAI Whisper variants (base.en, small.en, base, small), plus custom local ASR models.

**Remote verification** — configurable Hugging Face API targets and supported cloud verification services managed from **Settings > Models**.

---

## Analysis Features

- **Perturbation Metrics** — L0, L1, L2, L-infinity, SSIM, PSNR, and MSE for every run
- **GradCAM** — Attention overlays showing how attacks redirect model focus
- **Batch Mode** — Run the same attack across multiple inputs and get aggregate success rates
- **Robustness Comparison** — Run one attack across many models to see what breaks and what holds
- **Jobs Queue** — Monitor long-running attacks and benchmarks, review event streams, cancel, and resume supported runs
- **Model Management** — Add, test, duplicate, disable, and organize image/audio models and remote verification targets from the UI
- **Dashboard** — Track attack success rates, model vulnerability, and transferability heatmaps over time
- **History** — Search, filter, compare, replay, and export prior experiments
- **Reports** — Export styled HTML and JSON evidence for papers, demos, and client-facing work

---

## Plugin System

Extend transfer verification to internal models, proprietary APIs, or lab-specific tooling with custom Python classifiers:

```python
PLUGIN_NAME = "My Classifier"
PLUGIN_TYPE = "image"  # or "audio" or "both"

def classify(adversarial_image, *, original_image=None, config={}):
    # config contains API keys from Settings > Variables
    token = config.get("MY_API_KEY", "")
    # Call your model/API and return predictions
    return [
        {"label": "cat", "confidence": 0.92},
        {"label": "dog", "confidence": 0.05},
    ]
```

- Built-in CodeMirror editor with Python syntax highlighting
- Live playground for testing before deployment
- Global variable store for secrets (masked in UI)
- Upload `.py` files or `.zip` packages

---

## Credential Configuration

Supports 9 providers with multiple auth methods each:

| Provider | Auth Methods |
|---|---|
| **AWS** | Access Keys, IAM Profile, Assume Role (STS), Environment |
| **Azure** | API Key, Service Principal, Environment |
| **GCP** | JSON Key File, Inline JSON, Application Default Credentials |
| **HuggingFace** | API Token, Environment |
| **ElevenLabs** | API Key, Environment |
| **OpenAI** | API Key, Environment |
| **Anthropic** | API Key, Environment |
| **Replicate** | API Token, Environment |
| **Deepgram** | API Key, Environment |

Configure in the UI at **Settings > Credentials**, or set environment variables directly.

Remote verification targets can use the provider profiles configured in **Settings > Credentials**, keeping target setup and service access aligned.

---

## Typical Workflows

- **Custom model onboarding** — register a new image, audio, or ASR model in **Settings > Models**, choose the workflows it should support, and immediately reuse it across the platform
- **Targeted image evasion** — craft one adversarial image, inspect predictions, verify distortion, and export the result
- **Batch attack study** — run the same attack across multiple inputs and identify success-rate trends
- **Model robustness sweep** — test one attack against many architectures to see which families are fragile
- **Transfer validation** — check whether adversarial examples survive across cloud APIs, local models, and custom plugins
- **Long-running benchmark** — launch broad experiments, monitor them in Jobs, then return to history, dashboard, and exports when they finish

---

## Citation

```bibtex
@software{sarabcraft2026,
  title={SarabCraft: Multimodal Adversarial AI Research Framework},
  author={SarabCraft Research},
  year={2026},
  url={https://github.com/maerifat/SarabCraft}
}
```

---

## License

MIT
