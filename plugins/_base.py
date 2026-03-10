"""
Plugin system for SarabCraft.

Local Python plugins: .py files in the plugins/ folder.
All plugins appear as services in Transfer Verification.
"""

import importlib.util
import json
import os
import io
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

PLUGINS_DIR = Path(__file__).resolve().parent
VARS_FILE = Path(os.path.expanduser("~/.mlsec/variables.json"))
PLUGIN_SETTINGS_FILE = Path(os.path.expanduser("~/.mlsec/plugin_settings.json"))


# ── Plugin Settings (enabled/disabled) ───────────────────────────────────────

def _load_plugin_settings() -> dict:
    if PLUGIN_SETTINGS_FILE.exists():
        try:
            return json.loads(PLUGIN_SETTINGS_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_plugin_settings(settings: dict):
    PLUGIN_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PLUGIN_SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


def set_plugin_enabled(plugin_id: str, enabled: bool) -> dict:
    settings = _load_plugin_settings()
    settings[plugin_id] = {"enabled": enabled}
    _save_plugin_settings(settings)
    return {"ok": True, "plugin_id": plugin_id, "enabled": enabled}


def is_plugin_enabled(plugin_id: str) -> bool:
    settings = _load_plugin_settings()
    entry = settings.get(plugin_id, {})
    return entry.get("enabled", True)


# ── Global Variables ──────────────────────────────────────────────────────────

def _load_vars() -> list:
    """Load global variables. Structure: [{key, value, masked, description}]"""
    VARS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not VARS_FILE.exists():
        VARS_FILE.write_text("[]")
    try:
        data = json.loads(VARS_FILE.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_vars(data: list):
    VARS_FILE.parent.mkdir(parents=True, exist_ok=True)
    VARS_FILE.write_text(json.dumps(data, indent=2))


def get_all_vars(unmask: bool = False) -> list:
    """Get all global variables. Masks values by default."""
    variables = _load_vars()
    if unmask:
        return variables
    return [
        {**v, "value": "••••••••" if v.get("masked") else v["value"]}
        for v in variables
    ]


def get_global_config() -> dict:
    """Get all variables as a flat dict for injection into classify(). Always unmasked."""
    return {v["key"]: v["value"] for v in _load_vars()}


def set_var(key: str, value: str, masked: bool = False, description: str = "") -> dict:
    """Add or update a global variable."""
    variables = _load_vars()
    for v in variables:
        if v["key"] == key:
            v["value"] = value
            v["masked"] = masked
            v["description"] = description
            _save_vars(variables)
            return v
    entry = {"key": key, "value": value, "masked": masked, "description": description}
    variables.append(entry)
    _save_vars(variables)
    return entry


def delete_var(key: str) -> bool:
    """Delete a global variable by key."""
    variables = _load_vars()
    before = len(variables)
    variables = [v for v in variables if v["key"] != key]
    if len(variables) < before:
        _save_vars(variables)
        return True
    return False


# ── Local Plugin Discovery ───────────────────────────────────────────────────

_local_cache: dict = {}


def _discover_local_plugins() -> list:
    """Scan plugins/ folder for .py files with PLUGIN_NAME and classify()."""
    global _local_cache
    results = []
    for f in PLUGINS_DIR.glob("*.py"):
        if f.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(f.stem, str(f))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            name = getattr(mod, "PLUGIN_NAME", None)
            ptype = getattr(mod, "PLUGIN_TYPE", "image")
            classify_fn = getattr(mod, "classify", None)

            if name and callable(classify_fn):
                _local_cache[f.stem] = mod
                pid = f"local_{f.stem}"
                results.append({
                    "id": pid,
                    "name": name,
                    "type": ptype,
                    "source": "local",
                    "file": f.name,
                    "enabled": is_plugin_enabled(pid),
                    "description": getattr(mod, "PLUGIN_DESCRIPTION", ""),
                })
        except Exception as e:
            results.append({
                "id": f"local_{f.stem}",
                "name": f.stem,
                "type": "unknown",
                "source": "local",
                "file": f.name,
                "enabled": False,
                "error": str(e),
            })
    return results


# ── Plugin Execution ─────────────────────────────────────────────────────────

def run_local_plugin(plugin_id: str, adversarial_image=None, original_image=None,
                     adversarial_audio=None, original_audio=None,
                     sample_rate: int = 16000) -> dict:
    """Run a local Python plugin and return predictions."""
    stem = plugin_id.replace("local_", "", 1)
    mod = _local_cache.get(stem)
    if mod is None:
        _discover_local_plugins()
        mod = _local_cache.get(stem)
    if mod is None:
        return {"error": f"Plugin '{stem}' not found", "predictions": []}

    classify_fn = getattr(mod, "classify", None)
    if not callable(classify_fn):
        return {"error": "Plugin has no classify() function", "predictions": []}

    config = get_global_config()

    try:
        t0 = time.time()
        ptype = getattr(mod, "PLUGIN_TYPE", "image")
        use_audio = (ptype == "audio") or (ptype == "both" and adversarial_audio is not None)
        if use_audio:
            result = classify_fn(adversarial_audio, original_audio=original_audio,
                                 sample_rate=sample_rate, config=config)
        else:
            result = classify_fn(adversarial_image, original_image=original_image,
                                 config=config)
        elapsed = (time.time() - t0) * 1000

        if isinstance(result, list):
            return {"predictions": result, "elapsed_ms": elapsed}
        return {"error": "Plugin must return a list of {label, confidence}", "predictions": []}
    except Exception as e:
        return {"error": str(e), "predictions": [], "traceback": traceback.format_exc()}


# ── Unified Listing ──────────────────────────────────────────────────────────

def run_playground(code: str, adversarial_image=None, original_image=None,
                   adversarial_audio=None, original_audio=None,
                   sample_rate: int = 16000) -> dict:
    """Execute plugin code in-memory without saving, for playground testing."""
    validation = _validate_plugin_code(code)
    if not validation["valid"]:
        return {"error": "; ".join(validation["errors"]), "predictions": []}

    try:
        ns = {}
        exec(compile(code, "<playground>", "exec"), ns)
    except Exception as e:
        return {"error": f"Execution error: {e}", "predictions": [],
                "traceback": traceback.format_exc()}

    classify_fn = ns.get("classify")
    if not callable(classify_fn):
        return {"error": "No classify() function found after execution", "predictions": []}

    ptype = ns.get("PLUGIN_TYPE", "image")
    config = get_global_config()

    try:
        t0 = time.time()
        use_audio = (ptype == "audio") or (ptype == "both" and adversarial_audio is not None)
        if use_audio:
            result = classify_fn(adversarial_audio, original_audio=original_audio,
                                 sample_rate=sample_rate, config=config)
        else:
            result = classify_fn(adversarial_image, original_image=original_image,
                                 config=config)
        elapsed = (time.time() - t0) * 1000

        if isinstance(result, list):
            return {"predictions": result, "elapsed_ms": elapsed,
                    "plugin_name": ns.get("PLUGIN_NAME", "Playground"),
                    "plugin_type": ptype}
        return {"error": "classify() must return a list of {label, confidence}",
                "predictions": []}
    except Exception as e:
        return {"error": str(e), "predictions": [],
                "traceback": traceback.format_exc()}


def get_all_plugins() -> list:
    """Return all local plugins for the UI."""
    return _discover_local_plugins()


def get_enabled_plugins(plugin_type: str = "image") -> list:
    """Return only enabled plugins matching the given type."""
    return [
        p for p in get_all_plugins()
        if p.get("enabled", True) and p.get("type") in (plugin_type, "both")
    ]


# ── Local Plugin File Management ─────────────────────────────────────────────

FORBIDDEN_STEMS = {"_base", "__init__", "__pycache__"}


def _safe_stem(name: str) -> str:
    """Sanitize a filename into a safe Python module stem."""
    import re
    stem = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower())
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem or stem in FORBIDDEN_STEMS:
        stem = f"plugin_{uuid.uuid4().hex[:8]}"
    return stem


def _validate_plugin_code(code: str) -> dict:
    """Validate that code defines PLUGIN_NAME and classify(). Returns errors or metadata."""
    errors = []
    if "PLUGIN_NAME" not in code:
        errors.append("Missing PLUGIN_NAME variable")
    if "def classify(" not in code:
        errors.append("Missing classify() function")
    if errors:
        return {"valid": False, "errors": errors}

    try:
        compile(code, "<plugin>", "exec")
    except SyntaxError as e:
        return {"valid": False, "errors": [f"Syntax error on line {e.lineno}: {e.msg}"]}

    return {"valid": True, "errors": []}


def save_plugin_code(filename: str, code: str) -> dict:
    """Save Python code as a local plugin file. Returns the created plugin info."""
    validation = _validate_plugin_code(code)
    if not validation["valid"]:
        return {"error": "; ".join(validation["errors"])}

    stem = _safe_stem(filename.replace(".py", ""))
    target = PLUGINS_DIR / f"{stem}.py"

    if target.exists():
        return {"error": f"File '{stem}.py' already exists — use edit instead"}

    target.write_text(code, encoding="utf-8")
    _local_cache.pop(stem, None)
    return {"ok": True, "file": f"{stem}.py", "id": f"local_{stem}"}


def update_plugin_code(plugin_id: str, code: str) -> dict:
    """Update the code of an existing local plugin."""
    stem = plugin_id.replace("local_", "", 1)
    target = PLUGINS_DIR / f"{stem}.py"
    if not target.exists():
        return {"error": f"File '{stem}.py' not found"}

    validation = _validate_plugin_code(code)
    if not validation["valid"]:
        return {"error": "; ".join(validation["errors"])}

    target.write_text(code, encoding="utf-8")
    _local_cache.pop(stem, None)
    return {"ok": True, "file": f"{stem}.py"}


def update_plugin_type(plugin_id: str, new_type: str) -> dict:
    """Change PLUGIN_TYPE in a local plugin's source code."""
    import re
    if new_type not in ("image", "audio", "both"):
        return {"error": "Type must be 'image', 'audio', or 'both'"}

    stem = plugin_id.replace("local_", "", 1)
    target = PLUGINS_DIR / f"{stem}.py"
    if not target.exists():
        return {"error": f"File '{stem}.py' not found"}

    code = target.read_text(encoding="utf-8")
    pattern = r'(PLUGIN_TYPE\s*=\s*)["\'][a-zA-Z]+["\']'
    if re.search(pattern, code):
        code = re.sub(pattern, f'\\1"{new_type}"', code)
    else:
        code = code.rstrip() + f'\nPLUGIN_TYPE = "{new_type}"\n'

    target.write_text(code, encoding="utf-8")
    _local_cache.pop(stem, None)
    return {"ok": True, "plugin_id": plugin_id, "type": new_type}


def read_plugin_code(plugin_id: str) -> dict:
    """Read the source code of a local plugin."""
    stem = plugin_id.replace("local_", "", 1)
    target = PLUGINS_DIR / f"{stem}.py"
    if not target.exists():
        return {"error": f"File '{stem}.py' not found"}
    return {"code": target.read_text(encoding="utf-8"), "file": f"{stem}.py"}


def delete_local_plugin(plugin_id: str) -> dict:
    """Delete a local plugin .py file."""
    stem = plugin_id.replace("local_", "", 1)
    if stem in FORBIDDEN_STEMS:
        return {"error": "Cannot delete system files"}
    target = PLUGINS_DIR / f"{stem}.py"
    if not target.exists():
        return {"error": f"File '{stem}.py' not found"}
    target.unlink()
    _local_cache.pop(stem, None)
    return {"ok": True}


def upload_plugin_file(filename: str, content: bytes) -> dict:
    """Save an uploaded .py file to the plugins directory."""
    if not filename.endswith(".py"):
        return {"error": "Only .py files are accepted"}

    code = content.decode("utf-8", errors="replace")
    return save_plugin_code(filename, code)


def upload_plugin_zip(content: bytes) -> dict:
    """Extract a .zip archive and install any valid .py plugin files."""
    import zipfile
    results = []
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            py_files = [n for n in zf.namelist()
                        if n.endswith(".py") and "/" not in n and not n.startswith("_")]
            if not py_files:
                nested = [n for n in zf.namelist() if n.endswith(".py") and not os.path.basename(n).startswith("_")]
                py_files = nested

            if not py_files:
                return {"error": "No .py plugin files found in archive", "installed": []}

            for name in py_files:
                code = zf.read(name).decode("utf-8", errors="replace")
                basename = os.path.basename(name)
                res = save_plugin_code(basename, code)
                results.append({"file": basename, **res})
    except zipfile.BadZipFile:
        return {"error": "Invalid zip archive", "installed": []}
    except Exception as e:
        return {"error": str(e), "installed": results}

    return {"installed": results}
