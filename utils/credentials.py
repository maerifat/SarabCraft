"""
Load and save API credentials for external services.

Credentials are stored in a local state directory and applied to
os.environ so verification backends can use them. Never commit the
credential state to version control.
"""

import configparser
import json
import logging
import os
from pathlib import Path

_creds_logger = logging.getLogger(__name__)


def _state_dir(create: bool = False) -> Path:
    path = Path(os.getenv("SARABCRAFT_STATE_DIR") or (Path(__file__).resolve().parent.parent / ".sarabcraft"))
    if create:
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass
    return path


def _creds_file() -> Path:
    return _state_dir() / "credentials.json"

# Legacy env-var map (kept for backward compat)
SERVICE_ENV_MAP = {
    "HuggingFace API": ["HF_API_TOKEN"],
    "AWS Rekognition": [
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        "AWS_DEFAULT_REGION",
    ],
    "Azure Computer Vision": ["AZURE_VISION_ENDPOINT", "AZURE_VISION_KEY"],
    "Google Cloud Vision": ["GOOGLE_APPLICATION_CREDENTIALS"],
    "ElevenLabs STT": ["ELEVENLABS_API_KEY"],
}


def load_credentials() -> dict:
    """Load credentials from file. Returns dict of env_var -> value."""
    creds_file = _creds_file()
    if not creds_file.exists():
        return {}
    try:
        with open(creds_file, "r") as f:
            data = json.load(f)
        return {k: (v or "") for k, v in data.items() if isinstance(v, str)}
    except Exception:
        return {}


def save_credentials(creds: dict) -> None:
    """Save credentials to file and apply to os.environ."""
    allowed = {
        "HF_API_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN", "AWS_DEFAULT_REGION",
        "AZURE_VISION_ENDPOINT", "AZURE_VISION_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "ELEVENLABS_API_KEY",
    }
    data = {k: (v or "") for k, v in creds.items() if k in allowed}
    creds_file = _state_dir(create=True) / "credentials.json"
    with open(creds_file, "w") as f:
        json.dump(data, f, indent=2)
    os.chmod(creds_file, 0o600)
    for k, v in data.items():
        if v:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]


def apply_credentials() -> None:
    """Apply stored credentials to os.environ (call at app startup)."""
    creds = load_credentials()
    for k, v in creds.items():
        if v:
            os.environ[k] = v

    mp = _load_multiprofile()
    for prov, data in mp.items():
        for prof in data.get("profiles", []):
            if prof.get("active"):
                _apply_profile(prov, prof)
    _apply_environment_overrides()


# ── Multi-profile credential management ────────────────────────────────────


def _multiprofile_file() -> Path:
    return _state_dir() / "profiles.json"

PROVIDER_SCHEMA = {
    "aws": {
        "label": "AWS",
        "auth_methods": [
            {
                "id": "access_key",
                "label": "Access Key",
                "description": "IAM user access key + secret",
                "fields": {
                    "AWS_ACCESS_KEY_ID": {"label": "Access Key ID", "secret": False, "required": True},
                    "AWS_SECRET_ACCESS_KEY": {"label": "Secret Access Key", "secret": True, "required": True},
                    "AWS_SESSION_TOKEN": {"label": "Session Token", "secret": True, "required": False},
                    "AWS_DEFAULT_REGION": {"label": "Region", "secret": False, "required": True, "default": "us-east-1"},
                },
            },
            {
                "id": "profile",
                "label": "AWS Profile",
                "description": "Import from ~/.aws/credentials",
                "fields": {
                    "AWS_PROFILE_NAME": {"label": "Profile Name", "secret": False, "required": True, "default": "default"},
                    "AWS_DEFAULT_REGION": {"label": "Region", "secret": False, "required": True, "default": "us-east-1"},
                },
            },
            {
                "id": "assume_role",
                "label": "Assume Role",
                "description": "Cross-account role assumption",
                "fields": {
                    "AWS_ACCESS_KEY_ID": {"label": "Access Key ID", "secret": False, "required": True},
                    "AWS_SECRET_ACCESS_KEY": {"label": "Secret Access Key", "secret": True, "required": True},
                    "AWS_ROLE_ARN": {"label": "Role ARN", "secret": False, "required": True},
                    "AWS_ROLE_SESSION_NAME": {"label": "Session Name", "secret": False, "required": False, "default": "mlsec-session"},
                    "AWS_DEFAULT_REGION": {"label": "Region", "secret": False, "required": True, "default": "us-east-1"},
                },
            },
            {
                "id": "env",
                "label": "Environment",
                "description": "Auto-detect from environment variables",
                "fields": {},
            },
        ],
    },
    "azure": {
        "label": "Azure",
        "auth_methods": [
            {
                "id": "api_key",
                "label": "API Key",
                "description": "Endpoint URL + API key",
                "fields": {
                    "AZURE_VISION_ENDPOINT": {"label": "Endpoint URL", "secret": False, "required": True},
                    "AZURE_VISION_KEY": {"label": "API Key", "secret": True, "required": True},
                },
            },
            {
                "id": "service_principal",
                "label": "Service Principal",
                "description": "Azure AD / Entra ID app credentials",
                "fields": {
                    "AZURE_VISION_ENDPOINT": {"label": "Endpoint URL", "secret": False, "required": True},
                    "AZURE_TENANT_ID": {"label": "Tenant ID", "secret": False, "required": True},
                    "AZURE_CLIENT_ID": {"label": "Client ID", "secret": False, "required": True},
                    "AZURE_CLIENT_SECRET": {"label": "Client Secret", "secret": True, "required": True},
                },
            },
            {
                "id": "env",
                "label": "Environment",
                "description": "Auto-detect from environment variables",
                "fields": {},
            },
        ],
    },
    "gcp": {
        "label": "Google Cloud",
        "auth_methods": [
            {
                "id": "json_file",
                "label": "Key File Path",
                "description": "Path to service account JSON on server",
                "fields": {
                    "GOOGLE_APPLICATION_CREDENTIALS": {"label": "Path to JSON Key File", "secret": False, "required": True},
                },
            },
            {
                "id": "json_inline",
                "label": "Paste JSON",
                "description": "Paste service account JSON content directly",
                "fields": {
                    "GCP_SERVICE_ACCOUNT_JSON": {"label": "Service Account JSON", "secret": True, "required": True, "multiline": True},
                },
            },
            {
                "id": "adc",
                "label": "Default Credentials",
                "description": "Use Application Default Credentials (gcloud auth)",
                "fields": {},
            },
        ],
    },
    "huggingface": {
        "label": "HuggingFace",
        "auth_methods": [
            {
                "id": "token",
                "label": "API Token",
                "description": "Personal access token from huggingface.co/settings/tokens",
                "fields": {
                    "HF_API_TOKEN": {"label": "API Token", "secret": True, "required": True},
                },
            },
            {
                "id": "env",
                "label": "Environment",
                "description": "Auto-detect HF_API_TOKEN from environment",
                "fields": {},
            },
        ],
    },
    "elevenlabs": {
        "label": "ElevenLabs",
        "auth_methods": [
            {
                "id": "api_key",
                "label": "API Key",
                "description": "API key from elevenlabs.io",
                "fields": {
                    "ELEVENLABS_API_KEY": {"label": "API Key", "secret": True, "required": True},
                },
            },
        ],
    },
    "openai": {
        "label": "OpenAI",
        "auth_methods": [
            {
                "id": "api_key",
                "label": "API Key",
                "description": "API key from platform.openai.com",
                "fields": {
                    "OPENAI_API_KEY": {"label": "API Key", "secret": True, "required": True},
                    "OPENAI_ORG_ID": {"label": "Organization ID", "secret": False, "required": False},
                },
            },
            {
                "id": "env",
                "label": "Environment",
                "description": "Auto-detect OPENAI_API_KEY from environment",
                "fields": {},
            },
        ],
    },
    "anthropic": {
        "label": "Anthropic",
        "auth_methods": [
            {
                "id": "api_key",
                "label": "API Key",
                "description": "API key from console.anthropic.com",
                "fields": {
                    "ANTHROPIC_API_KEY": {"label": "API Key", "secret": True, "required": True},
                },
            },
        ],
    },
    "replicate": {
        "label": "Replicate",
        "auth_methods": [
            {
                "id": "api_key",
                "label": "API Token",
                "description": "API token from replicate.com/account/api-tokens",
                "fields": {
                    "REPLICATE_API_TOKEN": {"label": "API Token", "secret": True, "required": True},
                },
            },
        ],
    },
    "deepgram": {
        "label": "Deepgram",
        "auth_methods": [
            {
                "id": "api_key",
                "label": "API Key",
                "description": "API key from deepgram.com",
                "fields": {
                    "DEEPGRAM_API_KEY": {"label": "API Key", "secret": True, "required": True},
                },
            },
        ],
    },
}


def _get_secret_fields() -> set:
    """Derive the set of secret field keys from the schema."""
    secrets = set()
    for schema in PROVIDER_SCHEMA.values():
        for method in schema["auth_methods"]:
            for fk, fdef in method["fields"].items():
                if fdef.get("secret"):
                    secrets.add(fk)
    return secrets


def _load_multiprofile() -> dict:
    mp_file = _multiprofile_file()
    if not mp_file.exists():
        return {}
    try:
        with open(mp_file, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_multiprofile(data: dict):
    mp_file = _state_dir(create=True) / "profiles.json"
    with open(mp_file, "w") as f:
        json.dump(data, f, indent=2)
    os.chmod(mp_file, 0o600)


def _mask(val: str) -> str:
    if len(val) <= 6:
        return "•" * len(val)
    return val[:3] + "•" * (len(val) - 6) + val[-3:]


def _gen_id() -> str:
    import random, string
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


# ── Listing ──────────────────────────────────────────────────────────────────

def get_all_profiles() -> dict:
    """Return all providers with profiles and auth_methods for the UI."""
    mp = _load_multiprofile()
    secret_fields = _get_secret_fields()
    out = {}

    for prov_key, schema in PROVIDER_SCHEMA.items():
        prov_data = mp.get(prov_key, {})

        items = []
        active_id = None
        for p in prov_data.get("profiles", []):
            masked = {}
            for fk, fv in p.get("fields", {}).items():
                masked[fk] = _mask(fv) if fk in secret_fields and fv else fv
            items.append({
                "id": p["id"],
                "label": p.get("label", ""),
                "auth_method": p.get("auth_method", schema["auth_methods"][0]["id"]),
                "fields": masked,
            })
            if p.get("active"):
                active_id = p["id"]

        out[prov_key] = {
            "label": schema["label"],
            "auth_methods": schema["auth_methods"],
            "items": items,
            "active_id": active_id,
        }
    return out


# ── CRUD ─────────────────────────────────────────────────────────────────────

def add_profile(provider: str, label: str = "", fields: dict = None,
                auth_method: str = "") -> dict:
    if provider not in PROVIDER_SCHEMA:
        raise ValueError(f"Unknown provider: {provider}")
    schema = PROVIDER_SCHEMA[provider]
    if not auth_method:
        auth_method = schema["auth_methods"][0]["id"]

    mp = _load_multiprofile()
    if provider not in mp:
        mp[provider] = {"profiles": []}
    pid = _gen_id()
    is_first = len(mp[provider]["profiles"]) == 0
    item = {
        "id": pid,
        "label": label or f"Profile {len(mp[provider]['profiles']) + 1}",
        "active": is_first,
        "auth_method": auth_method,
        "fields": fields or {},
    }
    mp[provider]["profiles"].append(item)
    _save_multiprofile(mp)
    if is_first:
        _apply_profile(provider, item)
    return item


def delete_profile(provider: str, profile_id: str) -> bool:
    mp = _load_multiprofile()
    profiles = mp.get(provider, {}).get("profiles", [])
    new = [p for p in profiles if p["id"] != profile_id]
    if len(new) == len(profiles):
        return False
    mp[provider]["profiles"] = new
    _save_multiprofile(mp)
    return True


def activate_profile(provider: str, profile_id: str) -> bool:
    mp = _load_multiprofile()
    profiles = mp.get(provider, {}).get("profiles", [])
    found = False
    for p in profiles:
        if p["id"] == profile_id:
            p["active"] = True
            found = True
            _apply_profile(provider, p)
        else:
            p["active"] = False
    if found:
        _save_multiprofile(mp)
    return found


def update_active_field(provider: str, field_name: str, value: str) -> bool:
    mp = _load_multiprofile()
    profiles = mp.get(provider, {}).get("profiles", [])
    for p in profiles:
        if p.get("active"):
            p.setdefault("fields", {})[field_name] = value
            _save_multiprofile(mp)
            os.environ[field_name] = value
            return True
    return False


# ── Apply credentials (special auth flows) ───────────────────────────────────

def _apply_profile(provider: str, profile: dict):
    """Apply a single profile's credentials to os.environ with special handling."""
    method = profile.get("auth_method", "")
    fields = profile.get("fields", {})

    if method == "env":
        return

    if provider == "aws" and method == "profile":
        _apply_aws_profile(fields)
        return

    if provider == "aws" and method == "assume_role":
        _apply_aws_assume_role(fields)
        return

    if provider == "gcp" and method == "json_inline":
        _apply_gcp_inline_json(fields)
        return

    if provider == "gcp" and method == "adc":
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        return

    for k, v in fields.items():
        if v:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]


def _apply_profile_fields(fields: dict):
    """Legacy: apply flat fields to env (backward compat for old profiles)."""
    for k, v in fields.items():
        if v:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]


def _apply_aws_profile(fields: dict):
    """Read ~/.aws/credentials for the named profile and set env vars."""
    profile_name = fields.get("AWS_PROFILE_NAME", "default")
    region = fields.get("AWS_DEFAULT_REGION", "us-east-1")

    creds = _read_aws_credentials_file(profile_name)
    if creds.get("aws_access_key_id"):
        os.environ["AWS_ACCESS_KEY_ID"] = creds["aws_access_key_id"]
    if creds.get("aws_secret_access_key"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = creds["aws_secret_access_key"]
    if creds.get("aws_session_token"):
        os.environ["AWS_SESSION_TOKEN"] = creds["aws_session_token"]
    os.environ["AWS_DEFAULT_REGION"] = region


def _apply_aws_assume_role(fields: dict):
    """Use STS to assume a role, then set temporary credentials."""
    os.environ["AWS_ACCESS_KEY_ID"] = fields.get("AWS_ACCESS_KEY_ID", "")
    os.environ["AWS_SECRET_ACCESS_KEY"] = fields.get("AWS_SECRET_ACCESS_KEY", "")
    region = fields.get("AWS_DEFAULT_REGION", "us-east-1")
    os.environ["AWS_DEFAULT_REGION"] = region

    role_arn = fields.get("AWS_ROLE_ARN", "")
    session_name = fields.get("AWS_ROLE_SESSION_NAME", "mlsec-session")
    if not role_arn:
        return

    try:
        import boto3
        sts = boto3.client("sts", region_name=region)
        resp = sts.assume_role(RoleArn=role_arn, RoleSessionName=session_name)
        creds = resp["Credentials"]
        os.environ["AWS_ACCESS_KEY_ID"] = creds["AccessKeyId"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = creds["SecretAccessKey"]
        os.environ["AWS_SESSION_TOKEN"] = creds["SessionToken"]
    except Exception as exc:
        _creds_logger.warning("Failed to assume AWS role %s: %s", role_arn, exc)


def _apply_gcp_inline_json(fields: dict):
    """Write inline JSON to shared state and set GOOGLE_APPLICATION_CREDENTIALS."""
    json_content = fields.get("GCP_SERVICE_ACCOUNT_JSON", "")
    if not json_content:
        return
    tmp_dir = _state_dir(create=True)
    tmp_file = tmp_dir / "gcp_service_account.json"
    tmp_file.write_text(json_content, encoding="utf-8")
    os.chmod(tmp_file, 0o600)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(tmp_file)


def _apply_environment_overrides() -> None:
    """Apply runtime env conveniences after stored profiles load."""
    gcp_inline_json = (
        os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
        or os.environ.get("GOOGLE_VISION_CREDENTIALS_JSON")
    )
    if gcp_inline_json:
        _apply_gcp_inline_json({"GCP_SERVICE_ACCOUNT_JSON": gcp_inline_json})


# ── AWS profile discovery ────────────────────────────────────────────────────

def _read_aws_credentials_file(profile_name: str = "default") -> dict:
    """Parse ~/.aws/credentials for a specific profile."""
    creds_path = Path.home() / ".aws" / "credentials"
    if not creds_path.exists():
        return {}
    cp = configparser.ConfigParser()
    cp.read(str(creds_path))
    if profile_name not in cp:
        return {}
    section = cp[profile_name]
    return {
        "aws_access_key_id": section.get("aws_access_key_id", ""),
        "aws_secret_access_key": section.get("aws_secret_access_key", ""),
        "aws_session_token": section.get("aws_session_token", ""),
    }


def list_aws_profiles() -> list:
    """Return profile names from ~/.aws/credentials."""
    creds_path = Path.home() / ".aws" / "credentials"
    if not creds_path.exists():
        return []
    cp = configparser.ConfigParser()
    cp.read(str(creds_path))
    return list(cp.sections())


# ── Connection testing ───────────────────────────────────────────────────────

def _get_env_keys_for_provider(provider: str) -> list:
    """Return the expected env var keys for auto-detect checking."""
    mapping = {
        "aws": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        "azure": ["AZURE_VISION_ENDPOINT", "AZURE_VISION_KEY"],
        "gcp": ["GOOGLE_APPLICATION_CREDENTIALS"],
        "huggingface": ["HF_API_TOKEN"],
        "elevenlabs": ["ELEVENLABS_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "replicate": ["REPLICATE_API_TOKEN"],
        "deepgram": ["DEEPGRAM_API_KEY"],
    }
    return mapping.get(provider, [])


def detect_env_credentials(provider: str) -> dict:
    """Check which env vars are already set for a provider."""
    keys = _get_env_keys_for_provider(provider)
    found = {}
    for k in keys:
        val = os.environ.get(k, "")
        found[k] = bool(val)
    all_present = all(found.values()) if found else False
    return {"keys": found, "detected": all_present}


def test_connection(provider: str, profile_id: str) -> dict:
    """Lightweight connectivity test for a provider profile."""
    mp = _load_multiprofile()
    profiles = mp.get(provider, {}).get("profiles", [])
    profile = next((p for p in profiles if p["id"] == profile_id), None)
    if not profile:
        return {"ok": False, "error": "Profile not found"}

    method = profile.get("auth_method", "")
    fields = profile.get("fields", {})

    try:
        if provider == "aws":
            return _test_aws(method, fields)
        elif provider == "azure":
            return _test_azure(method, fields)
        elif provider == "gcp":
            return _test_gcp(method, fields)
        elif provider == "huggingface":
            return _test_huggingface(method, fields)
        elif provider == "elevenlabs":
            return _test_elevenlabs(method, fields)
        elif provider == "openai":
            return _test_openai(method, fields)
        elif provider == "anthropic":
            return _test_anthropic(method, fields)
        elif provider == "replicate":
            return _test_replicate(method, fields)
        elif provider == "deepgram":
            return _test_deepgram(method, fields)
        return {"ok": False, "error": f"No test available for {provider}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _test_aws(method, fields):
    import boto3
    if method == "env":
        sts = boto3.client("sts")
    elif method == "profile":
        profile_name = fields.get("AWS_PROFILE_NAME", "default")
        session = boto3.Session(profile_name=profile_name)
        sts = session.client("sts")
    elif method == "assume_role":
        sts = boto3.client(
            "sts",
            aws_access_key_id=fields.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=fields.get("AWS_SECRET_ACCESS_KEY"),
            region_name=fields.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
        role_arn = fields.get("AWS_ROLE_ARN", "")
        if role_arn:
            resp = sts.assume_role(
                RoleArn=role_arn,
                RoleSessionName=fields.get("AWS_ROLE_SESSION_NAME", "test"),
            )
            return {"ok": True, "message": f"Assumed role, expires {resp['Credentials']['Expiration']}"}
    else:
        sts = boto3.client(
            "sts",
            aws_access_key_id=fields.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=fields.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=fields.get("AWS_SESSION_TOKEN") or None,
            region_name=fields.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
    identity = sts.get_caller_identity()
    return {"ok": True, "message": f"Account {identity['Account']}, ARN {identity['Arn']}"}


def _test_azure(method, fields):
    import urllib.request
    endpoint = fields.get("AZURE_VISION_ENDPOINT") or os.environ.get("AZURE_VISION_ENDPOINT", "")
    if not endpoint:
        return {"ok": False, "error": "No endpoint URL configured"}
    if method == "service_principal":
        return {"ok": True, "message": f"Service principal configured for {endpoint} (full test requires azure-identity SDK)"}
    key = fields.get("AZURE_VISION_KEY") or os.environ.get("AZURE_VISION_KEY", "")
    if not key:
        return {"ok": False, "error": "No API key configured"}
    url = f"{endpoint.rstrip('/')}/vision/v3.2/models"
    req = urllib.request.Request(url, headers={"Ocp-Apim-Subscription-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"ok": resp.status == 200, "message": f"Connected to {endpoint}"}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"ok": True, "message": f"Endpoint reachable: {endpoint}"}
        return {"ok": False, "error": f"HTTP {e.code}: {e.reason}"}


def _test_gcp(method, fields):
    if method == "adc":
        return {"ok": True, "message": "Using Application Default Credentials (validated at call time)"}
    if method == "json_inline":
        blob = fields.get("GCP_SERVICE_ACCOUNT_JSON", "")
        try:
            data = json.loads(blob)
            email = data.get("client_email", "unknown")
            return {"ok": True, "message": f"Service account: {email}"}
        except Exception:
            return {"ok": False, "error": "Invalid JSON content"}
    path = fields.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not path or not Path(path).exists():
        return {"ok": False, "error": f"File not found: {path}"}
    try:
        data = json.loads(Path(path).read_text())
        email = data.get("client_email", "unknown")
        return {"ok": True, "message": f"Service account: {email}"}
    except Exception:
        return {"ok": False, "error": "Could not parse JSON key file"}


def _test_huggingface(method, fields):
    import urllib.request
    token = fields.get("HF_API_TOKEN") or os.environ.get("HF_API_TOKEN", "")
    if not token and method != "env":
        return {"ok": False, "error": "No token configured"}
    req = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}"} if token else {},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
        return {"ok": True, "message": f"Authenticated as {data.get('name', 'unknown')}"}


def _test_elevenlabs(method, fields):
    import urllib.request
    key = fields.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        return {"ok": False, "error": "No API key configured"}
    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/user",
        headers={"xi-api-key": key},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return {"ok": True, "message": "API key valid"}


def _test_openai(method, fields):
    import urllib.request
    key = fields.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return {"ok": False, "error": "No API key configured"}
    req = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return {"ok": True, "message": "API key valid"}


def _test_anthropic(method, fields):
    import urllib.request
    key = fields.get("ANTHROPIC_API_KEY", "")
    if not key:
        return {"ok": False, "error": "No API key configured"}
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
        data=b'{"model":"claude-3-haiku-20240307","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}',
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return {"ok": True, "message": "API key valid"}
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, "read") else ""
        if e.code == 401:
            return {"ok": False, "error": "Invalid API key"}
        return {"ok": True, "message": f"API key accepted (HTTP {e.code})"}


def _test_replicate(method, fields):
    import urllib.request
    token = fields.get("REPLICATE_API_TOKEN", "")
    if not token:
        return {"ok": False, "error": "No API token configured"}
    req = urllib.request.Request(
        "https://api.replicate.com/v1/account",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
        return {"ok": True, "message": f"Authenticated as {data.get('username', 'unknown')}"}


def _test_deepgram(method, fields):
    import urllib.request
    key = fields.get("DEEPGRAM_API_KEY", "")
    if not key:
        return {"ok": False, "error": "No API key configured"}
    req = urllib.request.Request(
        "https://api.deepgram.com/v1/projects",
        headers={"Authorization": f"Token {key}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return {"ok": True, "message": "API key valid"}
