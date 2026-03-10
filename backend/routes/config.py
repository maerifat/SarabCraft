"""
Configuration and credentials API — multi-profile per provider.
"""

import os
import random
import string

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict

from utils.credentials import (
    get_all_profiles, add_profile, delete_profile, activate_profile,
    update_active_field, apply_credentials, load_credentials, save_credentials,
    test_connection, list_aws_profiles, detect_env_credentials,
)

router = APIRouter()


@router.get("")
def get_config():
    """Return all providers with their profiles (secrets masked)."""
    return get_all_profiles()


# ── Static AWS routes (must precede /{provider} wildcards) ──

@router.get("/aws/profiles")
def get_aws_profiles():
    """Return profile names from ~/.aws/credentials."""
    return {"profiles": list_aws_profiles()}

@router.get("/aws/buckets")
def list_s3_buckets():
    try:
        import boto3
    except ImportError:
        raise HTTPException(400, "boto3 not installed")

    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        raise HTTPException(400, "AWS credentials not configured — add an AWS profile first")

    try:
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        session_token = os.environ.get("AWS_SESSION_TOKEN") or None
        s3 = boto3.client(
            "s3", region_name=region, aws_access_key_id=key,
            aws_secret_access_key=secret,
            **({"aws_session_token": session_token} if session_token else {}),
        )
        resp = s3.list_buckets()
        buckets = [b["Name"] for b in resp.get("Buckets", [])]
        current = os.environ.get("AWS_TRANSCRIBE_BUCKET", "")
        return {"buckets": buckets, "current": current}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/aws/buckets/create")
def create_s3_bucket():
    try:
        import boto3
    except ImportError:
        raise HTTPException(400, "boto3 not installed")

    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        raise HTTPException(400, "AWS credentials not configured")

    try:
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        session_token = os.environ.get("AWS_SESSION_TOKEN") or None
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        bucket_name = f"mlsec-transcribe-{suffix}"
        s3 = boto3.client(
            "s3", region_name=region, aws_access_key_id=key,
            aws_secret_access_key=secret,
            **({"aws_session_token": session_token} if session_token else {}),
        )
        create_args = {"Bucket": bucket_name}
        if region != "us-east-1":
            create_args["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**create_args)
        update_active_field("aws", "AWS_TRANSCRIBE_BUCKET", bucket_name)
        return {"bucket": bucket_name, "created": True}
    except Exception as e:
        raise HTTPException(500, str(e))


class SetBucketRequest(BaseModel):
    bucket: str

@router.post("/aws/buckets/select")
def select_s3_bucket(req: SetBucketRequest):
    """Save selected bucket to the active AWS profile."""
    ok = update_active_field("aws", "AWS_TRANSCRIBE_BUCKET", req.bucket)
    if not ok:
        raise HTTPException(400, "No active AWS profile — add AWS credentials first")
    return {"success": True, "bucket": req.bucket}


# ── Dynamic provider routes ──

class AddProfileRequest(BaseModel):
    label: str = ""
    fields: Dict[str, str] = {}
    auth_method: str = ""

@router.post("/{provider}/add")
def add_provider_profile(provider: str, req: AddProfileRequest):
    try:
        item = add_profile(provider, req.label, req.fields, req.auth_method)
        return {"success": True, "item": item}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/{provider}/env-detect")
def detect_env(provider: str):
    """Check which env vars are already set for a provider."""
    return detect_env_credentials(provider)


@router.delete("/{provider}/{profile_id}")
def delete_provider_profile(provider: str, profile_id: str):
    ok = delete_profile(provider, profile_id)
    if not ok:
        raise HTTPException(404, "Profile not found")
    return {"success": True}


@router.post("/{provider}/{profile_id}/activate")
def activate_provider_profile(provider: str, profile_id: str):
    ok = activate_profile(provider, profile_id)
    if not ok:
        raise HTTPException(404, "Profile not found")
    return {"success": True}


@router.post("/{provider}/{profile_id}/test")
def test_provider_connection(provider: str, profile_id: str):
    """Test connectivity for a specific profile."""
    result = test_connection(provider, profile_id)
    return result


# ── Legacy flat endpoints (backward compat) ──

class CredentialsUpdate(BaseModel):
    HF_API_TOKEN: str = ""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_SESSION_TOKEN: str = ""
    AWS_DEFAULT_REGION: str = "us-east-1"
    AWS_TRANSCRIBE_BUCKET: str = ""
    AZURE_VISION_ENDPOINT: str = ""
    AZURE_VISION_KEY: str = ""
    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    ELEVENLABS_API_KEY: str = ""

@router.post("")
def update_credentials_legacy(data: CredentialsUpdate):
    try:
        save_credentials(data.model_dump())
        apply_credentials()
        return {"success": True, "message": "Credentials saved"}
    except Exception as e:
        raise HTTPException(500, f"Failed to save credentials: {str(e)}")
