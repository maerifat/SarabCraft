"""
Global variables API routes.
"""

import os
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from plugins._base import get_all_vars, set_var, delete_var

router = APIRouter()


class SetVarRequest(BaseModel):
    key: str
    value: str
    masked: bool = False
    description: str = ""


@router.get("/list")
def list_variables():
    """List all global variables (masked values hidden)."""
    return {"variables": get_all_vars(unmask=False)}


@router.post("/set")
def add_or_update_variable(req: SetVarRequest):
    """Add or update a global variable."""
    if not req.key.strip():
        raise HTTPException(400, "Key is required")
    entry = set_var(req.key.strip(), req.value, req.masked, req.description)
    if req.masked:
        entry = {**entry, "value": "••••••••"}
    return {"variable": entry}


@router.delete("/{key}")
def remove_variable(key: str):
    """Delete a global variable."""
    if not delete_var(key):
        raise HTTPException(404, "Variable not found")
    return {"ok": True}
