import hashlib
import json
from pydantic import BaseModel
from typing import Any

def canonical_json_dump(obj: Any) -> str:
    """Serialize Pydantic models or dicts to a deterministic, sorted-key JSON string."""
    if isinstance(obj, BaseModel):
        data = obj.model_dump(mode="json")
    elif isinstance(obj, dict):
        data = obj
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    return json.dumps(data, sort_keys=True, separators=(",", ":"))

def compute_config_hash(model: BaseModel) -> str:
    """Compute a reproducible SHA-256 hash of a Pydantic model configuration."""
    canonical_str = canonical_json_dump(model)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
