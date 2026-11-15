"""
Bit-Reproducibility and Provenance Verification Script.
Proves deterministic generation by generating the dataset twice with the same seed,
recording config hash, git code SHA, and asserting byte-level SHA-256 equality.
"""

import hashlib
import json
import subprocess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from acoustisolar_schemas.module import ModuleTwinConfig
from acoustisolar_schemas.hashing import compute_config_hash
from acoustisolar_ml.generator import DatasetGenerator

def get_git_code_sha() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_DIRTY"

def run_bit_reproducibility_test(seed: int = 9999, n_events: int = 200) -> dict:
    cfg = ModuleTwinConfig(module_id="MOD-BIT-REPRO")
    cfg_hash = compute_config_hash(cfg)
    code_sha = get_git_code_sha()
    
    print(f"=== Running Bit-Reproducibility Verification ===")
    print(f"  Config Hash: {cfg_hash}")
    print(f"  Code Git SHA: {code_sha}")
    print(f"  Random Seed: {seed}")
    
    # Run 1
    print("  [1/2] Generating Run 1...")
    gen1 = DatasetGenerator(module_config=cfg, seed=seed)
    waves1, df1 = gen1.generate_batch(n_events=n_events, crack_ratio=0.5)
    bytes1 = waves1.tobytes()
    sha256_waves1 = hashlib.sha256(bytes1).hexdigest()
    
    # Run 2
    print("  [2/2] Generating Run 2...")
    gen2 = DatasetGenerator(module_config=cfg, seed=seed)
    waves2, df2 = gen2.generate_batch(n_events=n_events, crack_ratio=0.5)
    bytes2 = waves2.tobytes()
    sha256_waves2 = hashlib.sha256(bytes2).hexdigest()
    
    # Assertions
    arrays_identical = bool(np.array_equal(waves1, waves2))
    sha_identical = bool(sha256_waves1 == sha256_waves2)
    meta_identical = bool(df1.drop(columns=["feature_vector"]).equals(df2.drop(columns=["feature_vector"])))
    
    print(f"  Arrays bit-equal: {arrays_identical}")
    print(f"  SHA-256 Waves 1: {sha256_waves1}")
    print(f"  SHA-256 Waves 2: {sha256_waves2}")
    print(f"  SHA-256 Match:   {sha_identical}")
    print(f"  Metadata Match:  {meta_identical}")
    
    proof = {
        "status": "PASS" if (arrays_identical and sha_identical and meta_identical) else "FAIL",
        "seed": seed,
        "n_events": n_events,
        "config_hash": cfg_hash,
        "code_git_sha": code_sha,
        "sha256_waveforms_run1": sha256_waves1,
        "sha256_waveforms_run2": sha256_waves2,
        "byte_length": len(bytes1),
        "bit_identical": arrays_identical and sha_identical,
    }
    
    out_file = Path("data/reproducibility_proof.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(proof, f, indent=2)
        
    return proof

if __name__ == "__main__":
    res = run_bit_reproducibility_test()
    print("\nResult:", res["status"])

