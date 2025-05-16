"""
set_up.py — Auto‑install the **correct cupy or pytorch wheel** for your CUDA runtime.

  1. Detects CUDA via `nvcc --version`.
  2. Creates/uses a virtual‑env (`--venv DIR`) or the current interpreter.
  3. Installs the correct **cupy‑cuda11x / cupy‑cuda12x** or torch (if calls install_torch instead of install_cupy in main) package.

Example
-------
python set_up.py --venv gpu_prop
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ------------------------------------------------------------
# CUDA → wheel tag mapping
# ------------------------------------------------------------
CUDA_TAG_MAP: dict[str, str] = {
    "11.0": "cu111",  
    "11.1": "cu111",
    "11.3": "cu113",
    "11.6": "cu116",
    "11.7": "cu117",
    "11.8": "cu118",
    "12.1": "cu121",
    "12.4": "cu124",  
    "12.6": "cu126"
}

# torch version to install for each tag 
TORCH_VER = {
    "cu111": "1.13.1",
    "cu113": "1.13.1",
    "cu116": "1.13.1",
    "cu117": "2.0.1",
    "cu118": "2.2.2",
    "cu121": "2.3.1",
    "cu124": "2.6.0",
}

def run(cmd: list[str]):
    """Run *cmd*; echo and abort on failure."""
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def get_cuda_version() -> Optional[str]:
    try:
        out = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT).decode()
    except FileNotFoundError:
        return None
    m = re.search(r"release (\d+\.\d+),", out)
    return m.group(1) if m else None


def ensure_venv(path: Path) -> Path:
    if not path.exists():
        print(f"Creating virtual‑env at {path} …")
        run([sys.executable, "-m", "venv", str(path)])
    py = path / ("Scripts" if os.name == "nt" else "bin") / "python"
    return py.resolve()


def pip_install(py: Path, *args: str):
    run([str(py), "-m", "pip", "install", *args])


def install_torch(py: Path, cuda_ver: Optional[str]):
    """Install *torch* matched to CUDA; fall back to CPU‑only."""
    if cuda_ver is None:
        print("CUDA not detected")
        return

    tag = CUDA_TAG_MAP.get(cuda_ver)
    if tag is None or tag not in TORCH_VER:
        print(f"CUDA {cuda_ver} unsupported ")
        return

    version = TORCH_VER[tag]
    print(f"CUDA {cuda_ver} detected → installing torch=={version} from index {tag} …")
    pip_install(py, f"torch=={version}", "--extra-index-url", f"https://download.pytorch.org/whl/{tag}")


def install_cupy(py: Path, cuda_ver: Optional[str]):
    """Install the CuPy wheel that matches *cuda_ver* (11.x → 11x, 12.x → 12x)."""
    if cuda_ver is None:
        print("CUDA runtime not found → installing CPU-only cupy …")
        pip_install(py, "cupy")
        return

    major = cuda_ver.split(".")[0]
    if major == "11":
        pkg = "cupy-cuda11x"
    elif major == "12":
        pkg = "cupy-cuda12x"
    else:
        print(f"Unsupported CUDA {cuda_ver}")
        return

    print(f"CUDA {cuda_ver} detected → installing {pkg} …")
    pip_install(py, pkg)