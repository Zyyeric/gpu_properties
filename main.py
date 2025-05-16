import argparse 
import subprocess 
from pathlib import Path
import sys 
import setup
import properties


def run_properties(gpu_idx: int | None = None) -> None:
    """ Display the GPU Properties Info"""
    properties.show_gpu_info(gpu_idx)


def main() -> None: 
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--venv",
        default=Path("gpu_prop"),
        type=Path,
        help="Virtual‑env directory (default: ./gpu_prop)",
    )
    ap.add_argument(
        "--gpu_idx",
        type=int,
        help="Show only a single GPU index (default: all GPUs)",
    )

    args = ap.parse_args()

    # Create or reuse the virtual environment
    venv_python = setup.ensure_venv(args.venv)

    # Install the corresponding version of CuPy to the Cuda Version
    setup.install_cupy(venv_python, setup.get_cuda_version()) 

    # Print out the properties of the GPU devices 
    run_properties(args.gpu_idx)

if __name__ == "__main__":
    main()

