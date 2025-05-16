# GPU Properties Viewer

This lightweight Python tool detects your system's CUDA-compatible GPUs and prints their detailed hardware properties using [CuPy](https://github.com/cupy/cupy).

It automatically:
- Sets up or reuses a virtual environment
- Installs the correct CuPy version based on your local CUDA runtime
- Prints SM count, warp size, memory specs, and more

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/gpu_properties.git
cd gpu_properties
python main.py             # Automatically sets up venv + prints all GPUs
python main.py --gpu_idx 0  # Print only GPU 0
````

---

## 🔍 Example Output

```
=== GPU 0: NVIDIA A30X ===
Compute capability        : 8.0
SM count (multiprocessors): 56
Max threads / SM          : 2048
Max threads / block       : 1024
Warp size                 : 32
Shared memory / block     : 48 KiB
L2 cache size             : 24576 KiB
Total global memory       : 23.60 GiB
```

---

## ⚙️ Notes

* Only requires Python 3.7+
* No need to manage CUDA manually — `main.py` auto-detects and installs the matching CuPy wheel
* The virtual environment defaults to `./gpu_prop`, but you can override it:

```bash
python main.py --venv myenv
```

---

## 📁 File Structure

```
gpu_properties/
├── main.py         # CLI entrypoint
├── setup.py        # Virtualenv + CuPy installer
├── properties.py   # GPU info printing logic
├── README.md
└── .gitignore
```