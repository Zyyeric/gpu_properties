import cupy as cp
import textwrap

def show_gpu_info(idx: int | None = None) -> None:
    """Pretty-print key CUDA properties via CuPy."""
    if idx is None:
        indices = range(cp.cuda.runtime.getDeviceCount())
    else:
        indices = [idx]

    for i in indices:
        props = cp.cuda.runtime.getDeviceProperties(i)

        name  = props["name"].decode()
        major = props["major"]
        minor = props["minor"]

        print(textwrap.dedent(f"""
        === GPU {i}: {name} ===
        Compute capability        : {major}.{minor}
        SM count (multiprocessors): {props['multiProcessorCount']}
        Max threads / SM          : {props['maxThreadsPerMultiProcessor']}
        Max threads / block       : {props['maxThreadsPerBlock']}
        Warp size                 : {props['warpSize']}
        Shared memory / block     : {props['sharedMemPerBlock'] // 1024} KiB
        L2 cache size             : {props['l2CacheSize'] // 1024} KiB
        Total global memory       : {props['totalGlobalMem'] / 2**30:.2f} GiB
        """).strip(), end="\n\n")