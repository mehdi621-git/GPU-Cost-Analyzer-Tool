"""gpu-cost-meter: measure GPU inference cost and idle waste."""

import time
import json
import os
import subprocess
from functools import wraps
from collections import defaultdict

__version__ = "0.1.0"

_RECORDS = []
_GPU_RATE = float(os.getenv("GPU_RATE", "0.526"))  # USD/hr, g4dn.xlarge on-demand


def _get_gpu_util():
    """Return GPU utilization percent, or -1 if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=1,
        )
        return float(out.decode().strip().split("\n")[0])
    except Exception:
        return -1


def meter(endpoint_name):
    """Decorator: record duration and GPU utilization per inference call."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            gpu_before = _get_gpu_util()
            ok = True
            try:
                result = func(*args, **kwargs)
            except Exception:
                ok = False
                raise
            finally:
                elapsed = time.perf_counter() - start
                gpu_after = _get_gpu_util()
                _RECORDS.append({
                    "endpoint": endpoint_name,
                    "duration_s": elapsed,
                    "gpu_util_avg": (gpu_before + gpu_after) / 2
                    if gpu_before >= 0 and gpu_after >= 0 else -1,
                    "ok": ok,
                    "ts": time.time(),
                })
            return result
        return wrapper
    return decorator


def report():
    """Print a cost report to stdout."""
    if not _RECORDS:
        print("No data recorded.")
        return

    total = len(_RECORDS)
    window_s = _RECORDS[-1]["ts"] - _RECORDS[0]["ts"]
    window_cost = _GPU_RATE * (window_s / 3600)

    idle_records = [r for r in _RECORDS
                    if 0 <= r["gpu_util_avg"] < 5]
    idle_cost = sum(r["duration_s"] for r in idle_records) * _GPU_RATE / 3600
    cpri = (window_cost / total) * 1000 if total else 0.0

    print("\n=== GPU Inference Cost Report ===")
    print(f"Calls: {total}")
    print(f"Window: {window_s:.1f}s")
    print(f"GPU rate: ${_GPU_RATE}/hr")
    print(f"Window cost: ${window_cost:.4f}")
    print(f"Cost per 1k inferences: ${cpri:.4f}")
    print(f"Idle cost (util<5%): ${idle_cost:.4f}  <-- paid but wasted")

    by_ep = defaultdict(list)
    for r in _RECORDS:
        by_ep[r["endpoint"]].append(r)

    print("\nBy endpoint:")
    for ep, rs in by_ep.items():
        utils = [x["gpu_util_avg"] for x in rs if x["gpu_util_avg"] >= 0]
        avg_util = sum(utils) / len(utils) if utils else -1
        util_str = f"{avg_util:.1f}%" if avg_util >= 0 else "n/a"
        print(f"  {ep}: {len(rs)} calls, avg GPU util {util_str}")


def reset():
    """Clear recorded data (useful between tests)."""
    _RECORDS.clear()