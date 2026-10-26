"""Compare native CUDA initialization and real package/library versions without training."""
import glob
import importlib.metadata as metadata
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys


def probe(name, args, environment=None, timeout=60):
    print(f"PROBE START {name}: {args}", flush=True)
    env = os.environ.copy()
    if environment:
        env.update(environment)
    try:
        result = subprocess.run(args, env=env, timeout=timeout)
        print(f"PROBE END {name}: exit={result.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"PROBE END {name}: TIMEOUT", flush=True)


print("HOST", socket.gethostname(), "PYTHON", sys.version, flush=True)
print("ENV", {k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "LD_PRELOAD", "CUDA_FORCE_PRELOAD_LIBRARIES"]}, flush=True)
packages = ["torch", "torchvision", "triton", "accelerate", "timm", "numpy",
            "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12", "nvidia-cublas-cu12",
            "nvidia-cudnn-cu12", "nvidia-nccl-cu12", "nvidia-nvjitlink-cu12"]
for name in packages:
    try:
        print("PACKAGE", name, metadata.version(name), flush=True)
    except metadata.PackageNotFoundError:
        print("PACKAGE", name, "NOT INSTALLED", flush=True)
version_path = Path("/proc/driver/nvidia/version")
if version_path.exists():
    print("KERNEL DRIVER", version_path.read_text(), flush=True)
probe("gpu_inventory", ["nvidia-smi", "--query-gpu=name,driver_version,pci.bus_id", "--format=csv,noheader"])
ldconfig = shutil.which("ldconfig") or "/sbin/ldconfig"
result = subprocess.run([ldconfig, "-p"], text=True, capture_output=True)
for line in result.stdout.splitlines():
    if re.search(r"libcuda\.so|libnvidia-(ptxjitcompiler|nvvm|ml)\.so", line):
        path = line.split("=>")[-1].strip()
        print("DRIVER LIBRARY", line.strip(), "REALPATH", os.path.realpath(path), flush=True)
print("DEBUGGERS", {name: shutil.which(name) for name in ["gdb", "cuda-gdb", "lldb", "strace", "cc", "gcc"]}, flush=True)
print("CUDA_GDB_PATHS", glob.glob("/usr/local/cuda*/bin/cuda-gdb"), flush=True)
runtime_libs = glob.glob(str(Path(sys.prefix) / "lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*"))
print("WHEEL RUNTIME", runtime_libs, flush=True)

compiler = shutil.which("cc") or shutil.which("gcc")
binary = f"/tmp/repa_cuda_native_probe_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
if compiler:
    result = subprocess.run([compiler, "-g", "-O0", "-rdynamic", "cuda_native_probe.c", "-ldl", "-o", binary])
    if result.returncode == 0:
        probe("native_driver_default", [binary, "driver", "libcuda.so.1"])
        probe("native_driver_without_jit_preload", [binary, "driver", "libcuda.so.1"], {"CUDA_FORCE_PRELOAD_LIBRARIES": "0"})
        probe("native_driver_with_jit_preload", [binary, "driver", "libcuda.so.1"], {"CUDA_FORCE_PRELOAD_LIBRARIES": "1"})
        if runtime_libs:
            probe("native_runtime_default", [binary, "runtime", runtime_libs[0]])

torch_code = """
import torch
print('TORCH IMPORT',torch.__version__,flush=True)
print('CUDA AVAILABLE',torch.cuda.is_available(),flush=True)
x=torch.ones(32,device='cuda:0')
y=x @ x
torch.cuda.synchronize()
print('CUDA COMPUTATION OK',y.item(),torch.cuda.get_device_name(),flush=True)
for line in open('/proc/self/maps'):
 if any(s in line for s in ['libcuda.', 'libcudart.', 'libnvidia-ptxjitcompiler.', 'libnvidia-nvvm.']):
  print('LOADED',line.strip(),flush=True)
"""
probe("torch_default", [sys.executable, "-u", "-X", "faulthandler", "-c", torch_code])
probe("torch_without_jit_preload", [sys.executable, "-u", "-X", "faulthandler", "-c", torch_code], {"CUDA_FORCE_PRELOAD_LIBRARIES": "0"})
probe("torch_with_jit_preload", [sys.executable, "-u", "-X", "faulthandler", "-c", torch_code], {"CUDA_FORCE_PRELOAD_LIBRARIES": "1"})
print("VERSION DIAGNOSTICS COMPLETE", flush=True)
