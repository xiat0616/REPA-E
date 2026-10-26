"""Compare the system/reference bytes and test a process-local driver override."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

directory = Path("checkpoints/diagnostics/reference_driver_595.91.07").resolve()
reference = directory / "libcuda.so.595.91.07"
system = Path("/usr/lib/x86_64-linux-gnu/libcuda.so.595.91.07")
a = system.read_bytes()
b = reference.read_bytes()
assert len(a) == len(b)
print("SYSTEM SHA256", hashlib.sha256(a).hexdigest(), flush=True)
print("REFERENCE SHA256", hashlib.sha256(b).hexdigest(), flush=True)
differences = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
print("DIFFERENT BYTE COUNT", len(differences), flush=True)
if differences:
    print("FIRST DIFFERENT OFFSET", hex(differences[0]), "LAST", hex(differences[-1]), flush=True)
print("SYSTEM CRASH REGION", a[0x2b8d00:0x2b8d70].hex(), flush=True)
print("REFERENCE CRASH REGION", b[0x2b8d00:0x2b8d70].hex(), flush=True)
print("SYSTEM TOTAL ZERO BYTES", a.count(0), "REFERENCE", b.count(0), flush=True)
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = str(directory)
print("PROCESS-LOCAL LD_LIBRARY_PATH", env["LD_LIBRARY_PATH"], flush=True)
binary = f"/tmp/repa_cuda_reference_probe_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
subprocess.run(["cc", "-g", "-O0", "-rdynamic", "cuda_native_probe.c", "-ldl", "-o", binary], check=True)
result = subprocess.run([binary, "driver", str(directory / "libcuda.so.1")], env=env, timeout=60)
print("REFERENCE NATIVE EXIT", result.returncode, flush=True)
code = """
import torch
print('TORCH',torch.__version__,flush=True)
print('CUDA AVAILABLE',torch.cuda.is_available(),flush=True)
x=torch.ones(32,device='cuda:0')
y=x @ x
torch.cuda.synchronize()
print('CUDA COMPUTATION OK',y.item(),torch.cuda.get_device_name(),flush=True)
print('LOADED DRIVER',sorted({line.split()[-1] for line in open('/proc/self/maps') if 'libcuda.so' in line}),flush=True)
"""
result = subprocess.run([sys.executable, "-u", "-X", "faulthandler", "-c", code], env=env, timeout=90)
print("REFERENCE TORCH EXIT", result.returncode, flush=True)
print("REFERENCE VALIDATION COMPLETE", flush=True)
