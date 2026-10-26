"""Read-only comparison of native driver binaries and fault details."""
import hashlib
import os
from pathlib import Path
import shutil
import socket
import subprocess

print("HOST", socket.gethostname(), flush=True)
for key in ["LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "CUDA_VISIBLE_DEVICES", "CUDA_INJECTION64_PATH", "NVML_INJECTION64_PATH"]:
    print("ENV", key, os.environ.get(key), flush=True)
for path in ["/etc/ld.so.preload", "/proc/driver/nvidia/version", "/sys/module/nvidia/version", "/sys/module/nvidia/srcversion"]:
    p = Path(path)
    if p.exists():
        try:
            print("SYSTEM", path, p.read_text()[:4000], flush=True)
        except OSError as error:
            print("SYSTEM", path, str(error), flush=True)
for stem in ["libcuda.so.1", "libnvidia-ml.so.1", "libnvidia-ptxjitcompiler.so.1", "libnvidia-nvvm.so.4"]:
    p = Path("/usr/lib/x86_64-linux-gnu") / stem
    if not p.exists():
        continue
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    print("BINARY", p.resolve(), "SIZE", p.stat().st_size, "SHA256", h.hexdigest(), flush=True)
    if shutil.which("readelf"):
        result = subprocess.run(["readelf", "-n", str(p)], text=True, capture_output=True)
        print("BUILD_ID", "\n".join(line for line in result.stdout.splitlines() if "Build ID" in line), flush=True)
driver = "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
if shutil.which("objdump"):
    subprocess.run(["objdump", "-d", "--start-address=0x2b8d00", "--stop-address=0x2b8d70", driver], timeout=30)
compiler = shutil.which("cc") or shutil.which("gcc")
binary = f"/tmp/repa_cuda_driver_detail_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
if compiler:
    subprocess.run([compiler, "-g", "-O0", "-rdynamic", "cuda_native_probe.c", "-ldl", "-o", binary], check=True)
    print("NATIVE DRIVER REPRODUCTION", flush=True)
    result = subprocess.run([binary, "driver", "libcuda.so.1"], timeout=60)
    print("NATIVE EXIT", result.returncode, flush=True)
    if shutil.which("strace") and result.returncode:
        trace = Path("checkpoints/diagnostics") / f"cuda_native.{os.environ.get('SLURM_JOB_ID', os.getpid())}.strace"
        result = subprocess.run(["strace", "-f", "-e", "trace=openat,readlink,ioctl,mmap,mprotect", "-o", str(trace), binary, "driver", "libcuda.so.1"], timeout=60)
        print("STRACE EXIT", result.returncode, "PATH", trace, flush=True)
        print("STRACE TAIL", "\n".join(trace.read_text(errors="replace").splitlines()[-45:]), flush=True)
print("DRIVER FILE DIAGNOSTICS COMPLETE", flush=True)
