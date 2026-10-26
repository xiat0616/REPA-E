"""Run independent tiny CUDA-startup probes in fresh interpreters."""
import os
import socket
import subprocess
import sys

print(f"CUDA probes on {socket.gethostname()}", flush=True)
print({key: os.environ.get(key) for key in ["CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "LD_PRELOAD"]}, flush=True)
probes = {
    "plain_torch_cuda": """
import torch
print('torch imported', torch.__version__, flush=True)
print('CUDA available query', flush=True)
print(torch.cuda.is_available(), flush=True)
print('CUDA allocation', flush=True)
x = torch.ones(4, device='cuda:0')
torch.cuda.synchronize()
print('allocation OK', x.cpu(), flush=True)
""",
    "training_import_then_cuda": """
import train_ldm_only
import torch
print('training imports OK; querying CUDA', flush=True)
print(torch.cuda.is_available(), flush=True)
x = torch.ones(4, device='cuda:0')
torch.cuda.synchronize()
print('allocation OK', x.cpu(), flush=True)
""",
    "cuda_before_training_import": """
import torch
print('querying CUDA first', flush=True)
print(torch.cuda.is_available(), flush=True)
x = torch.ones(4, device='cuda:0')
torch.cuda.synchronize()
print('initial allocation OK; importing training', flush=True)
import train_ldm_only
from accelerate import Accelerator
a = Accelerator(mixed_precision='bf16')
print('Accelerator OK', a.device, flush=True)
""",
}
for name, code in probes.items():
    print(f"PROBE START {name}", flush=True)
    result = subprocess.run([sys.executable, "-u", "-X", "faulthandler", "-c", code], timeout=90)
    print(f"PROBE END {name}: exit={result.returncode}", flush=True)
