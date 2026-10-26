"""Small staged native-startup diagnostic; does not load data or train a model."""
import faulthandler
import os
import socket
import sys

faulthandler.enable(all_threads=True)


def stage(message):
    print(f"[rank {os.environ.get('RANK', '?')}] {message}", flush=True)


stage(f"Python started: {sys.version}; host={socket.gethostname()}")
for name in ["torch", "wandb", "PIL.Image", "accelerate", "timm", "torchvision",
             "data_handle.embed", "models.autoencoder", "models.sit", "plot_utils", "utils",
             "train_ldm_only"]:
    stage(f"IMPORT START {name}")
    __import__(name)
    stage(f"IMPORT OK {name}")

import torch
from accelerate import Accelerator

stage(f"torch={torch.__version__}; CUDA runtime={torch.version.cuda}")
stage("ACCELERATOR INIT START")
accelerator = Accelerator(mixed_precision="bf16")
stage(f"ACCELERATOR INIT OK: device={accelerator.device}")
torch.cuda.set_device(accelerator.device)
stage("CUDA OP START")
x = torch.ones(16, device=accelerator.device)
torch.cuda.synchronize()
stage(f"CUDA OP OK: GPU={torch.cuda.get_device_name()}")
if torch.distributed.is_initialized():
    stage("ALLREDUCE START")
    torch.distributed.all_reduce(x)
    torch.cuda.synchronize()
    stage(f"ALLREDUCE OK: sum={x.sum().item()}")
stage("STARTUP DIAGNOSTIC PASSED")
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
