"""Copy the verified, matching-release reference driver into diagnostic storage."""
import hashlib
from pathlib import Path
import shutil
import socket

source = Path("/usr/lib/x86_64-linux-gnu/libcuda.so.595.91.07")
expected = "4839b5da17cd8f58a8e9c57e97c9b61f3c41be7934479f48afc983416b2492c3"
print("REFERENCE HOST", socket.gethostname(), flush=True)
digest = hashlib.sha256(source.read_bytes()).hexdigest()
print("SOURCE SHA256", digest, flush=True)
assert digest == expected, "Reference driver changed; refusing to copy"
directory = Path("checkpoints/diagnostics/reference_driver_595.91.07")
directory.mkdir(exist_ok=True)
target = directory / source.name
if not target.exists():
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst)
assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
for name in ["libcuda.so.1", "libcuda.so"]:
    link = directory / name
    if not link.exists():
        link.symlink_to(source.name)
print("REFERENCE DIRECTORY", directory.resolve(), flush=True)
with target.open("rb") as stream:
    stream.seek(0x2b8d00)
    print("REFERENCE CRASH REGION", stream.read(112).hex(), flush=True)
print("REFERENCE DRIVER READY", flush=True)
