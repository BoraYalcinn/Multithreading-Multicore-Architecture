# PC40: Parallel Computing, Fall 2026 (A26)
#!/usr/bin/env python3
"""Download and decompress the four standard MNIST IDX files using only stdlib."""
from pathlib import Path
from urllib.request import urlretrieve
import gzip, shutil

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]
HERE = Path(__file__).resolve().parent

for name in FILES:
    gz = HERE / name
    raw = HERE / name[:-3]
    if raw.exists():
        print(f"OK: {raw.name}")
        continue
    print(f"Downloading {name} ...")
    urlretrieve(BASE + name, gz)
    print(f"Decompressing {name} ...")
    with gzip.open(gz, "rb") as src, raw.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    gz.unlink()
print("MNIST is ready.")
