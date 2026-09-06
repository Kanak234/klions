#!/usr/bin/env python3
"""Generate a synthetic IDX dataset in the MNIST format.

The real MNIST files are a download away (see tools/fetch_mnist.sh), but the
examples must run on a machine with no network. This produces the same file
format and the same shapes, so examples/mnist.kl is exercised end to end.

Ten classes of 28x28 digits, each built from a class-specific stroke pattern
plus noise, so a small dense network can actually learn them.
"""
import struct, math, random, sys, os

def write_idx(path, dims, data, type_code=0x08):
    with open(path, "wb") as f:
        f.write(bytes([0, 0, type_code, len(dims)]))
        for d in dims:
            f.write(struct.pack(">I", d))
        f.write(bytes(data))

def render(cls, rng):
    """A 28x28 image with a class-dependent structure."""
    px = [0] * 784
    cx, cy = 14 + rng.randint(-2, 2), 14 + rng.randint(-2, 2)
    # Each class gets a distinct number of strokes at distinct angles.
    strokes = 1 + cls % 3
    for s in range(strokes):
        angle = (cls * 36 + s * 120 + rng.randint(-12, 12)) * math.pi / 180
        length = 8 + (cls % 4) * 2
        for t in range(-length, length + 1):
            x = int(cx + t * math.cos(angle))
            y = int(cy + t * math.sin(angle))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < 28 and 0 <= yy < 28:
                        px[yy * 28 + xx] = min(255, px[yy * 28 + xx] + 120)
    for i in range(784):
        px[i] = max(0, min(255, px[i] + rng.randint(-18, 18)))
    return px

def build(n, seed, out_dir, prefix):
    rng = random.Random(seed)
    images, labels = [], []
    for i in range(n):
        cls = i % 10
        images.extend(render(cls, rng))
        labels.append(cls)
    write_idx(os.path.join(out_dir, f"{prefix}-images.idx"), [n, 28, 28], images)
    write_idx(os.path.join(out_dir, f"{prefix}-labels.idx"), [n], labels)
    print(f"  {prefix}: {n} samples")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "examples/data"
    os.makedirs(out, exist_ok=True)
    print(f"writing synthetic IDX data to {out}/")
    build(3000, 1, out, "train")
    build(600, 2, out, "test")
