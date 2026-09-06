#!/usr/bin/env bash
# Download the real MNIST dataset and convert it into the layout the examples
# expect. Requires curl and gunzip.
#
# The examples ship with synthetic data (tools/make_dataset.py) so they run
# offline. Run this to swap in the real thing — nothing in mnist.kl changes.

set -euo pipefail

DEST="${1:-examples/data}"
mkdir -p "$DEST"

BASE="https://ossci-datasets.s3.amazonaws.com/mnist"

fetch() {
    local remote="$1" local_name="$2"
    if [ -f "$DEST/$local_name" ]; then
        echo "  $local_name already present"
        return
    fi
    echo "  fetching $remote"
    curl -fsSL "$BASE/$remote" -o "$DEST/$remote"
    gunzip -c "$DEST/$remote" > "$DEST/$local_name"
    rm -f "$DEST/$remote"
}

echo "downloading MNIST into $DEST/"
fetch train-images-idx3-ubyte.gz train-images.idx
fetch train-labels-idx1-ubyte.gz train-labels.idx
fetch t10k-images-idx3-ubyte.gz  test-images.idx
fetch t10k-labels-idx1-ubyte.gz  test-labels.idx

echo
echo "done — 60000 training and 10000 test samples"
echo "run: klions run examples/mnist.kl"
