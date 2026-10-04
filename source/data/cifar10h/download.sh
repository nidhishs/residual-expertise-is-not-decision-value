#!/usr/bin/env bash
# Download the CIFAR-10 test batch and CIFAR-10H annotator labels.
set -eu

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$DATA_DIR/raw/cifar10h"
mkdir -p "$RAW_DIR"

curl -Lf https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | \
  tar -xz -C "$RAW_DIR" cifar-10-batches-py/test_batch
curl -Lf https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-raw.zip \
  -o "$RAW_DIR/cifar10h-raw.zip"
unzip -p "$RAW_DIR/cifar10h-raw.zip" > "$RAW_DIR/cifar10h-raw.csv"
rm "$RAW_DIR/cifar10h-raw.zip"
