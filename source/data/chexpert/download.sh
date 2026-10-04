#!/usr/bin/env bash
# Download the CheXpert inputs used by the real-data experiments.
set -eu

: "${CHEXLOCALIZE_DOWNLOAD_URL:?Set this to the AIMI CheXlocalize container download URL}"
DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$DATA_DIR/raw"
URL="${CHEXLOCALIZE_DOWNLOAD_URL%%\?*}/CheXpert/test?${CHEXLOCALIZE_DOWNLOAD_URL#*\?}"

mkdir -p "$RAW_DIR/CheXpert"
azcopy copy "$URL" "$RAW_DIR/CheXpert" --recursive --overwrite=ifSourceNewer
[ -d "$RAW_DIR/cheXpert-test-set-labels" ] || \
  git clone --depth 1 https://github.com/rajpurkarlab/cheXpert-test-set-labels.git "$RAW_DIR/cheXpert-test-set-labels"
