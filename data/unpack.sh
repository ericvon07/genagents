#!/usr/bin/env bash
# Extract the built populations into data/populations/, where the readers in analysis/ look
# for them when LAB_DATA is not set.
set -euo pipefail
cd "$(dirname "$0")"
for f in gss22-*.tar.gz; do
  tar -xzf "$f"
  echo "unpacked ${f%.tar.gz}"
done
