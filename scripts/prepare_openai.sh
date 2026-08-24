# Script to install the open CLIP model required by SED
#!/usr/bin/env bash
set -e

pushd segmentors/open_clip
make install

echo "Open CLIP model installed successfully."