# Script to compile mask2former ops. Run this script before using the mask2former, masqclip, and SED segmentors.

#!/usr/bin/env bash
set -e

pushd segmentors/mask2former/modeling/pixel_decoder/ops
sh make.sh

echo "Mask2Former compiled successfully."