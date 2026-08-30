#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"
mkdir -p data/model/segmentors
pushd data/model/segmentors

echo "Downloading the following segmentor weights:"
if [[ "$@" == *"all"* ]] || [[ "$@" == *"cat_seg"* ]]; then
    echo " - CAT-Seg"
fi
if [[ "$@" == *"all"* ]] || [[ "$@" == *"mask2former"* ]]; then
    echo " - Mask2Former"
fi
if [[ "$@" == *"all"* ]] || [[ "$@" == *"openseed"* ]]; then
    echo " - OpenSeeD"
fi
if [[ "$@" == *"all"* ]] || [[ "$@" == *"cat_seg"* ]]; then
    if [[ ! -f cat_seg_large.pth ]]; then
        wget -progress=bar \
       https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_large.pth -O cat_seg_large.pth
    else
        echo "CAT-Seg model weights already exist. Skipping download."
    fi
fi

if [[ "$@" == *"all"* ]] || [[ "$@" == *"mask2former"* ]]; then
    if [[ ! -f mask2former.pkl ]]; then
        echo "Downloading Mask2Former model weights. This may take a while..."
        wget -nc --progress=bar \
        https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_tiny_bs16_50ep/model_final_9fd0ae.pkl -O mask2former.pkl
    else
        echo "Mask2Former model weights already exist. Skipping download."
    fi
fi
if [[ "$@" == *"all"* ]] || [[ "$@" == *"openseed"* ]]; then
    if [[ ! -f openseed.pth ]]; then
        echo "Downloading OpenSeeD model weights. This may take a while..."
        wget -nc --progress=bar \
       https://github.com/IDEA-Research/OpenSeeD/releases/download/openseed/model_state_dict_swint_51.2ap.pt -O openseed.pth
    else
        echo "OpenSeeD model weights already exist. Skipping download."
    fi
fi

echo "Segmentor weights downloaded successfully."