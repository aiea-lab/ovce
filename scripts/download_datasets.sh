#!/usr/bin/env bash
set -e

export DETECTRON2_DATASETS=$PWD/data/datasets
# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

mkdir -p data/datasets

# Check if the flag --ade20k is passed as an argument
if [[ "$@" == *"all"* ]] || [[ "$@" == *"ade20k-150"* ]]; then
    pushd data/datasets
    # Unzip the dataset if it hasn't been unzipped yet
    if [ ! -d ADEChallengeData2016 ]; then
        echo "Downloading ADE20K-150 dataset..."
        wget -nc --progress=bar \
        http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
        # Unzip the dataset and remove the zip file to save space
        echo "Unzipping ADE20K-150 dataset..."
        unzip -q ADEChallengeData2016.zip
        rm ADEChallengeData2016.zip
    fi
    pushd ../../
    echo "Preparing Ade20k-150 dataset..."
    python datasets/prepare_datasets/prepare_ade20k_150.py
fi

if [[ "$@" == *"all"* ]] || [[ "$@" == *"ade20k-full"* ]]; then
    pushd data/datasets
    if [ ! -d ADE20K_2021_17_01 ]; then
        echo "Downloading ADE20K-150-full dataset from Kaggle..."
        wget -nc --progress=bar \
            https://www.kaggle.com/api/v1/datasets/download/kallurivasanthsai/ade20k-2021-17-01 -O ade20k-2021-17-01.zip
        echo "Unzipping ADE20K-150-full dataset from Kaggle..."
        unzip -q ade20k-2021-17-01.zip
        rm ade20k-2021-17-01.zip
    fi
    pushd ../../
    echo "Preparing Ade20k-full dataset..."
    python datasets/prepare_datasets/prepare_ade20k_full.py
fi

if [[ "$@" == *"all"* ]] || [[ "$@" == *"coco-stuff"* ]]; then
    pushd data/datasets
    # COCO-Stuff dataset
    if [ ! -d coco-stuff/images/train2017 ] || [ ! -d coco-stuff/images/val2017 ]; then
        mkdir -p coco-stuff/images
        pushd coco-stuff/images
        echo "Downloading COCO-Stuff dataset..."
        wget -nc --progress=bar \
            http://images.cocodataset.org/zips/train2017.zip
        wget -nc --progress=bar \
            http://images.cocodataset.org/zips/val2017.zip
        unzip -q train2017.zip
        unzip -q val2017.zip
        rm train2017.zip
        rm val2017.zip
        pushd ../../
    fi
    if [ ! -d coco-stuff/annotations/train2017 ] || [ ! -d coco-stuff/annotations/val2017 ]; then
        mkdir -p coco-stuff/annotations
        pushd coco-stuff/annotations
        wget -nc --progress=bar \
            http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
        unzip -q stuffthingmaps_trainval2017.zip
        rm stuffthingmaps_trainval2017.zip
        pushd ../../
    fi
    pushd ../../
    echo "Preparing COCO-Stuff dataset..."
    python datasets/prepare_datasets/prepare_coco_stuff.py
fi

# PASCAL VOC 2012
if [[ "$@" == *"all"* ]] || [[ "$@" == *"pascal-voc"* ]]; then
    pushd data/datasets
    if [ ! -d VOCdevkit ]; then
        echo "Downloading PASCAL VOC 2012 dataset..."
        wget -nc --progress=bar \
            http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
        tar -xf VOCtrainval_11-May-2012.tar
        rm VOCtrainval_11-May-2012.tar
        pushd VOCdevkit/VOC2012/
        wget -nc --progress=bar \
            https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip
        unzip -q SegmentationClassAug.zip
        rm SegmentationClassAug.zip
        pushd ../
    fi
    pushd ../../
    echo "Preparing PASCAL VOC 2012 dataset..."
    python datasets/prepare_datasets/prepare_voc.py
fi

# CUB-200-2011
if [[ "$@" == *"all"* ]] || [[ "$@" == *"cub200"* ]]; then
    pushd data/datasets
    if [ ! -d CUB_200_2011 ]; then
        echo "Downloading CUB-200-2011 dataset..."
        wget -nc --progress=bar \
            https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
        tar -xf CUB_200_2011.tgz
        rm CUB_200_2011.tgz
        rm attributes.txt
    fi
    pushd ../../
    echo "Preparing CUB-200-2011 dataset..."
    python datasets/prepare_datasets/prepare_cub200.py
fi



if [[ "$@" == *"pascal-context"* ]]; then
    pushd data/datasets
    if [ ! -d VOCdevkit/VOC2010 ]; then
        mkdir -p VOCdevkit
        echo "Downloading PASCAL VOC 2010 dataset..."
        wget -nc --progress=bar \
            http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
        tar -xf VOCtrainval_03-May-2010.tar
        rm VOCtrainval_03-May-2010.tar
        pushd VOCdevkit/VOC2010
        wget -nc --progress=bar \
            https://roozbehm.info/pascal-context/trainval.tar.gz
        tar -xzf trainval.tar.gz
        rm trainval.tar.gz
        pushd ../../
    fi
    pushd ../../
fi

echo "done"