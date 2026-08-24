# Open Vocabulary Compositional Explanations for Neuron Alignment

This repository contains the official implementation for the paper:

"Open Vocabulary Compositional Explanations for Neuron Alignment" by Biagio La Rosa and Leilani H. Gilpin
Transactions on Machine Learning Research (TMLR), 2026



This work introduces an open-vocabulary framework for explaining individual neurons in vision models using compositional explanations derived from segmentation models and a knowledge-graph process to study misalignment between explanations computed by different pipelines.

## Main Scripts and Command-Line Interface

### Command-Line Arguments
Most of the main scripts share most of the command-line arguments, which can be inspected in the file `src/common_flags.py`. The main arguments include:
- `--segmentor`: the segmentor to use for computing explanations. Supported segmentors include: `human`, `catseg`, `mask2former`, `masqclip`, `scan`, `sed`, and `openseed`.
- `--dataset`: the dataset to use for computing explanations. Supported datasets include: ``ade20k_150_test_sem_seg`, `cub200`, `cityscapes_fine_sem_seg_val`,  `ade20k_full_sem_seg_freq_val_all`, `mapillary_vistas_sem_seg_val`, `context_459_test_sem_seg`, `coco_2017_test_stuff_all_sem_seg`, `voc_2012_test_sem_seg`,  and `cub200`.  
- `--model`: the probed model for which to compute explanations. Supported models include: `resnet18`, `resnet_cub200`, `alexnet`, `densenet161`, `cvt`, `maxvit`, `convnext`, and `efficientvit`.
- `--layer`: the layer of the probed model for which to compute explanations. 
- `--custom_classes`: a list of custom classes to use for computing explanations. If not provided, the default classes for the segmentor will be used.
- `--configuration_name`: a name for the configuration to use for computing explanations. This is integrated and used to organize the results of the experiments.

To see the full list of command-line arguments and their descriptions, please refer to the [FLAGS.md](FLAGS.md) file. 

### Main Functionality
The main functionality of the repository is to compute explanations for individual neurons in vision models using segmentation models and a knowledge-graph process to study misalignment between explanations computed by different pipelines. The main scripts for this functionality:

- `run_fast.py` and `run_legacy.py`, which implement the compositional explanation algorithm used in the paper. The `run_fast.py` script is generally faster and uses the `compositional_explanations` pip package, while `run_legacy.py` is the legacy implementation that implements the algorithm used in the paper. Example usage:
```
python run_legacy.py --segmentor=catseg --dataset=ade20k_150_test_sem_seg --model=resnet18 --layer=layer4 --random_units=50
```
- `evaluate.py`, which evaluates the results of the experiments and computes metrics for compositional explanations. Example usage:
```
python evaluate.py --segmentor=catseg --dataset=ade20k_150_test_sem_seg --model=resnet18 --layer=layer4 --random_units=50
```
- `analyze_diff.py`, which analyzes the differences between explanations computed by different pipelines/segmentors and produces mappings between segmentor labels and WordNet concepts. Example:
```
python analyze_diff.py --compare=human,catseg --cluster_to_analyze=4
```
- `misalignment.py`, which analyzes the misalignment between segmentor labels and WordNet concepts and produces a mapping that aims to unify these differences through the use of WordNet concepts. Example usage:
```
python misalignment.py --segmentor=catseg --dataset=ade20k_150_test_sem_seg --model=resnet18 --layer=layer4 --random_units=50 --mapping_file=human_catseg_mapping_step1.json --output_file_name=human_catseg_mapping_step2.json
```

## Quick Start

### Setup the Environment
#### Option 1: Use Docker (Strongly Recommended)
Move to the Docker directory and build the Docker image using the following command:
```bash
docker build -t opence .
```

Then run the container and execute the repository scripts inside it. For example:
```bash
docker run --runtime=nvidia -it --rm -v <PARENT_DIR>:/workspace/ --net=host--ipc=host opence```
```

#### Option 2: Install dependencies manually

Because the repository relies on multiple segmentors, it is recommended to use the provided Docker setup. However, if you prefer to install dependencies manually, below you can find a basic list of the required dependencies. Please note that this list may not be exhaustive, and additional dependencies may be required depending on the specific segmentors and datasets you plan to use.

```bash
pip install torch torchvision
pip install git+https://github.com/facebookresearch/detectron2.git
pip install -U openmim
mim install mmcv-full==1.6.2
pip install mmsegmentation==0.27.0
pip install git+https://github.com/openai/CLIP.git
pip install open_clip_torch
pip install transformers
pip install compositional_explanations
pip install einops
pip install wn
pip install timm
```

### Downloading Resources
The repository includes scripts to download pretrained models, segmentor weights, and datasets. See the [Resources](RESOURCES.md) file for instructions on how to download and prepare the necessary resources.

### Set the Environment Variables
The repository requires the `DETECTRON2_DATASETS` environment variable to be set to the path where the datasets are stored. For example, if you have downloaded the datasets to `/path/to/datasets`, you can set the environment variable as follows:
```bash
export DETECTRON2_DATASETS=/path/to/datasets
```

### Example: compute explanations for a segmentor on ADE20K

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_fast.py \
  --segmentor=catseg \
  --dataset=ade20k_150_test_sem_seg \
  --model=resnet18 \
  --layer=layer4 \
  --random_units=50
```

Then evaluate the same run:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
  --segmentor=catseg \
  --dataset=ade20k_150_test_sem_seg \
  --model=resnet18 \
  --layer=layer4 \
  --random_units=50
```

## Supported Models and Segmentors

### Models

The project supports backbone models such as:

- `resnet18`
- `resnet_cub200`
- `alexnet`
- `densenet161`
- `cvt`
- `maxvit`
- `convnext`
- `efficientvit`

### Segmentors

Supported segmentors include:

- `human`
- `catseg`
- `mask2former`
- `masqclip`
- `scan`
- `sed`
- `openseed`

The implementations of these segmentors are taken from the following official repositories:
- [CATSeg](
  https://github.com/cvlab-kaist/CAT-Seg)
- [Mask2Former](
  https://github.com/facebookresearch/Mask2Former)
- [MaskQCLIP](
  https://github.com/mlpc-ucsd/MasQCLIP)
- [SCAN](
  https://github.com/yongliu20/SCAN)
- [SED](
  https://github.com/xb534/SED/tree/main)
- [OpenSeeD](
  https://github.com/idea-research/openseed)

### Datasets
Supported datasets include:
- `ade20k_150_test_sem_seg`
- `cub200`
- `cityscapes_fine_sem_seg_val`
- `ade20k_full_sem_seg_freq_val_all`
- `mapillary_vistas_sem_seg_val`
- `context_459_test_sem_seg`
- `coco_2017_test_stuff_all_sem_seg`
- `voc_2012_test_sem_seg`

## Add your own segmentor/dataset/model
To add your own segmentor, dataset, or model, please refer to the [EXTENDING.md](EXTENDING.md) file for instructions on how to extend the repository with new components.

## Reproducing the Paper Results

The paper-specific reproduction commands are documented in [PAPER.md](PAPER.md). 

## Citation

If you use this code in your research, please cite the paper associated with this repository:

@article{ <br>
anonymous2026open, <br>
title={Open Vocabulary Compositional Explanations for Neuron Alignment}, <br>
author={Biagio {La Rosa} and Leilani H. Gilpin}, <br>
journal={Transactions on Machine Learning Research}, <br>
year={2026}, <br>
url={https://openreview.net/forum?id=iS38vzTMdd}, <br>
}

