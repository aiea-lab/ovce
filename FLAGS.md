# COMMON AND SPECIFIC FLAGS DESCRIPTION
This file contains a description of the common and specific flags used in all the script of the repository. The flags are grouped by script and by type (common or specific). The common flags are used in all the scripts, while the specific flags are used only in the specific script.

## Common Flags (defined in `src/common_flags.py`)
- `--segmentor`: The name of the segmentor to be used. Currently supported segmentors are: `catseg`, `masqclip`, `sed`, `scan`, `openseed`, `mask2former`, and `human`. Default is `human`.
- `--configuration_name`: The name of the configuration to be used. This flag is used to specify a custom name for the current experiments and it is used to create a subdirectory in the `results` directory where the results of the current experiments will be saved. Default is `std`.
- `--device`: The device to be used for the experiments. The device should be one `cpu` or `cuda`. Default is `cuda`.
- `--dataset`: The name of the dataset to be used. The dataset should be one of the datasets defined in `datasets/DATASETS.md`. The dataset should be preprocessed and ready to use. The dataset should be located in the `$DETECTRON2_DATASETS` directory. Currently supported datasets are: `ade20k_150_test_sem_seg`, `cub200`, `cityscapes_fine_sem_seg_val`, `ade20k_full_sem_seg_freq_val_all`, `mapillary_vistas_sem_seg_val`, `context_459_test_sem_seg`, `coco_2017_test_stuff_all_sem_seg`, `voc_2012_test_sem_seg`. Default is `ade20k_150_test_sem_seg`.
- `--custom_classes`: A list of custom classes to be used for the segmentor. If not provided, the default classes for the dataset or the segmentor will be used. The custom classes should be a comma-separated list of class names. For example: `--custom_classes=cat,dog,car`. Default is `None`.
- `--predefined_concept_set`: The name of the predefined concept set to be used for the segmentor. This flag is used to specify a predefined set of concepts for the given dataset. Currently supported predefined concepts set are only for CUB200 and they are one of `granularity_0`, `granularity_1`, `granularity_2`, `all`.  Default is `None`, meaning that the default classes for the dataset or the segmentor will be used.
- `--ignore_concepts`: A list of concepts to be ignored when computing the explanations. The ignored concepts should be a comma-separated list of class names. For example: `--ignore_concepts=cat,dog,car`. Default is `[]`.


The following flags are used to specify the probed model and the layer to be probed:
- `--model`: The name of the probed model to be used. The probed model should be pre-trained and ready to use. The probed model should be located in the `data/model/other` directory or in the `data/model/zoo` directory. Currently supported models are: `resnet18`, `resnet_cub200`, `alexnet`, `densenet161`, `cvt`, `maxvit`, `convnext`, and `efficientvit`. Default is `resnet18`.
- `--layer`: The name of the layer to be probed. For `cvt`, `maxvit`, `convnext`, and `efficientvit`, only the `stage3` layer is supported. Default is `layer4`.
- `--random_units`: The number of random units to be probed. The random units are selected from the specified layer of the probed model. Default is 0, meaning that all the units of the specified layer will be probed. 

The following flags are used to control the behavior of the explanation algorithm:
- `--length`: Maximum explanation length. This flag is used to limit the maximum number of concepts in the explanation. Default is `3`.
- `--num_clusters`: The number of clusters to be used for the clustering of the activations. Default is `5`. Setting it to `1` means computing Network Dissection explanations.
- `--beam_limit`: The beam limit to be used for the beam search algorithm. This flag is used to limit the number of concepts to be considered at each step of the beam search algorithm. It is considered only when the number of clusters is `1`.Default is `0.01`.


The following flags arerelevant for `run_fast.py`, `run_legacy.py`, and `run_wordnet_refinement.py` and they are used to control the generation of the segmentation masks and the computation of the concept masks.

- `--batch_parsing`: Whether to load all the segmentations in memory at once or to load and parse them batch by batch in order to compute concept masks. This flag is used to reduce the memory usage when computing concept masks for large datasets. Default is `True`, meaning that the default choice is the slower but more memory efficient approach.
- `--parallel_concepts`: Number of concepts to be processed in parallel when computing concept masks. This flag is used to speed up the computation of concept masks when using a large number of concepts. Default is `50`.

Finally, the following flags are used to set the directories where the results of the experiments, the segmentation masks, the activations of the probed model, and the probed models are saved or loaded from:
- `--root_results`: The root directory where the results of the experiments will be saved. Default is `data/results`.
- `--root_segmentation`: The root directory where the segmentation masks will be saved. Default is `data/cache/segmentations`.
- `--root_activations`: The root directory where the activations of the probed model will be saved. Default is `data/cache/activations`.
- `--root_models`: The root directory where the probed models are be saved. Default is `data/model`.

## Script-Specific Flags
The following flags are specific to the `misalignment.py` script and they are used to control the computation of the mapping between the concepts of two segmentors and the WordNet concepts:
- `--root_mapping`: The root directory where the mapping between the concepts of two segmentors and the WordNet concepts will be saved. Default is `data/mapping`.
- `--output_file_name`: The name of the output file where the mapping between the concepts of two segmentors and the WordNet concepts will be saved. Default is `merging_results.json`.
- `--mapping_file`: The path to the mapping file to be used for the computation of the mapping between the concepts of two segmentors and the WordNet concepts. This flag is used to specify a pre-computed mapping file to be used for the computation of the mapping between the concepts of two segmentors and the WordNet concepts. Default is `None`, meaning that the mapping will be computed from scratch.  
- `--compare`: The names of the two segmentors to be compared. This flag is used to specify the two segmentors for which the mapping between the concepts and the WordNet concepts will be computed. The two segmentors should be separated by a comma. For example: `--compare=human,catseg`. Default is `human,catseg`.
- `--cluster_to_analyze`: The index of the cluster to be analyzed. This flag is used to specify the cluster for which the mapping between the concepts of two segmentors and the WordNet concepts will be computed. Default is `0`, meaning that the first cluster will be analyzed.

The flag `--compare` is also used by the `analyze_diff.py` script to analyze the differences between the explanations computed by two segmentors. Optionally, you can also use `--cluster_to_analyze`, and `--mapping_file` if the configurations you want to compare use mapping files to define their concept sets (e.g., after unification through WordNet).

The flags `--cluster_to_analyze`, and `--mapping_file` are also used in `run_wordnet_refinement.py` to compute explanations for a specific cluster and to use a pre-computed mapping file to define the concept set for the segmentor.

