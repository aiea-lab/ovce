
# Reproducing Results

This document provides instructions for reproducing the results in the paper.

**Note**: Almost all the segmentors are not compatible with reproducible settings (i.e., they use non-deterministic operations). 
The noise introduced by these operations is negligible and does not affect the results significantly since usually it involves few pixels per concepts. However, it is possible that the results may vary slightly across runs.


## Results

First, set up the environment as described in the [README.md](README.md) file. Then, to reproduce the results in the paper, run the following commands.

### Compute Results in Table 1, 4, 6, 7, 8, 9, 10, 11, 15
To reproduce the results in Table 1, 4, 6, 7, 8, 9, 10, 11, 15 run the following command:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_legacy.py --segmentor=<SEGMENTOR_NAME> --dataset=<DATASET_NAME> --random_units=50
```
where `<SEGMENTOR_NAME>` is one of the following: `catseg`, `masqclip`, `sed`, `scan`, `openseed`, `mask2former`, `human`, and `<DATASET_NAME>` is one of the following: `ade20k_150_test_sem_seg`, `cub200`,
`cityscapes_fine_sem_seg_val`, `ade20k_full_sem_seg_freq_val_all`,
`mapillary_vistas_sem_seg_val`,
`context_459_test_sem_seg`,
`coco_2017_test_stuff_all_sem_seg`,
`voc_2012_test_sem_seg`

and then to evaluate the results, run the following command using the same exact arguments:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py --segmentor=<SEGMENTOR_NAME> --dataset=<DATASET_NAME> --random_units=50
```

### Compute Results in Table 2 and 5 (CUB)
To reproduce the results in Table 2 and 5, run the following command for the results related to `mask2former` and `human` segmentors:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_legacy.py --segmentor=<SEGMENTOR_NAME> --random_units=50 --dataset=cub200 --model=resnet_cub200 --layer=features
```



where `<SEGMENTOR_NAME>` is either `mask2former` or `human`.

TO reproduce the results in Table 2 and 5 for the other segmentors, run the following command:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_legacy.py --segmentor=<SEGMENTOR_NAME> --random_units=50 --dataset=cub200 --model=resnet_cub200 --layer=features --predefined_concept_set=all --configuration_name=<CONFIGURATION_NAME>
```


 where `<SEGMENTOR_NAME>` is one of the following: `catseg`, `masqclip`, `sed`, `scan`, `openseed`, and `<CONFIGURATION_NAME>` is of your choice.
 
 Then to evaluate the results, run the following command using the same exact arguments used before:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py
```


### Compute Results in Table 12 and 13 (Ade20k)
To reproduce the results in Table 12 and 13, run the following command:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_legacy.py --segmentor=<SEGMENTOR_NAME> --random_units=50 --dataset=ade20k_150_test_sem_seg --model=<MODEL_NAME> --layer=features
```
where `<SEGMENTOR_NAME>` is one of the following: `catseg`, `masqclip`, `sed`, `scan`, `openseed`, `mask2former`, `human`, and `<MODEL_NAME>` is either `densenet161` or `alexnet`.

Then, to evaluate the results, run the following command using the same exact arguments:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py --segmentor=<SEGMENTOR_NAME> --random_units=50 --dataset=ade20k_150_test_sem_seg --model=<MODEL_NAME> --layer=features
```

### Compute Results in Table 12 and 13 (Ade20k)
To reproduce the results in Table 12 and 13, run the following command:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_legacy.py --segmentor=<SEGMENTOR_NAME> --random_units=50 --dataset=ade20k_150_test_sem_seg --model=<MODEL_NAME> --layer=features
```
where `<SEGMENTOR_NAME>` is one of the following: `catseg`, `mask2former`, `human`, and `<MODEL_NAME>` is either `densenet161` or `alexnet`.

Then, to evaluate the results, run the following command using the same exact arguments:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py --segmentor=<SEGMENTOR_NAME> --random_units=50 --dataset=ade20k_150_test_sem_seg --model=<MODEL_NAME> --layer=features
```

### Compute Results in Table 14 (Ade20k)
To reproduce the results in Table 14, run the following command:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_legacy.py --segmentor=<SEGMENTOR_NAME> --random_units=50 --dataset=ade20k_150_test_sem_seg --model=<MODEL_NAME> --layer=stage3 --num_clusters=1 --quantile=0.01
```
where `<SEGMENTOR_NAME>` is one of the following: `catseg`, `mask2former`, `human`, and `<MODEL_NAME>` is one of the following: `cvt`, `efficientvit`, `maxvit`, `convnext`.

Then, to evaluate the results, run the following command using the same exact arguments:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py --segmentor=<SEGMENTOR_NAME> --random_units=50 --dataset=ade20k_150_test_sem_seg --model=<MODEL_NAME> --layer=stage3 --num_clusters=1 --quantile=0.01
```

### COMPUTE WORDNET ANALYSIS

To reproduce the analysis of the misalignment between segmentors and WordNet concepts (Section 5.3), run the following script:
```bash
bash scripts/run_misalignment.sh
```

Then, analyze the results by running the following commands:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 analyze_diff.py --compare=human,catseg --cluster_to_analyze=4 
```
and
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 analyze_diff.py --compare=human,catseg --cluster_to_analyze=4 --configuration_name=wordnet_step2 --mapping_file=data/cache/mapping/wordnet_merge_step2
```

### Compute Results in Figure 6

To compute the values of the cells in Figure 6, first compute the explanations for each of the segmentor you would like to compare by running the following command:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_legacy.py --segmentor=<SEGMENTOR_NAME> --dataset=ade20k_150_test_sem_seg 
```
where `<SEGMENTOR_NAME>` is one of the following: `catseg`, `masqclip`, `sed`, `scan`, and `openseed`.

Note that this command is different from the one for figure 1, since here we compute the explanations for all the units in the network.

Then, to analyze the results, run the following command:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 analyze_diff.py --compare=<Segmentor1>,<Segmentor2> --cluster_to_analyze=4 
```

where `<Segmentor1>` and `<Segmentor2>` are the names of the segmentors you want to compare. 
