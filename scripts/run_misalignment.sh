#!/usr/bin/env bash
set -euo pipefail

# STEP 1
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 misalignment.py --compare=human,catseg --output_file_name=wordnet_merge_step1 --cluster_to_analyze=4

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_legacy_wordnet_refinement.py --mapping_file=data/cache/mapping/wordnet_merge_step1 --configuration_name=wordnet_step1 --cluster_to_analyze=4 --batch_parsing=False

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_legacy_wordnet_refinement.py --mapping_file=data/cache/mapping/wordnet_merge_step1 --segmentor=catseg --configuration_name=wordnet_step1 --cluster_to_analyze=4

# STEP 2
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 misalignment.py --compare=human,catseg --configuration_name=wordnet_step1 --mapping_file=data/cache/mapping/wordnet_merge_step1 --output_file_name=wordnet_merge_step2 --cluster_to_analyze=4 

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_legacy_wordnet_refinement.py --mapping_file=data/cache/mapping/wordnet_merge_step2 --configuration_name=wordnet_step2 --batch_parsing=False --cluster_to_analyze=4

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_legacy_wordnet_refinement.py --mapping_file=data/cache/mapping/wordnet_merge_step2 --configuration_name=wordnet_step2 --segmentor=catseg --cluster_to_analyze=4

# STEP 3
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 misalignment.py --compare=human,catseg --configuration_name=wordnet_step2 --mapping_file=data/cache/mapping/wordnet_merge_step2 --output_file_name=wordnet_merge_step3 --cluster_to_analyze=4

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_legacy_wordnet_refinement.py --mapping_file=data/cache/mapping/wordnet_merge_step3 --configuration_name=wordnet_step3 --batch_parsing=False --cluster_to_analyze=4

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_legacy_wordnet_refinement.py --mapping_file=data/cache/mapping/wordnet_merge_step3 --configuration_name=wordnet_step3 --segmentor=catseg --cluster_to_analyze=4


