"""Script to run the clustering algorithm for compositional explanations.
"""
from collections import defaultdict
import os
import logging
import pickle
from tqdm import tqdm

import torch
import absl.flags
import absl.app
import numpy as np

from utils import common_utils, activations_utils, results_utils, dataset_utils, segmentor_utils, concept_mask_utils
from src import common_flags # import all user flags
from src import settings


# Disable overly verbose logging of detectron2
logger = logging.getLogger("detectron2")
logger.setLevel(logging.WARNING)

FLAGS = absl.flags.FLAGS

def main(argv):
    if FLAGS.num_clusters < 1:
        raise ValueError("num_clusters must be greater than 0")

    # Set seed
    common_utils.set_seed(FLAGS.seed)

    #  Parameters
    cfg = settings.Settings(
        model= FLAGS.model, 
        dataset=FLAGS.dataset,
        segmentor=FLAGS.segmentor,
        layer=FLAGS.layer,
        device=FLAGS.device,
        batch_parsing=FLAGS.batch_parsing,
        parallel_concepts=FLAGS.parallel_concepts,
        root_models=FLAGS.root_models,
        root_segmentations=FLAGS.root_segmentations,
        root_activations=FLAGS.root_activations,
        root_results=FLAGS.root_results,
        configuration_name=FLAGS.configuration_name,
    )

    # Load Activations
    layer_activations = activations_utils.get_layer_activations(cfg)

    # Load Concept Set (optionally specified by the user)
    if FLAGS.predefined_concept_set is not None:
        assert FLAGS.predefined_concept_set in ['granularity_0', 'granularity_1', 'granularity_2', 'all'], "Predefined concept set must be one of ['granularity_0', 'granularity_1', 'granularity_2', 'all']"
        assert FLAGS.configuration_name != "std", "Predefined concept set can only be used with a custom configuration name"
        concept_set = dataset_utils.get_class_names(dataset_name=cfg.get_dataset_name(), custom_classes=FLAGS.predefined_concept_set)
        print(f"Using predefined concept set: {FLAGS.predefined_concept_set} with concepts: {concept_set}")
        ignore_concepts = ['other', 'background']
    elif FLAGS.custom_classes is not None:
        assert FLAGS.configuration_name != "std", "Custom classes can only be used with a custom configuration name"
        concept_set = FLAGS.custom_classes
        ignore_concepts = FLAGS.ignore_concepts
    else:
        concept_set = None
        ignore_concepts = []

    # Load Segmentation Masks
    masks, masks_labels, _ = segmentor_utils.get_segmentor_outputs(cfg, concept_set=concept_set)
    masks = concept_mask_utils.remove_unmeaningful_masks(concept_masks=masks, concept_labels=masks_labels, ignore_concepts=ignore_concepts)

    # Select Units
    selected_units = activations_utils.extract_random_units(layer_activations, random_units=FLAGS.random_units)
    print(f"Selected Units: {selected_units}")

    results = defaultdict(lambda: defaultdict(list))
    
    # Compute explanations
    for unit in tqdm(selected_units, desc="Computing explanation metrics for units"):
        # Split the activations of the unit into clusters
        unit_activations = layer_activations[:, unit, :, :]
        activation_ranges = activations_utils.compute_activation_ranges(
                    unit_activations, num_clusters=FLAGS.num_clusters, quantile=FLAGS.quantile)

        # Load explanations for each cluster of the unit
        for cluster_index, activation_range in enumerate(
                    sorted(activation_ranges)
                ):
            bitmaps = activations_utils.compute_bitmaps(
                unit_activations,
                activation_range,
                mask_shape=cfg.get_mask_shape(),
            )
            bitmaps = bitmaps.to(cfg.get_device())

            # Get the directory and file path for the results of the current unit and cluster  
            dir_results = results_utils.get_result_dir(cfg, unit=unit, activation_range=activation_range)
            file_results = results_utils.get_result_file(dir_results, cluster_index=cluster_index, num_clusters=FLAGS.num_clusters, quantile=FLAGS.quantile)

            # Load the explanations
            if os.path.exists(file_results):
                with open(file_results, "rb") as f:
                    explanation, _, _ = pickle.load(f)
            else:
                raise Exception(f"Results file {file_results} does not exist. Please run the explanation computation script first to generate the results.")

            # Collect metrics
            metrics_dict = common_utils.compute_scores(formula=explanation, masks=masks, activations=bitmaps)
            for metric in metrics_dict.keys():
                value = metrics_dict[metric]
                results[str(cluster_index)][metric].append(value)

    del masks

    # Print Results
    for index_cluster in range(FLAGS.num_clusters):
        print(f"Segmentor: {FLAGS.segmentor} Cluster: {index_cluster}")
        for metric in results[str(index_cluster)].keys():
            mean_metric = np.mean(results[str(index_cluster)][metric])
            std_dev_metric = np.std(results[str(index_cluster)][metric])
            print(f"{metric} Mean: {mean_metric:.3f} Std Dev: {std_dev_metric:.3f}")
        print("\n")


if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)
