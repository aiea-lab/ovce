"""Script to run the clustering algorithm for compositional explanations.
"""
import os
import logging
import pickle
import json

import torch
import absl.flags
import absl.app

from utils import common_utils, activations_utils, dataset_utils, segmentor_utils, concept_mask_utils, results_utils
from src import common_flags # import all user flags
from src import formula as F
from src import settings
from legacy import algorithms, heuristic_info

# Disable overly verbose logging of detectron2
logger = logging.getLogger("detectron2")
logger.setLevel(logging.WARNING)

absl.flags.DEFINE_string("mapping_file", None, "Path to the mapping file")
absl.flags.mark_flag_as_required("mapping_file")
absl.flags.DEFINE_list("cluster_to_analyze", None, "Cluster to analyze")

FLAGS = absl.flags.FLAGS

def main(argv):
    if FLAGS.num_clusters < 1:
        raise ValueError("num_clusters must be greater than 0")

    if FLAGS.configuration_name == "std":
        raise ValueError("The 'std' configuration name is reserved for the standard configuration. Please choose a different name for your custom configuration.")
    
    # Set seed
    common_utils.set_seed(FLAGS.seed)

    #  # Parameters
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

    # Load Mapping
    with open(FLAGS.mapping_file, 'r') as f:
        mapping = json.load(f)


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
    if cfg.get_segmentor_name() == "human":
        # In this case we can merge the masks after loading following the mapping
        masks, masks_labels, masks_are_regenerated = segmentor_utils.get_segmentor_outputs(cfg, concept_set=concept_set)
        masks = concept_mask_utils.remove_unmeaningful_masks(concept_masks=masks, concept_labels=masks_labels, ignore_concepts=ignore_concepts)
        masks, masks_labels = concept_mask_utils.merge_human_masks_with_mapping(masks, masks_labels, mapping=mapping)
        masks_are_regenerated = True  # Set to True since we have modified the masks
    elif cfg.get_segmentor_name() == "mask2former":
        raise ValueError("Mask2Former segmentor is a closed-world approach and cannot support modifications in the concept set.")
    else:
        # In this case we need to modify the concept set before loading the masks
        labels_before_merging = segmentor_utils.get_labels_config(cfg, concept_set=concept_set)
        concept_set = common_utils.refine_concept_set(labels_before_merging, mapping)
        masks, masks_labels, masks_are_regenerated = segmentor_utils.get_segmentor_outputs(cfg, concept_set=concept_set)

    print(f"Loaded {len(masks_labels)} masks")

    cache_info_dir = cfg.get_info_dir()
    if masks_are_regenerated:
        # We need to recompute the informations
        common_utils.clean_directory(cache_info_dir)

    # Load Mask Info
    masks_info = heuristic_info.get_masks_info(masks, info_directory=cache_info_dir,
             mask_shape=cfg.get_mask_shape(), device=cfg.get_device())  

    # Select Units
    selected_units = activations_utils.extract_random_units(layer_activations, random_units=FLAGS.random_units)
    print(f"Selected Units: {selected_units}")

    # Compute explanations only for clusters specified by the user (if any)
    clusters_to_analyze = list(FLAGS.cluster_to_analyze) if FLAGS.cluster_to_analyze else range(FLAGS.num_clusters)
    clusters_to_analyze = [int(cluster) for cluster in clusters_to_analyze]

    # Compute explanations
    for unit in selected_units:
        # Split the activations of the unit into clusters
        unit_activations = layer_activations[:, unit, :, :]
        activation_ranges = activations_utils.compute_activation_ranges(
                    unit_activations, num_clusters=FLAGS.num_clusters, quantile=FLAGS.quantile)

        print(f"Computing explanations for Unit {unit}")
        # Compute explanations for each cluster of the unit
        for cluster_index, activation_range in enumerate(
                    sorted(activation_ranges)
                ):
            if cluster_index not in clusters_to_analyze:
                continue  # Skip clusters not specified by the user

            # Get the directory and file path for the results of the current unit and cluster  
            dir_results = results_utils.get_result_dir(cfg, unit=unit, activation_range=activation_range)
            file_results = results_utils.get_result_file(dir_results, cluster_index=cluster_index, num_clusters=FLAGS.num_clusters, quantile=FLAGS.quantile)

            # Load or compute the explanations
            if os.path.exists(file_results):
                with open(file_results, "rb") as f:
                    numeric_explanation, string_explanation, iou = pickle.load(f)
            else:
                # Compute binary masks
                bitmaps = activations_utils.compute_bitmaps(
                    unit_activations,
                    activation_range,
                    mask_shape=cfg.get_mask_shape(),
                )
                bitmaps = bitmaps.to(cfg.get_device())


                # Compute explanations using beam search
                (
                    numeric_explanation,
                    iou,
                    _,
                ) = algorithms.get_heuristic_scores(
                    masks,
                    bitmaps,
                    segmentations_info=masks_info,
                    length=3,
                    beam_size=5,
                    max_size_mask=cfg.get_mask_shape()[0]*cfg.get_mask_shape()[1],
                    mask_shape=cfg.get_mask_shape(),
                    device=cfg.get_device(),
                )
                # Save the results
                with open(file_results, "wb") as f:
                    string_explanation = common_utils.get_formula_str(numeric_explanation, masks_labels)
                    pickle.dump((numeric_explanation, string_explanation, iou), f)

            # Print the results
            print(f"Unit {unit}, Cluster {cluster_index}: {string_explanation} ({numeric_explanation}). IoU: {iou:.4f}")


if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)
        absl.flags.mark_flag_as_required('mapping_file')