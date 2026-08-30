"""Script to run the clustering algorithm for compositional explanations.
"""
import os
import logging
import pickle

import torch
import absl.flags
import absl.app

from compositional_explanations import beam
from compositional_explanations import metrics as optimal_metrics
from compositional_explanations import formula as optimalF


from utils import common_utils, activations_utils, dataset_utils, segmentor_utils, concept_mask_utils, results_utils
from src import common_flags # import all user flags
from src import formula as F
from src import settings

# Disable overly verbose logging of detectron2
logger = logging.getLogger("detectron2")
logger.setLevel(logging.WARNING)

FLAGS = absl.flags.FLAGS

    
def covert_package_formula(optimal_formula):
    """
    Converts an optimal formula to a standard formula.
    """

    if isinstance(optimal_formula, optimalF.And):
        return F.And(covert_package_formula(optimal_formula.left), covert_package_formula(optimal_formula.right))
    elif isinstance(optimal_formula, optimalF.Or):
        return F.Or(covert_package_formula(optimal_formula.left), covert_package_formula(optimal_formula.right))
    elif isinstance(optimal_formula, optimalF.Not):
        return F.Not(covert_package_formula(optimal_formula.val))
    elif isinstance(optimal_formula, optimalF.Leaf):
        return F.Leaf(optimal_formula.val)
    else:
        raise ValueError("Unsupported formula type")


def main(argv):
    if FLAGS.num_clusters < 1:
        raise ValueError("num_clusters must be greater than 0")
    
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
    masks, masks_labels, masks_are_regenerated = segmentor_utils.get_segmentor_outputs(cfg, concept_set=concept_set)
    masks = concept_mask_utils.remove_unmeaningful_masks(concept_masks=masks, concept_labels=masks_labels, ignore_concepts=ignore_concepts)

    for concept, mask in zip(masks_labels, masks):
        print(f"Concept: {concept}, Mask shape: {mask.shape}, Mask sum: {mask.sum().item()}")
    cache_info_dir = cfg.get_info_dir()
    if masks_are_regenerated:
        # We need to recompute the informations
        common_utils.clean_directory(cache_info_dir)

    # Compute Disjoint Matrix
    if FLAGS.predefined_concept_set is None and FLAGS.custom_classes is None:
        # In this case the segmentations are fully disjoint by definition, since the segmentors use only 1 concept subset
        disjoint_matrix = concept_mask_utils.get_full_disjoint_matrix(num_concepts=len(masks))
    else:
        # More general case. This function can be slow but it is only computed once per configuration
        disjoint_matrix = concept_mask_utils.compute_disjoint_matrix(info_dir=cache_info_dir, concept_masks=masks, block_size=FLAGS.parallel_concepts)


    # Select Units
    selected_units = activations_utils.extract_random_units(layer_activations, random_units=FLAGS.random_units)
    print(f"Selected Units: {selected_units}")

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
                numeric_explanation = beam.compute_beam_explanations(
                    bitmaps=bitmaps, masks=masks, disjoint_info=disjoint_matrix, length=3, beam_size=5, device=cfg.get_device(), cache_dir= cfg.get_info_dir()

                )  

                # Compute IoU for the explanation
                iou = optimal_metrics.compute_iou_from_masks(bitmaps=bitmaps, masks=masks, formula=numeric_explanation)
            
                # Save the results
                with open(file_results, "wb") as f:
                    numeric_explanation = covert_package_formula(numeric_explanation)
                    string_explanation = common_utils.get_formula_str(numeric_explanation, masks_labels)
                    pickle.dump((numeric_explanation, string_explanation, iou), f)

            # Print the results
            print(f"Unit {unit}, Cluster {cluster_index}: {string_explanation} ({numeric_explanation}). IoU: {iou:.3f}")


if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)
