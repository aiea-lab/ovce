from types import SimpleNamespace
import os
import pickle
import json

import absl.app
import torch
from tqdm import tqdm
import wn

from utils import concept_mask_utils, wordnet_utils, segmentor_utils, common_utils, analysis_utils, results_utils, activations_utils, formula_utils, dataset_utils
from src import settings
from src import metrics
from src import common_flags # import all user flags

FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_string("mapping_file", None, "Path to the mapping file")
absl.flags.DEFINE_list(
    "compare", ['human', 'catseg'], "list of segmentors to compare"
)
absl.flags.DEFINE_list("cluster_to_analyze", None, "Cluster to analyze")


def map_concept_sets(concept_set_1, concept_set_2):
    """
    Map the concepts from concept_set_1 to concept_set_2 based on their names and synonyms.
    This function assumes that the two concept sets are aligned and share the same concepts.
    
    Args:
        concept_set_1 (list): List of concepts from the first segmentor.
        concept_set_2 (list): List of concepts from the second segmentor.
    Returns:
        dict: A mapping from the indices of concept_set_1 to the indices of concept_set_2.
    """
    mapping = {}
    for i, concept_1 in enumerate(concept_set_1):
        if concept_1 in concept_set_2:
            # In this case we map the index since the concept is shared between the two segmentors
            mapping[i] = concept_set_2.index(concept_1)
        else:
            # If there is no direct match, we can try to find a match based on synonyms 
            synonyms_1 = concept_1.split(", ")
            for concept_2 in concept_set_2:
                synonyms_2 = concept_2.split(", ")
                for synonym_1 in synonyms_1:
                    if synonym_1 in synonyms_2:
                        # They share at least one synonym, we can map them together
                        mapping[i] = concept_set_2.index(concept_2)
                        break
                else:
                    # If no mapping found so far, check in the other concepts in the concept set 2 i
                    continue
                # If we found a match, we can continue to the next concept in concept set 1
                break
            else:
                raise ValueError(f"No mapping found for concept {concept_1} in concept set 2. This script assumes that the two concept sets are aligned and share the same concepts")
                      
    return mapping


def get_diff_relations(*, undetected_concept, reference_label, other_label, bitmaps, reference_masks, other_masks, device):
    """
    Get the relations between an undetected concept and the other concepts in the other explanation.
    Args:
        undetected_concept: The undetected concept to analyze.
        reference_label: The reference explanation.
        other_label: The other explanation.
        bitmaps: The bitmaps of the samples.
        reference_masks: The masks of the reference segmentor.
        other_masks: The masks of the other segmentor.
        device: The device to use for computation.
    Returns:
        A SimpleNamespace object containing the number of hyper-related, related, and unrelated concepts.
    """
    candidate_concepts = formula_utils.get_positive_concepts(other_label)
    atom_active_in = analysis_utils.get_atom_conditionally_activation(undetected_concept, reference_label, reference_masks, bitmaps, device)

    is_hyper_related = False
    is_related = False
    for concept in candidate_concepts:
        concept_active_in = analysis_utils.get_atom_conditionally_activation(concept, other_label, other_masks, bitmaps, device)
        true_positive = torch.sum(atom_active_in & concept_active_in)
        false_negative = torch.sum(atom_active_in & ~concept_active_in)
        false_positive = torch.sum(~atom_active_in & concept_active_in)

        # Specialization
        specialization = true_positive/(true_positive + false_negative) if (true_positive + false_negative).sum() > 0 else 0.0

        # Generalization
        generalization = true_positive/(true_positive + false_positive) if (true_positive + false_positive).sum() > 0 else 0.0

        if specialization >= 0.75 or generalization >= 0.75:
            is_hyper_related = True
        elif specialization >= 0.5 or generalization >= 0.5:
            is_related = True

    return SimpleNamespace(
        hyper_related=is_hyper_related,
        related=not is_hyper_related and is_related, # If it is hyper-related, it is also related, but we want to count it only as hyper-related
        unrelated=not is_hyper_related and not is_related, # If it is not hyper-related and not related, it is unrelated
    )

def get_segmentor_masks(cfg, *, mapping=None, custom_concept_set=None):
    """
    Get the segmentor outputs (masks and labels) for the specified configuration. If a concept set is provided, extract the masks and labels from the concept set.
    Args:
        cfg: Configuration object containing the segmentor name, dataset name, and device.
        mapping: Optional mapping to refine the concept set.
        custom_concept_set: Optional list of custom concept sets to extract masks and labels from.
    Returns:
        masks: List of segmentor outputs (masks) for the specified configuration.
        segmentor_labels: List of labels for the specified configuration.
    """
    # Load Concept Set (optionally specified by the user)
    if isinstance(custom_concept_set, str):
        assert custom_concept_set in ['granularity_0', 'granularity_1', 'granularity_2', 'all'], "Predefined concept set must be one of ['granularity_0', 'granularity_1', 'granularity_2', 'all']"
        assert cfg.get_configuration_name() != "std", "Predefined concept set can only be used with a custom configuration name"
        concept_set = dataset_utils.get_class_names(dataset_name=cfg.get_dataset_name(), custom_classes=custom_concept_set)
        print(f"Using predefined concept set: {concept_set} with concepts: {concept_set}")
    elif isinstance(custom_concept_set, list):
        assert cfg.get_configuration_name() != "std", "Custom classes can only be used with a custom configuration name"
        concept_set = custom_concept_set
    else:
        concept_set = None
    # Load Segmentation Masks
    if cfg.get_segmentor_name() == "human":
        # In this case we can merge the masks after loading following the mapping
        masks, masks_labels, _ = segmentor_utils.get_segmentor_outputs(cfg, concept_set=concept_set)
        if mapping is not None:
            masks, masks_labels = concept_mask_utils.merge_human_masks_with_mapping(masks, masks_labels, mapping=mapping)
    elif cfg.get_segmentor_name() == "mask2former":
        raise ValueError("Mask2Former segmentor is a closed-world approach and cannot support modifications in the concept set.")
    else:
        # In this case we need to modify the concept set before loading the masks
        if mapping is not None:
            concept_set = segmentor_utils.get_labels_config(cfg, concept_set=concept_set)
            concept_set = common_utils.refine_concept_set(concept_set, mapping)
        masks, masks_labels, _ = segmentor_utils.get_segmentor_outputs(cfg, concept_set=concept_set)
    return masks, masks_labels

def load_explanation(cfg, *, unit, cluster_index, activation_range):
    """
    Load the explanation for a given unit and cluster index from the results directory.
    Args:
        cfg: The configuration object.
        unit: The unit index.
        cluster_index: The cluster index.
        activation_range: The activation range for the cluster.
    Returns:
        A tuple containing the numeric explanation, string explanation, and IoU.
    """
    # Get the directory and file path for the results of the current unit and cluster  
    dir_results = results_utils.get_result_dir(cfg, unit=unit, activation_range=activation_range)
    file_results = results_utils.get_result_file(dir_results, cluster_index=cluster_index, num_clusters=FLAGS.num_clusters, quantile=FLAGS.quantile)
    if os.path.exists(file_results):
        with open(file_results, "rb") as f:
            numeric_explanation, string_explanation, iou = pickle.load(f)
        return numeric_explanation, string_explanation, iou
    else:
        return None, None, None

def main(argv):

    # Set seed (necessary for reproducibility of random units selection)
    common_utils.set_seed(FLAGS.seed)

    # Load competitors explanations
    assert FLAGS.compare is not None, "Please provide a list of segmentors to compare using the --compare flag."
    assert len(FLAGS.compare) > 1, "Please provide at least two segmentors to compare."

    segmentor1 = FLAGS.compare[0]
    segmentor2 = FLAGS.compare[1]

    # Load Configs
    cfg_segmentor_1 = settings.Settings(
        model= FLAGS.model, 
        dataset=FLAGS.dataset,
        segmentor=segmentor1,
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

    cfg_segmentor_2 = settings.Settings(
        model= FLAGS.model, 
        dataset=FLAGS.dataset,
        segmentor=segmentor2,
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

    # Load Mapping
    if FLAGS.mapping_file is not None:
        with open(FLAGS.mapping_file, 'r') as f:
            mapping = json.load(f)
    else:
        mapping = None

    # Load Segmentor Masks and Labels
    segmentor1_masks, segmentor1_labels = get_segmentor_masks(cfg_segmentor_1, mapping=mapping, custom_concept_set=FLAGS.custom_classes)
    segmentor2_masks, segmentor2_labels = get_segmentor_masks(cfg_segmentor_2, mapping=mapping, custom_concept_set=FLAGS.custom_classes)

    # Create mapping from segmentor1_labels to segmentor2_labels
    segmentor1_to_segmentor2_mapping = map_concept_sets(segmentor1_labels, segmentor2_labels)

    # Load Activations
    layer_activations = activations_utils.get_layer_activations(cfg_segmentor_1)

    # Select Units
    selected_units = activations_utils.extract_random_units(layer_activations, random_units=FLAGS.random_units)

    # Compare explanations
    clusters_to_analyze = list(FLAGS.cluster_to_analyze) if FLAGS.cluster_to_analyze else range(FLAGS.num_clusters)
    clusters_to_analyze = [int(cluster) for cluster in clusters_to_analyze]

    # Utils Wordnet
    wn.download('oewn')
    wordnet = wn.Wordnet('oewn')
    segmentor1_mapping = wordnet_utils.load_synset_mapping(wordnet=wordnet, labels=segmentor1_labels)
    segmentor_1_id_to_concept_id = {int(v.split('-')[1]): segmentor1_labels.index(k) for k, v in segmentor1_mapping.items()}
    segmentor2_mapping = wordnet_utils.load_synset_mapping(wordnet=wordnet, labels=segmentor2_labels)

    # Difference in IoU
    diff_iou = []

    # Metrics
    hyper_related_total = 0
    related_total = 0
    unrelated_total = 0
    same = 0
    total_concepts = 0
    
    # Compute explanations
    for unit in tqdm(selected_units, desc="Analyzing Units"):
        # Split the activations of the unit into clusters
        unit_activations = layer_activations[:, unit, :, :]
        activation_ranges = activations_utils.compute_activation_ranges(
                    unit_activations, num_clusters=FLAGS.num_clusters, quantile=FLAGS.quantile)
        for cluster_index, activation_range in enumerate(
                    sorted(activation_ranges)
                ):
            if cluster_index not in clusters_to_analyze:
                continue  # Skip clusters not specified by the user

            # Compute binary masks
            bitmaps = activations_utils.compute_bitmaps(
                unit_activations,
                activation_range,
                mask_shape=cfg_segmentor_1.get_mask_shape(),
            )
            bitmaps = bitmaps.to(cfg_segmentor_1.get_device())
            sample_bitmaps = torch.any(bitmaps, dim=1)

            # Load explanations for both segmentors
            explanation_segmentor_1, _, iou_1 = load_explanation(cfg_segmentor_1, unit=unit, cluster_index=cluster_index, activation_range=activation_range)
            explanation_segmentor_2, _, iou_2 = load_explanation(cfg_segmentor_2, unit=unit, cluster_index=cluster_index, activation_range=activation_range)
            
            # Map explanations to Wordnet synsets so that they can be compared even if concepts are associated with different id or if the labels slightly differ
            map_expl1_to_wordnet = analysis_utils.map_explanation_to_wordnet(explanation_segmentor_1, segmentor1_labels, segmentor1_mapping)
            map_expl2_to_wordnet = analysis_utils.map_explanation_to_wordnet(explanation_segmentor_2, segmentor2_labels, segmentor2_mapping)

            # Get undetected concepts from reference explanation (explanation 1)
            undetected_concepts = analysis_utils.get_undetected_concepts(ref_explanation=map_expl1_to_wordnet, other_explanation=map_expl2_to_wordnet)
            undetected_concepts = [segmentor_1_id_to_concept_id[concept.val] if concept.val > 0 else abs(concept.val) for concept in undetected_concepts]   # Filter out concepts with val <= 0

            # Compute the relations for each undetected concept
            for concept in undetected_concepts:
                diff_results = get_diff_relations(undetected_concept=concept, reference_label=explanation_segmentor_1, other_label=explanation_segmentor_2, bitmaps=sample_bitmaps, reference_masks=segmentor1_masks, other_masks=segmentor2_masks, device=cfg_segmentor_1.get_device())

                # Update metrics
                hyper_related_total += 1 if diff_results.hyper_related else 0
                related_total += 1 if diff_results.related else 0
                unrelated_total += 1 if diff_results.unrelated else 0
            total_concepts += len(formula_utils.get_positive_concepts(explanation_segmentor_1))
            same += len(formula_utils.get_positive_concepts(explanation_segmentor_1)) - len(undetected_concepts)

            if len(undetected_concepts) > 0:
                converted_explanation_1 = formula_utils.convert_formula(explanation_segmentor_1, segmentor1_to_segmentor2_mapping)
                converted_mask = concept_mask_utils.get_formula_mask(converted_explanation_1, segmentor2_masks).to(cfg_segmentor_1.get_device())
                iou_converted = metrics.iou(converted_mask, bitmaps).item()
                diff_iou.append(max(iou_converted - iou_2, iou_2 - iou_converted))
      
    print(f"Segmentor {segmentor1} vs {segmentor2}")
    print("% Same: {:.2f}%".format(same/total_concepts*100))
    print(f"% Hyper Related: {hyper_related_total/total_concepts*100:.2f}%")
    print(f"% Related: {related_total/total_concepts*100:.2f}%")
    print(f"% Unrelated: {unrelated_total/total_concepts*100:.2f}%")
    print(f"Total Undetected: {total_concepts}")
    print(f"Average IoU Difference: {sum(diff_iou)/len(diff_iou) if len(diff_iou) > 0 else 0:.4f}")


if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)