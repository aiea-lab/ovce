
import json
import os
from types import SimpleNamespace

from tqdm import tqdm
import absl.flags
import absl.app
import wn
import torch

from src import common_flags  # Register shared command-line flags.
from utils import results_utils, analysis_utils, common_utils, wordnet_utils, formula_utils
from src import formula as F

absl.flags.DEFINE_string(
    "root_mapping", "data/cache/mapping", "root directory for mapping"
)
absl.flags.DEFINE_string(
    "output_file_name", "merging_results.json", "output file name for merging results"
)

absl.flags.DEFINE_list(
    "compare", ['human', 'catseg'], "list of segmentors to compare"
)

absl.flags.DEFINE_list("cluster_to_analyze", None, "Cluster to analyze")
absl.flags.DEFINE_string("mapping_file", None, "Path to the mapping file")


FLAGS = absl.flags.FLAGS


def merge_human_labels_with_mapping(masks_labels, mapping):
    """
    Merges human labels based on a provided mapping.
    Args:
        masks: List of masks to be merged.
        mapping: A dictionary containing the mapping of concepts.
    Returns:
        merged_masks: List of merged masks.
        merged_labels: Corresponding labels for the merged masks.
    """

    for label in mapping.keys():
        # Get the label to merge into
        merged_into = mapping[label]

        # Check if the new labels is an abstraction not present in the segmentor concept labels
        if merged_into not in masks_labels:
            masks_labels.append(merged_into)

    return masks_labels

def generalize_merging(proposed_merging, wordnet):
    """
    Generalize the proposed merging by finding the highest common ancestor between the identified common ancestors.
    Args:
        proposed_merging: A list of tuples containing undetected concepts, matched concepts, and their common ancestors.
        wordnet: Wordnet object
    Returns:
        A list of tuples containing undetected concepts, matched concepts, and their generalized common ancestors.
    """
    ancestors = [matched_synset for _, _, matched_synset in proposed_merging]
    # We unify ancestors that can be unified/generalized
    revised_merging = []
    for undetected_atom, matched_concept, ancestor in proposed_merging:
        paths_to_root = wn.taxonomy.hypernym_paths(ancestor)
        path_candidates = []
        for path in paths_to_root:
            highest_common_ancestor = None
            for node in path:
                if node in ancestors:
                    # Possible candidate
                    if highest_common_ancestor is None:
                        highest_common_ancestor = node
                    else:
                        # Check if the new candidate is a descendant of the current highest common ancestor
                        for path_highest in wn.taxonomy.hypernym_paths(highest_common_ancestor):
                                if node in path_highest:
                                    highest_common_ancestor = node
            if highest_common_ancestor is not None:
                path_candidates.append(highest_common_ancestor)
        if len(path_candidates) == 0:
            # There are no generalizations of this ancestor
            revised_merging.append((undetected_atom, matched_concept, ancestor))
        elif len(path_candidates) == 1:
            # There is only one generalization of this ancestor
            revised_merging.append((undetected_atom, matched_concept, path_candidates[0]))
        else:
            # There are multiple generalizations of this ancestor.
            # We consider the best one the one that is closest to the original synsets
            synset_1 = wordnet_utils.get_synset_from_number(wordnet, undetected_atom)
            synset_2 = wordnet_utils.get_synset_from_number(wordnet, matched_concept)
            best_candidate = None
            min_distance = float('inf')
            for candidate in path_candidates:
                path1 = wn.taxonomy.shortest_path(synset_1, candidate)
                path2 = wn.taxonomy.shortest_path(synset_2, candidate)
                distance = len(path1) + len(path2)
                if distance < min_distance:
                    min_distance = distance
                    best_candidate = candidate
            revised_merging.append((undetected_atom, matched_concept, best_candidate))
    return revised_merging
                
def extract_concepts_mapping(merging, id_to_concept):
    """
    Extract a mapping of concepts based on the proposed merging of undetected and matched concepts.
    Args:
        merging: A list of tuples containing undetected concepts, matched concepts, and their common ancestors
        id_to_concept: A dictionary mapping concept IDs to their string representations.
    Returns:
        A dictionary mapping original concepts to their merged representations.
    """
    mapping_concepts = {}
    for id1, id2, ancestor in merging:
        concept1 = id_to_concept[id1]
        concept2 = id_to_concept[id2]
        merged_concept = ', '.join(ancestor.lemmas())
        mapping_concepts[concept1] = merged_concept
        mapping_concepts[concept2] = merged_concept
    return mapping_concepts

def integreate_old_mapping(*, old_mapping, new_mapping):
    """
    Integrate an old mapping with a new mapping, ensuring that the original concepts are further generalized if necessary.
    Args:
        old_mapping: A dictionary containing the old mapping of concepts.
        new_mapping: A dictionary containing the new mapping of concepts.
    Returns:
        A dictionary containing the integrated mapping of concepts.
    """
    merged_mapping = new_mapping.copy()
    for concept_from, concept_to in old_mapping.items():
        # In this case the original concept must be further generalized
        if concept_to in new_mapping:
            merged_mapping[concept_from] = new_mapping[concept_to]
        else:
            merged_mapping[concept_from] = concept_to
    return merged_mapping

def update_segmentor_labels_with_mapping(segmentor_name, segmentor_labels, mapping):
    """
    Update the segmentor labels based on a provided mapping.
    Args:
        segmentor_name: Name of the segmentor (e.g., "human", "catseg", etc.)
        segmentor_labels: List of labels corresponding to the segmentor's concepts.
        mapping: A dictionary containing the mapping of concepts.
    Returns:
        updated_labels: List of updated labels after applying the mapping.
    """
    if segmentor_name == "human":
        return merge_human_labels_with_mapping(segmentor_labels, mapping)                
    else:
        return common_utils.refine_concept_set(segmentor_labels, mapping)



def main(argv):

    # Set seed (necessary for reproducibility of random units selection)
    common_utils.set_seed(FLAGS.seed)

    # Load competitors explanations
    assert FLAGS.compare is not None, "Please provide a list of segmentors to compare using the --compare flag."
    assert len(FLAGS.compare) > 1, "Please provide at least two segmentors to compare."

    segmentor1 = FLAGS.compare[0]
    segmentor2 = FLAGS.compare[1]

    # Load explanations
    flags_values = SimpleNamespace(**FLAGS.flag_values_dict())
    segmentor1_explanations, segmentor1_labels = results_utils.load_explanations(segmentor=segmentor1, flags=flags_values)
    segmentor2_explanations, segmentor2_labels = results_utils.load_explanations(segmentor=segmentor2, flags=flags_values)

    if FLAGS.mapping_file is not None:
        with open(FLAGS.mapping_file, 'r') as f:
            old_mapping = json.load(f)
        segmentor1_labels = update_segmentor_labels_with_mapping(segmentor_name=segmentor1, segmentor_labels=segmentor1_labels, mapping=old_mapping)
        segmentor2_labels = update_segmentor_labels_with_mapping(segmentor_name=segmentor2, segmentor_labels=segmentor2_labels, mapping=old_mapping)
    else:
        old_mapping = None

    # Map segmentor labels to Wordnet synsets
    wn.download('oewn')
    wordnet = wn.Wordnet('oewn')
    segmentor1_mapping = wordnet_utils.build_synset_mapping_best_synset(wordnet=wordnet, labels=segmentor1_labels)
    segmentor_1_id_to_concept = {int(v.split('-')[1]): k for k, v in segmentor1_mapping.items()}
    segmentor2_mapping = wordnet_utils.build_synset_mapping_best_synset(wordnet=wordnet, labels=segmentor2_labels)
    segmentor2_id_to_concept = {int(v.split('-')[1]): k for k, v in segmentor2_mapping.items()}

    # Compare explanations
    clusters_to_analyze = list(FLAGS.cluster_to_analyze) if FLAGS.cluster_to_analyze else range(FLAGS.num_clusters)
    clusters_to_analyze = [int(cluster) for cluster in clusters_to_analyze]

    merging_list = []
    for (expl_segm1, expl_segm2) in tqdm(zip(segmentor1_explanations, segmentor2_explanations), total=min(len(segmentor1_explanations), len(segmentor2_explanations)), desc="Comparing explanations"):
        unit_1, cluster_index_1, numeric_explanation_1, string_explanation_1, iou_1 = expl_segm1
        unit_2, cluster_index_2, numeric_explanation_2, string_explanation_2, iou_2 = expl_segm2
        assert unit_1 == unit_2, f"Unit mismatch: {unit_1} != {unit_2}"
        assert cluster_index_1 == cluster_index_2, f"Cluster index mismatch: {cluster_index_1} != {cluster_index_2}"

        # Skip clusters that are not in the list of clusters to analyze
        if cluster_index_1 not in clusters_to_analyze:
            continue

        map_expl1_to_wordnet = analysis_utils.map_explanation_to_wordnet(numeric_explanation_1, segmentor1_labels, segmentor1_mapping)
        map_expl2_to_wordnet = analysis_utils.map_explanation_to_wordnet(numeric_explanation_2, segmentor2_labels, segmentor2_mapping)
        
        undetected = analysis_utils.get_undetected_concepts(ref_explanation=map_expl1_to_wordnet, other_explanation=map_expl2_to_wordnet)

        # Search if the undetected concepts have a common ancestor with any of the concepts in the other explanation
        for undetected_atom in undetected:
            candidate_concepts = formula_utils.get_positive_concepts(map_expl2_to_wordnet)
            common_ancestors, matched_concepts = wordnet_utils.search_common_ancestors(wordnet=wordnet, concept_to_unify=undetected_atom, candidate_concepts=candidate_concepts)
            if len(common_ancestors) > 0:
                for common_ancestor, matched_concept in zip(common_ancestors, matched_concepts):
                    merging_list.append((undetected_atom.val, matched_concept.val, common_ancestor))

    print(f"Parsing proposed merging...")
    # Generalize merging (extract the highest common ancestor between the identified common ancestors)
    revised_merging = generalize_merging(merging_list, wordnet)

    # Map the old concepts to the new generalized concepts if possible
    synsets_segmentor_1 = [wordnet.synset(id) for _, id  in segmentor1_mapping.items()]
    synsets_segmentor_2 = [wordnet.synset(id) for _, id  in segmentor2_mapping.items()]
    additional_merging_1 = wordnet_utils.generalize_labels_with_ancestors(synsets=synsets_segmentor_1, ancestors=[ancestor for _, _, ancestor in revised_merging])
    additional_merging_2 = wordnet_utils.generalize_labels_with_ancestors(synsets=synsets_segmentor_2, ancestors=[ancestor for _, _, ancestor in revised_merging])
    additional_merging = additional_merging_1 + additional_merging_2

    # Compute the final merging by combining the revised merging and the additional merging
    final_merging = revised_merging + additional_merging

    # Extract mapping of concepts based on the proposed merging
    id_to_concepts = {**segmentor_1_id_to_concept, **segmentor2_id_to_concept}
    mapping_concepts = extract_concepts_mapping(final_merging, id_to_concepts)

    print(f"Final concept mapping results:")
    for concept_from in sorted(mapping_concepts.keys()):
        print(f"{concept_from} -> {mapping_concepts[concept_from]}")

    # We add and integrate the old mapping, if any
    if old_mapping is not None:
        mapping_concepts = integreate_old_mapping(old_mapping=old_mapping, new_mapping=mapping_concepts)

    # Save mapping to file
    mapping_file = os.path.join(FLAGS.root_mapping, FLAGS.output_file_name)
    with open(mapping_file, 'w') as f:
        json.dump(mapping_concepts, f)
        print(f"Mapping saved to {mapping_file}")

    segmentor1_revised_labels = common_utils.refine_concept_set(segmentor1_labels, mapping_concepts)
    segmentor2_revised_labels = common_utils.refine_concept_set(segmentor2_labels, mapping_concepts)
    print(f"Segmentor 1 ({segmentor1}) num original concepts: {len(segmentor1_labels)}, num revised concepts: {len(segmentor1_revised_labels)}")
    print(f"Segmentor 2 ({segmentor2}) num original concepts: {len(segmentor2_labels)}, num revised concepts: {len(segmentor2_revised_labels)}")


if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)