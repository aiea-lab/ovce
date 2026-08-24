
import pickle
import os
from tqdm import tqdm

from src import settings
from utils import activations_utils, dataset_utils, segmentor_utils
from utils import common_utils


def get_result_dir(config, *, unit, activation_range):
    """
    Get the directory where the results are stored for a given unit and activation range.
    Args:
        config (Config): Configuration object containing the settings.
        unit (int): The unit for which to get the results directory.
        activation_range (tuple): The activation range for which to get the results directory.
    Returns:
        str: The path to the results directory for the given unit and activation range.
    """
    results_dir = config.get_results_dir()
    dir_current_results = os.path.join(results_dir, f"{unit}/{activation_range}")   
    if not os.path.exists(dir_current_results):
        os.makedirs(dir_current_results)
    return dir_current_results


def get_result_file(result_dir, *, cluster_index, num_clusters, quantile):
    """
    Get the file path for the results of a given cluster index, number of clusters, and quantile.
    Args:
        result_dir (str): The directory where the results are stored.
        cluster_index (int): The index of the cluster for which to get the results file.
        num_clusters (int): The total number of clusters.
        quantile (float): The quantile used for thresholding.
    Returns:
        str: The path to the results file for the given cluster index, number of clusters, and quantile.
    """
    if num_clusters == 1:
        file_algo_results = os.path.join(result_dir, f"quantile_{quantile}.pickle")
    else:
        file_algo_results = os.path.join(result_dir, f"cluster_{cluster_index}_of_{num_clusters}.pickle")
    return file_algo_results

def load_explanations(segmentor, flags):
    # Set seed
    common_utils.set_seed(flags.seed)

    cfg = settings.Settings(
        model= flags.model, 
        dataset=flags.dataset,
        segmentor=segmentor,
        layer=flags.layer,
        device=flags.device,
        batch_parsing=flags.batch_parsing,
        parallel_concepts=flags.parallel_concepts,
        root_models=flags.root_models,
        root_segmentations=flags.root_segmentations,
        root_activations=flags.root_activations,
        root_results=flags.root_results,
        configuration_name=flags.configuration_name,
    )
    # Load Activations
    layer_activations = activations_utils.get_layer_activations(cfg)

    # Load Concept Set (optionally specified by the user)
    if flags.predefined_concept_set is not None:
        assert flags.predefined_concept_set in ['granularity_0', 'granularity_1', 'granularity_2', 'all'], "Predefined concept set must be one of ['granularity_0', 'granularity_1', 'granularity_2', 'all']"
        assert flags.configuration_name != "std", "Predefined concept set can only be used with a custom configuration name"
        concept_set = dataset_utils.get_class_names(dataset_name=cfg.get_dataset_name(), custom_classes=flags.predefined_concept_set)
        print(f"Using predefined concept set: {flags.predefined_concept_set} with concepts: {concept_set}")
    elif flags.custom_classes is not None:
        assert flags.configuration_name != "std", "Custom classes can only be used with a custom configuration name"
        concept_set = flags.custom_classes
    else:
        concept_set = None

    masks_labels = segmentor_utils.get_labels_config(cfg, concept_set=concept_set)
    
    # Select Units
    selected_units = activations_utils.extract_random_units(layer_activations, random_units=flags.random_units)
    results = []

    # Compute explanations
    for unit in tqdm(selected_units, desc="Loading explanations for selected units"):
        # Split the activations of the unit into clusters
        unit_activations = layer_activations[:, unit, :, :]
        activation_ranges = activations_utils.compute_activation_ranges(
                    unit_activations, num_clusters=flags.num_clusters, quantile=flags.quantile)

        # Compute explanations for each cluster of the unit
        for cluster_index, activation_range in enumerate(
                    sorted(activation_ranges)
                ):
            # Get the directory and file path for the results of the current unit and cluster  
            dir_results = get_result_dir(cfg, unit=unit, activation_range=activation_range)
            file_results = get_result_file(dir_results, cluster_index=cluster_index, num_clusters=flags.num_clusters, quantile=flags.quantile)

            # Load or compute the explanations
            if os.path.exists(file_results):
                with open(file_results, "rb") as f:
                    numeric_explanation, string_explanation, iou = pickle.load(f)
            else:
                numeric_explanation, string_explanation, iou = None, None, None  # Placeholder for cases where the file does not exist
            results.append((unit, cluster_index, numeric_explanation, string_explanation, iou))

    return results, masks_labels
