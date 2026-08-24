import os 

from src.segmentor_wrapper import Detectron2GroundTruth, CATSeg, Masqclip, SED, SCAN, OpenSeeD, Mask2Former
from utils import dataset_utils, common_utils


def get_segmentor(*, segmentor_name, dataset_name, device): 
    """
    Load the appropriate segmentor based on the segmentor name. Set the model and concept labels according to the dataset.
    Args:
        segmentor_name (str): The name of the segmentor to load.
        dataset_name (str): The name of the dataset to use.
        device (str): The device to use for the segmentor.
    Returns:
        Segmentor: The loaded segmentor.
    """
    if segmentor_name == "human":
        segmentor = Detectron2GroundTruth(dataset_name=dataset_name)
    elif segmentor_name == "catseg":
        segmentor = CATSeg(dataset_name=dataset_name, device=device)
    elif segmentor_name == "masqclip":
        segmentor = Masqclip(dataset_name=dataset_name, device=device)
    elif segmentor_name == "sed":
        segmentor = SED(dataset_name=dataset_name, device=device)
    elif segmentor_name == "scan":
        segmentor = SCAN(dataset_name=dataset_name, device=device)
    elif segmentor_name == "openseed":
        segmentor = OpenSeeD(dataset_name=dataset_name,  device=device)
    elif segmentor_name == "mask2former":
        segmentor = Mask2Former(dataset_name=dataset_name, device=device)
    else:
        raise ValueError(f"Segmentor {segmentor_name} not supported. Supported segmentors are: human, catseg, masqclip, sed, scan, openseed, mask2former")
    return segmentor


def extract_category(concept_sets):
    """
    Recursively extract concepts from nested concept sets.
    Args:
        concept_sets (list): A list of concept sets, which can be nested.
    Yields:
        str: Individual concepts extracted from the nested structure.
    """
    if concept_sets is None:
        return None
    item_concept_sets = concept_sets[0]
    if isinstance(item_concept_sets, str):
        yield concept_sets
    else:
        for concept_set in concept_sets:
            yield from extract_category(concept_set)


def get_masks(segmentor, *, segmentations_dir, mask_settings):
    """
    Get (load or compute) the segmentor outputs (masks) for the specified segmentations directory and mask settings.
    Args:
        segmentor: The segmentor object to use for extracting masks.
        segmentations_dir: Directory where the segmentations are stored or will be saved.
        mask_settings: Dictionary containing settings for mask extraction (e.g., mask shape, batch parsing, parallel concepts).
    Returns:
        List of segmentor outputs (masks) for the specified segmentations directory and mask settings.
    """
    mask_shape = mask_settings["mask_shape"]
    batch_parsing = mask_settings["batch_parsing"]
    parallel_concepts = mask_settings["parallel_concepts"]
    segmentor_masks = segmentor.load_concept_masks(segmentations_dir=segmentations_dir)
    generated = False
    if segmentor_masks is None:
        generated = True
        print(f"Regenerating concept masks for {len(segmentor.concept_labels)} concepts.")
        print(f"Cleaning {segmentations_dir} before regenerating concept masks.")
        common_utils.clean_directory(segmentations_dir)

        # Compute and save the segmentations if they do not exist
        segmentor.save_segmentations(parallel_concepts=parallel_concepts, output_dir=segmentations_dir, mask_shape=mask_shape, batch_parsing=batch_parsing)
        segmentor_masks = segmentor.load_concept_masks(segmentations_dir=segmentations_dir)
    return segmentor_masks, generated

def get_categorical_masks(segmentor, *, config, concept_set, mask_settings):
    """
    Get the segmentor outputs (masks and labels) for the specified categorical configuration and concept set.
    Args:
        segmentor: The segmentor object to use for extracting masks.
        config: Configuration object containing the segmentor name, dataset name, and device.
        concept_set: List of concept sets to extract masks and labels from.
        mask_settings: Dictionary containing settings for mask extraction (e.g., mask shape, batch parsing, parallel concepts).
    Returns:
        masks: List of segmentor outputs (masks) for the specified configuration and concept set.
        segmentor_labels: List of labels for the specified configuration and concept set.
    """
    segmentations_dir = config.get_segmentations_dir()
    segmentor_name = config.get_segmentor_name()
    dataset_name = config.get_dataset_name()
    device = config.get_device()
    masks = []
    segmentor_labels = []
    regenerate = False
    for index_subset, concept_subset in enumerate(extract_category(concept_set)):
        # Recreate the segmentor per subset to avoid model state carryover across concept batches.
        subset_segmentor = get_segmentor(segmentor_name=segmentor_name, dataset_name=dataset_name, device=device)
        subset_segmentor.set_concept_labels(concept_subset)

        # Create a directory for the current subset of concepts
        subset_segmentations_dir = os.path.join(segmentations_dir, f"subset_{index_subset}")
        os.makedirs(subset_segmentations_dir, exist_ok=True)

        # Get the masks for the current subset of concepts and add them to the list of masks
        masks_category, are_regenerated = get_masks(subset_segmentor, segmentations_dir=subset_segmentations_dir, mask_settings=mask_settings)
        del subset_segmentor

        if are_regenerated:
            regenerate = True
        masks += masks_category

        # Get the labels for the current subset of concepts and add them to the list of labels
        segmentor_labels.extend(concept_subset)
    return masks, segmentor_labels, regenerate

def get_segmentor_outputs(config, concept_set=None):
    """
    Get the segmentor outputs (masks and labels) for the specified configuration. If a concept set is provided, extract the masks and labels from the concept set.
    Args:
        config: Configuration object containing the segmentor name, dataset name, and device.
        concept_set: Optional list of concept sets to extract masks and labels from.
    Returns:
        masks: List of segmentor outputs (masks) for the specified configuration.
        segmentor_labels: List of labels for the specified configuration.
    """
    segmentor_name = config.get_segmentor_name()
    dataset_name = config.get_dataset_name()
    device = config.get_device()
    masks_settings = config.get_mask_settings()
    are_regenerated = False
    segmentor = get_segmentor(segmentor_name=segmentor_name, dataset_name=dataset_name, device=device)
    if concept_set is None:
        masks, are_regenerated = get_masks(segmentor, segmentations_dir=config.get_segmentations_dir(), mask_settings=masks_settings)
        segmentor_labels = segmentor.concept_labels
    else:
        masks, segmentor_labels, are_regenerated = get_categorical_masks(segmentor, config=config, concept_set=concept_set, mask_settings=masks_settings)
        if masks is None:
            raise ValueError(f"No masks found for the specified configuration. Please check the segmentor and dataset settings.")

    print(f"Loaded {len(masks)} segmentations for {segmentor_name} on {dataset_name}")
    del segmentor # Free memory after segmentations are extracted and saved to disk
    return masks, segmentor_labels, are_regenerated



def get_labels_config(config, concept_set=None):
    """
    Get the labels for the specified configuration. If a concept set is provided, extract the labels from the concept set.
    Args:
        config: Configuration object containing the dataset name.
        concept_set: Optional list of concept sets to extract labels from.
    Returns:
        List of labels for the specified configuration."""
    dataset_name = config.get_dataset_name()
    if concept_set is None:
        labels = dataset_utils.get_class_names(dataset_name=dataset_name)
    else:
        labels = []
        for index_subset, concept_subset in enumerate(extract_category(concept_set)):
            labels.extend(concept_subset)
    return labels
