import random
import os

import numpy as np
import torch
from numpy.random import RandomState

from src import metrics
from src import formula as F


def sparse_to_torch(vector):
    """
    Convert a sparse matrix to a torch tensor.

    Args:
        vector (scipy.sparse.csr_matrix): sparse matrix to convert

    Returns:
        torch.Tensor: tensor
    """
    return torch.from_numpy(vector.toarray())

def set_seed(seed: int) -> RandomState:
    """Method to set seed across runs to ensure reproducibility.
    It fixes seed for single-gpu machines.
    Args:
        seed (int): Seed to fix reproducibility. It should different for
            each run
    Returns:
        RandomState: fixed random state to initialize dataset iterators
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = (
        False  # set to false for reproducibility, True to boost performance
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return g


def refine_concept_set(concept_set, mapping):
    """
    Refine a set of concepts by applying a mapping to unify concepts.
    Args:
        concept_set: A set of concepts to refine.
        mapping: A dictionary mapping concepts to their unified representations.
    Returns:
        A list of refined concepts after applying the mapping.
    """
    new_concept_set = set()
    for concept in concept_set:
        if concept in mapping:
            new_concept_set.add(mapping[concept])
        else:
            new_concept_set.add(concept)
    return sorted(list(new_concept_set))

def get_formula_str(compositional_label, concept_labels):
    """
    Function to get the string representation of a formula.

    Args:
        compositional_label: Formula to get the string representation of.
        concept_labels: List of names for the variables in the formula.

    Returns:
        String representation of the formula.
    """
    if isinstance(compositional_label, F.And):
        masks_l = get_formula_str(compositional_label.left, concept_labels)
        masks_r = get_formula_str(compositional_label.right, concept_labels)
        return f"({masks_l} AND {masks_r})"
    elif isinstance(compositional_label, F.Or):
        masks_l = get_formula_str(compositional_label.left, concept_labels)
        masks_r = get_formula_str(compositional_label.right, concept_labels)
        return f"({masks_l} OR {masks_r})"
    elif isinstance(compositional_label, F.Not):
        return f"NOT {get_formula_str( compositional_label.val, concept_labels)}"
    elif isinstance(compositional_label, F.Leaf):
        return concept_labels[compositional_label.val]
    elif isinstance(compositional_label, int):
        return concept_labels[compositional_label]

def compute_scores(*, formula, masks, activations):
    """Compute the scores for the given masks.
    Args:
        formula: A compositional formula representing the explanation.
        masks: A list of masks corresponding to the concepts in the formula.
        activations: A tensor of shape (N, H*W) where N is the number of samples.
    Returns:
        dict_results (dict): A dictionary containing the scores.
    """
    iou = metrics.compute_iou_from_masks(
        formula=formula, masks=masks, activations=activations
    )
    activation_coverage = metrics.compute_activation_coverage_from_masks(
        formula=formula, masks=masks, activations=activations
    )
    detection_accuracy = metrics.compute_detection_accuracy_from_masks(
        formula=formula, masks=masks, activations=activations
    )
    samples_coverage = metrics.compute_samples_coverage_from_masks(
        formula=formula, masks=masks, activations=activations
    )

    explanation_coverage = metrics.compute_explanation_coverage_from_masks(
        formula=formula, masks=masks, activations=activations
    )

    dict_results = {
        "iou": iou.item(),
        "activation_coverage": activation_coverage.item(),
        "label_coverage": detection_accuracy.item(),
        "samples_coverage": samples_coverage.item(),
        "explanation_coverage": explanation_coverage.item(),
    }
    return dict_results

def clean_directory(directory):
    """
    Clean a directory by removing all files.
    Args:
        directory (str): The path to the directory to clean.
    """
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)