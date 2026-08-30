import functools

import torch

from utils import concept_mask_utils

def get_num_nonzerosamples(mask):
    """Compute the number of samples with at least one pixel.
    Args:
        mask (torch.Tensor): A tensor of shape (N, H*W) where N is the
            number of sample.
    Returns:
        num_nonzerosamples (int): The number of samples with at
            least one pixel.
    """
    return torch.sum(torch.sum(mask, 1, dtype=torch.int32) > 0)

def activations_coverage(activations, segmentations):
    """Compute the activation coverage for the given activations and
    segmentations.
    Args:
        activations (torch.Tensor): A tensor of shape (N, H*W) where N is
            the number of sample.
        segmentations (torch.Tensor): A tensor of shape (N, H*W) where N is
            the number of sample.
    Returns:
        activation_coverage (float): The activation coverage.
    """

    return torch.count_nonzero(
        activations & segmentations
    ) / torch.count_nonzero(activations)

def compute_activation_coverage_from_masks(*, formula, masks, activations):
    """Compute the activation coverage for the given formula, masks, and segmentations.
    Args:
        formula: A compositional formula representing the explanation.
        masks: A list of masks corresponding to the concepts in the formula.
        activations: A tensor of shape (N, H*W) where N is the number of samples.
    Returns:
        activation_coverage (float): The activation coverage.
    """
    formula_mask = concept_mask_utils.get_formula_mask(formula, masks)
    formula_mask = formula_mask.to(activations.device)
    return activations_coverage(formula_mask, activations)
    
def explanation_coverage(activations, segmentations):
    """Compute the explanation coverage for the given activations and
    segmentations.
    Args:
        activations (torch.Tensor): A tensor of shape (N, H*W) where N is the number of sample.
        segmentations (torch.Tensor): A tensor of shape (N, H*W) where N is the number of sample.
    Returns:
        explanation_coverage (float): The explanation coverage.
    """
    return get_num_nonzerosamples(
        activations & segmentations) / (
            activations.sum(1) > 0).sum()

def compute_explanation_coverage_from_masks(*, formula, masks, activations):
    """Compute the explanation coverage for the given formula, masks, and segmentations.
    Args:
        formula: A compositional formula representing the explanation.
        masks: A list of masks corresponding to the concepts in the formula.
        activations: A tensor of shape (N, H*W) where N is the number of samples.
    Returns:
        explanation_coverage (float): The explanation coverage.
    """
    formula_mask = concept_mask_utils.get_formula_mask(formula, masks)
    formula_mask = formula_mask.to(activations.device)
    return explanation_coverage(formula_mask, activations)    

def detection_accuracy(activations, segmentations):
    """Compute the segmentations coverage for the given activations and
    segmentations.
    Args:
        activations (torch.Tensor): A tensor of shape (N, H*W) where N is the
            number of sample.
        segmentations (torch.Tensor): A tensor of shape (N, H*W) where N is
            the number of sample.
    Returns:
        segmentations_coverage (float): The segmentations coverage.
    """
    return torch.count_nonzero(
        activations & segmentations
    ) / torch.count_nonzero(segmentations)

def compute_detection_accuracy_from_masks(*, formula, masks, activations):
    """Compute the detection accuracy for the given formula, masks, and segmentations.
    Args:
        formula: A compositional formula representing the explanation.
        masks: A list of masks corresponding to the concepts in the formula.
        activations: A tensor of shape (N, H*W) where N is the number of samples.
    Returns:
        detection_accuracy (float): The detection accuracy.
    """
    formula_mask = concept_mask_utils.get_formula_mask(formula, masks)
    formula_mask = formula_mask.to(activations.device)
    return detection_accuracy(formula_mask, activations)

def samples_coverage(activations, segmentations):
    """Compute the samples coverage for the given activations and
    segmentations.
    Args:
        activations (torch.Tensor): A tensor of shape (N, H*W) where N is
            the number of sample.
        segmentations (torch.Tensor): A tensor of shape (N, H*W) where N is
        the number of sample.
    Returns:
        samples_coverage (float): The samples coverage.
    """
    samples_overlap = (
        torch.sum(activations & segmentations, 1, dtype=torch.int32) > 0
    )
    segmentation_in = torch.sum(segmentations, 1, dtype=torch.int32) > 0
    return torch.sum(samples_overlap) / torch.sum(segmentation_in)

def compute_samples_coverage_from_masks(*, formula, masks, activations):
    """Compute the samples coverage for the given formula, masks, and segmentations.
    Args:
        formula: A compositional formula representing the explanation.
        masks: A list of masks corresponding to the concepts in the formula.
        activations: A tensor of shape (N, H*W) where N is the number of samples.
    Returns:
        samples_coverage (float): The samples coverage.
    """ 
    formula_mask = concept_mask_utils.get_formula_mask(formula, masks)
    formula_mask = formula_mask.to(activations.device)
    return samples_coverage(formula_mask, activations)

@functools.lru_cache(maxsize=10)
def compute_hits(vector):
    """Compute the number of ones in the given vector.
    Args:
        vector (torch.Tensor): A tensor of shape (N, H*W) where N is the
            number of sample.
    Returns:
        hits (int): The number of ones in the given vector.
    """
    return torch.count_nonzero(vector)


def iou(vector1, vector2):
    """Compute the intersection over union between two vectors.
    Args:
        vector1 (torch.Tensor): A tensor of shape (N, H*W) where N is the
            number of sample.
        vector2 (torch.Tensor): A tensor of shape (N, H*W) where N is the
            number of sample.
    Returns:
        iou (float): The intersection over union between the two vectors.
    """
    intersection = torch.count_nonzero(vector1 & vector2)
    v1_size = compute_hits(vector1)
    v2_size = compute_hits(vector2)
    score = intersection / (v1_size + v2_size - intersection)
    return score

def compute_iou_from_masks(*, formula, masks, activations):
    """Compute the intersection over union for the given formula, masks, and segmentations.
    Args:
        formula: A compositional formula representing the explanation.
        masks: A list of masks corresponding to the concepts in the formula.
        activations: A tensor of shape (N, H*W) where N is the number of samples.
    Returns:
        iou (float): The intersection over union.
    """
    formula_mask = concept_mask_utils.get_formula_mask(formula, masks)
    formula_mask = formula_mask.to(activations.device)
    return iou(formula_mask, activations)