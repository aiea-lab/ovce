"""
This module contains the implementation of the algorithms described in
the paper. It includes NetDissect, the compositional explanations algorithm
and the heuristic search.
"""
from collections import Counter
import torch

from . import heuristic_search
from utils import common_utils
from src import metrics
# from . import utils
# from . import metrics


def get_netdissect_scores(bitmaps, masks):
    """Compute the NetDissect score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H, W).
        candidate_concepts (list): A list of candidate concepts.

    Returns:
        netdissect_rank (dict): A dictionary of concept scores. Each score is
            a float.
    """
    netdissect_rank = {}
    mask_type = "tensor" if isinstance(masks[1], torch.Tensor) else "sparse"
    candidate_concepts = range(len(masks))
    for concept in candidate_concepts:
        if mask_type == "tensor":
            concept_mask = masks[concept]
        else:
            concept_mask = common_utils.sparse_to_torch(masks[concept])
        concept_mask = concept_mask.to(bitmaps.device)
        concept_iou = metrics.iou(concept_mask, bitmaps)
        netdissect_rank[concept] = concept_iou.item()

    return netdissect_rank


def get_augmented_netdissect_scores(bitmaps, masks):
    """Compute the NetDissect score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H, W).
        candidate_concepts (list): A list of candidate concepts.

    Returns:
        netdissect_rank (dict): A dictionary of concept scores. Each score is
            a float.
    """
    netdissect_rank = {}
    areas = []
    mask_type = "tensor" if isinstance(masks[1], torch.Tensor) else "sparse"
    candidate_concepts = range(len(masks))
    for concept in candidate_concepts:
        if mask_type == "tensor":
            concept_mask = masks[concept]
        else:
            concept_mask = common_utils.sparse_to_torch(masks[concept])
        concept_mask = concept_mask.to(bitmaps.device)
        concept_iou = metrics.iou(concept_mask, bitmaps)
        intersection_area = (concept_mask & bitmaps).sum(
            dim=1, dtype=torch.int32
        )
        netdissect_rank[concept] = concept_iou
        areas.append(intersection_area)
    return netdissect_rank, areas


def get_heuristic_scores(
    segmentations,
    activation_masks,
    *,
    segmentations_info=None,
    max_size_mask,
    beam_size=5,
    length=3,
    mask_shape=None,
    device=torch.device("cpu")
):
    """Compute the heuristic score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        segmentations (dict): A dictionary of concept masks. Each mask is a
            tensor of shape (N, H, W) where N is
            the number of sample.
        activation_masks (torch.Tensor): A tensor of shape (N, H, W) where N is
            the number of sample.
        segmentations_info (dict): A dictionary of information about the
            segmentations. None can be used only when the heuristic is none.
        max_size_mask (int): The maximum size of the masks.
        beam_size (int): The beam size for the search.
        length (int): The length of the search.
        mask_shape (tuple): The shape of the masks.
        device (torch.device): The device to use for the computation.

    Returns:
        best_label (int): The label of the best concept.
        best_iou (float): The IOU of the best concept.
        visited (int): The number of visited nodes.
    """

    # Compute commong parameters
    num_hits = activation_masks.sum()

    if length == 1:
        # vanilla netdissect
        rank = get_netdissect_scores(activation_masks, segmentations)
        best_label = Counter(rank).most_common(1)[0][0]
        best_iou = Counter(rank).most_common(1)[0][1]
        return best_label, best_iou, 0

    sample_activation_areas = activation_masks.sum(1)
    netdissect_scores, intersect_areas = get_augmented_netdissect_scores(
        activation_masks, segmentations
    )
    heuristic_info = (
        segmentations_info,
        sample_activation_areas,
        intersect_areas,
    )
    best_label, best_iou, visited = heuristic_search.perform_heuristic_search(
        netdissect_scores,
        segmentations,
        activation_masks,
        heuristic_info,
        num_hits,
        beam_size=beam_size,
        length=length,
        max_size_mask=max_size_mask,
        mask_shape=mask_shape,
        device=device,
    )
    return best_label, best_iou, visited
