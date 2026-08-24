import os

from tqdm import tqdm
import torch
import numpy as np
import scipy.sparse as sparse

from utils import common_utils
from src import formula as F

def compute_concept_masks(segmentations, *, concept_index, mask_shape):
    """
    Compute the concept masks for a given concept from the available segmentations.
    Args:
        segmentations (torch.Tensor): Tensor of shape (batch_size, height, width) containing the segmentations.
        concept_index (int): Index of the concept for which to compute the masks.
        mask_shape (tuple): Tuple of (height, width) specifying the desired mask shape.
    Returns:
        torch.Tensor: Tensor of shape (num_samples, height, width) containing the concept masks
    """
    concept_mask_batch = segmentations == concept_index
    # Reshape the concept mask to desired mask shape to reduce memory usage
    reshaped_concept_mask_batch  = reshape_concept_mask(concept_mask_batch, mask_shape)
    return reshaped_concept_mask_batch

def compute_bulk_concept_masks(segmentation_list, *, concept_indices, mask_shape):
    """
    Compute the concept masks for a given concept from the available segmentations.
    Args:
        segmentation_list (list): List of tensors containing the segmentations for each batch.
        concept_indices (list): List of indices of the concepts for which to compute the masks.
        mask_shape (tuple): Tuple of (height, width) specifying the desired mask shape.
    Returns:
        torch.Tensor: Tensor of shape (num_samples, height, width) containing the concept masks
    """
    concept_masks = [ [] for _ in range(len(concept_indices)) ]
    for segmentations in segmentation_list:
        for concept_index in concept_indices:
            concept_mask = compute_concept_masks(segmentations, concept_index=concept_index, mask_shape=mask_shape)
            concept_masks[concept_indices.index(concept_index)].append(concept_mask)
    concept_masks = [torch.cat(concept_mask, 0) for concept_mask in concept_masks]
    return concept_masks

def reshape_concept_mask(concept_mask, mask_shape):
    """
    Reshape the concept mask to the desired mask shape.
    Args:
        concept_mask (torch.Tensor): Tensor of shape (batch_size, height, width) containing the concept mask.
        mask_shape (tuple): Tuple of (height, width) specifying the desired mask shape.
    Returns:
        torch.Tensor: Tensor of shape (batch_size, height, width) containing the reshaped concept mask.
    """
    reshaped_concept_mask = torch.nn.functional.interpolate(
        concept_mask.float().unsqueeze(0),
        size=mask_shape,
        mode="nearest",
    ).bool().squeeze(0)
    return reshaped_concept_mask

def remove_unmeaningful_masks(*, concept_masks, concept_labels, ignore_concepts):
    """
    Zero out concept masks that are unmeaningful based on the ignore list.
    Args:
        concept_masks (list): List of tensors containing the concept masks for each concept.
        concept_labels (list): List of labels for each concept.
        ignore_concepts (list): List of indices of concepts to ignore.
    Returns:
        list: List of tensors containing the concept masks with unmeaningful masks zeroed out.
    """

    for index, label in enumerate(concept_labels):
        if label in ignore_concepts:
            concept_masks[index] = torch.zeros_like(concept_masks[index])
    return concept_masks


def get_full_disjoint_matrix(num_concepts):
    """
    Get a full disjoint matrix of shape (num_concepts, num_concepts) with all entries set to 1.
    Args:
        num_concepts (int): Number of concepts.
    Returns:
        torch.Tensor: Full disjoint matrix of shape (num_concepts, num_concepts) with all entries set to 1.
    """
    return torch.ones((num_concepts, num_concepts), dtype=torch.int8)

def compute_disjoint_matrix(*, info_dir, concept_masks, block_size=32):
    """
    Compute the disjoint matrix for the given concept masks.
    Args:
        info_dir (str): Directory to save the disjoint matrix.
        concept_masks (list): List of tensors containing the concept masks for each concept.
        block_size (int): Number of concepts to process per block when comparing pairs.
    Returns:
        torch.Tensor: Disjoint matrix of shape (num_concepts, num_concepts) indicating whether each pair of concepts is disjoint (1) or not (0).
    """
    if os.path.exists(os.path.join(info_dir, "disjoint_matrix.pt")):
        print(f"Loading disjoint matrix from {info_dir}")
        return torch.load(os.path.join(info_dir, "disjoint_matrix.pt"))

    print(f"Computing disjoint matrix for {len(concept_masks)} concepts. This may take a while...")
    num_concepts = len(concept_masks)
    if num_concepts == 0:
        return torch.zeros((0, 0), dtype=torch.int8)

    def _to_sparse_row(mask):
        if sparse.issparse(mask):
            return mask.reshape(1, -1).tocsr().astype(np.bool_, copy=False)
        if torch.is_tensor(mask):
            dense_row = mask.reshape(1, -1).bool().cpu().numpy()
            return sparse.csr_matrix(dense_row)
        raise TypeError(f"Unsupported mask type: {type(mask)}")

    def _to_sparse_block(mask_list):
        return sparse.vstack([_to_sparse_row(mask) for mask in mask_list], format="csr")

    # Keep output behavior compatible with previous code (CPU int8 matrix).
    disjoint_matrix = torch.zeros((num_concepts, num_concepts), dtype=torch.int8)

    num_blocks = (num_concepts + block_size - 1) // block_size
    for bi in tqdm(range(num_blocks), desc="Computing disjoint matrix"):
        i_start = bi * block_size
        i_end = min(i_start + block_size, num_concepts)
        masks_i = _to_sparse_block(concept_masks[i_start:i_end])

        for bj in range(bi, num_blocks):
            j_start = bj * block_size
            j_end = min(j_start + block_size, num_concepts)
            masks_j = _to_sparse_block(concept_masks[j_start:j_end])

            if masks_i.shape[1] != masks_j.shape[1]:
                raise ValueError(
                    f"Inconsistent flattened mask sizes: {masks_i.shape[1]} vs {masks_j.shape[1]}"
                )

            # overlap_counts[p, q] > 0 means concepts p and q share at least one active element.
            overlap_counts = masks_i.astype(np.int8) @ masks_j.astype(np.int8).T
            block_disjoint = torch.from_numpy((overlap_counts.toarray() == 0).astype(np.int8))

            disjoint_matrix[i_start:i_end, j_start:j_end] = block_disjoint
            if bi != bj:
                disjoint_matrix[j_start:j_end, i_start:i_end] = block_disjoint.T

    disjoint_matrix.fill_diagonal_(0)

    # Save the disjoint matrix
    torch.save(disjoint_matrix, os.path.join(info_dir, "disjoint_matrix.pt"))
    return disjoint_matrix

def get_formula_mask(f, masks, optional_masks=None):
    """
    Function to return a mask for a given formula.
    Args:
        f (src.formula.Formula): formula.
        masks (list): list of masks.
        optional_masks (dict): dictionary of additional masks (beam masks).
    Returns:
        Formula's Mask.
    """
    if optional_masks is not None and f in optional_masks.keys():
        mask = optional_masks[f]
        if isinstance(mask, sparse.csr.csr_matrix):
            return common_utils.sparse_to_torch(mask)
        else:
            return mask
    if isinstance(f, F.Leaf):
        mask = masks[f.val]
        if isinstance(mask, sparse.csr.csr_matrix):
            return common_utils.sparse_to_torch(mask)
        else:
            return mask
    elif isinstance(f, F.Or):
        masks_l = get_formula_mask(f.left, masks, optional_masks)
        masks_r = get_formula_mask(f.right, masks, optional_masks)
        return masks_l | masks_r
    elif isinstance(f, F.And):
        masks_l = get_formula_mask(f.left, masks, optional_masks)
        masks_r = get_formula_mask(f.right, masks, optional_masks)
        return masks_l & masks_r
    elif isinstance(f, F.Not):
        return ~get_formula_mask(f.val, masks, optional_masks)
    elif isinstance(f, int):
        mask = masks[f]
        if isinstance(mask, sparse.csr.csr_matrix):
            return common_utils.sparse_to_torch(mask)
        else:
            return mask
    else:
        raise ValueError(f"Unknown formula type {type(f)}")

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

def merge_human_masks_with_mapping(masks, masks_labels, mapping):
    """
    Merges masks based on a provided mapping.
    Args:
        masks: List of masks to be merged.
        masks_labels: Corresponding labels for the masks.
        mapping: A dictionary containing the mapping of concepts.
    Returns:
        merged_masks: List of merged masks.
        merged_labels: Corresponding labels for the merged masks.
    """

    for label in mapping.keys():
        # Get the label to merge into
        merged_into = mapping[label]
        if label == merged_into:
            continue  # Skip if the label is the same as the merged_into label

        # Check if the new labels is an abstraction not present in the segmentor concept labels
        if merged_into not in masks_labels:
            masks.append(torch.zeros_like(masks[0]))
            masks_labels.append(merged_into)

        # Get the index of the labels
        index_label = masks_labels.index(label)
        index_merged_into = masks_labels.index(merged_into)

        # Merge the masks
        masks[index_merged_into] = torch.logical_or(masks[index_label], masks[index_merged_into])

        # Zero out the mask to be merged
        masks[index_label] = torch.zeros_like(masks[index_label])

    return masks, masks_labels