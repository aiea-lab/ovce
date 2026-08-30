
import os
import pickle

from tqdm import tqdm
import torch
from scipy import sparse

from legacy import heuristics

def get_mask_type(masks):
    """
    Returns the type of the masks.
    Args:
        masks (list): list of masks.
    Returns:
        Type of the masks.
    """
    if isinstance(masks[1], sparse.csr_matrix):
        return "csr"
    else:
        return "torch"


def extract_mask(index, masks, mask_type="torch"):
    """
    Extracts the mask from the list of masks.
    Args:
        index (int): index of the mask to extract.
        masks (list): list of masks.
        mask_type (str): type of the masks.
    Returns:
        Mask.
    """
    if mask_type == "csr":
        return torch.from_numpy(masks[index].toarray())
    else:
        return masks[index]
    
def get_areas_mask(masks, info_directory, device=torch.device("cpu")):
    """
    Returns the areas per sample of the masks for each atomic concept.
    Args:
        masks (list): list of masks.
        info_directory (str): directory where the information is stored.
        device (torch.device): device to use.
    Returns:
        List of areas of the masks.
    """
    areas = []
    file_concept_areas = f"{info_directory}/concept_areas_list.pkl"
    if os.path.exists(file_concept_areas):
        with open(file_concept_areas, "rb") as file:
            areas = pickle.load(file)
    else:
        mask_type = get_mask_type(masks)
        for concept in range(len(masks)):
            areas.append(
                torch.sum(
                    extract_mask(concept, masks, mask_type),
                    1,
                    dtype=torch.int32,
                )
            )
        with open(file_concept_areas, "wb") as file:
            pickle.dump(areas, file)
    for i in range(len(areas)):
        if areas[i] is not None:
            areas[i] = areas[i].to(device)
    return areas

def get_bounding_boxes(masks, mask_shape, info_directory, device):
    """ Returns the bounding boxes of the masks.

    Args:
        masks (list): list of masks.
        mask_shape (tuple): shape of a mask.
        info_directory (str): directory where to save/load the information.
        device (torch.device): device to use.

    Returns:
        list: list of bounding boxes of the masks.
        """
    overscribed = []
    file_path = f"{info_directory}/positive_rectangles.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            overscribed = pickle.load(file)
    else:
        mask_type = get_mask_type(masks)
        for concept in tqdm(
            range(len(masks)),
            total=len(masks),
            desc="Getting bounding box for masks",
        ):
            concept_masks = extract_mask(concept, masks, mask_type)
            concept_masks = torch.reshape(
                concept_masks, (-1, mask_shape[0], mask_shape[1])
            )
            overscribed.append(
                heuristics.get_overscribed_rectangles(concept_masks, mask_shape).numpy()
            )
        with open(file_path, "wb") as file:
            pickle.dump(overscribed, file)
    for i in range(len(overscribed)):
        overscribed[i] = torch.from_numpy(overscribed[i]).to(device)
    return overscribed

def get_concept_inscribed_masks(masks, mask_shape, info_directory, device):
    """ Returns the inscribed rectangles of the masks.
    
    Args:
        masks (list): list of masks.
        mask_shape (tuple): shape of a mask.
        info_directory (str): directory where to save/load the information.
        device (torch.device): device to use.

    Returns:
        list: list of inscribed rectangles of the masks.
    """
    inscribed = []
    file_path = f"{info_directory}/positive_inscripted.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            inscribed = pickle.load(file)
    else:
        mask_type = get_mask_type(masks)
        for concept in tqdm(
            range(len(masks)),
            total=len(masks),
            desc="Getting inscribed masks",
        ):
            concept_masks = extract_mask(concept, masks, mask_type)
            concept_masks = torch.reshape(
                concept_masks, (-1, mask_shape[0], mask_shape[1])
            ).to(device)
            inscribed.append(
                heuristics.get_inscribed_rectangles(concept_masks, device).numpy()
            )
        with open(file_path, "wb") as file:
            pickle.dump(inscribed, file)
    for i in range(len(inscribed)):
        inscribed[i] = torch.from_numpy(inscribed[i]).to(device)
    return inscribed

def get_masks_info(masks, info_directory, mask_shape, device):
    """ Returns the masks information useful for the heuristics.

    Args:
        masks (list): list of masks.
        info_directory (str): path to the directory where the information will be stored.
        mask_shape (tuple): shape of the masks.
        device (torch.device): device on which the tensors will be stored.

    Returns:
        tuple: tuple containing:
            - concept_areas (list): list of areas of the masks.
            - inscribed_rectangles (list): list of inscribed
                rectangles of the masks.
            - bounding_boxes (list): list of bounding boxes of the masks.
    """
    if not os.path.exists(info_directory):
        os.makedirs(info_directory)
    concept_areas = get_areas_mask(masks, info_directory, device)
    inscribed_rectangles = get_concept_inscribed_masks(
        masks, mask_shape=mask_shape, info_directory=info_directory, device=device
    )
    bounding_boxes = get_bounding_boxes(
        masks, mask_shape=mask_shape, info_directory=info_directory, device=device
    )
    masks_info = (concept_areas, (inscribed_rectangles, bounding_boxes))
    return masks_info