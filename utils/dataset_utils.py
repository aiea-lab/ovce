import logging
logger = logging.getLogger("detectron2")
logger.setLevel(logging.WARNING)

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_test_loader

from src.mapper import DatasetMapper
from config import CUB


def get_data_loader(dataset, batch_size, transforms=[]):
    """
    Get a data loader for the given dataset with specified batch size and transformations.
    Args:
        dataset (list): List of dataset dictionaries.
        batch_size (int): Number of samples per batch.
        transforms (list, optional): List of transformations to apply to the images. Defaults to an empty list.
    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """
    data_loader = build_detection_test_loader(
        dataset, mapper=DatasetMapper(
            is_train=False,
            # We need to resize the image to the same size used for the probed model
            augmentations=transforms,
            image_format="RGB",
            ), batch_size=batch_size)

    return data_loader


def get_dataset(dataset_name):
    """
    Get the dataset from the Detectron2 DatasetCatalog.
    Args:
        dataset_name (str): Name of the dataset registered in Detectron2 DatasetCatalog.
    Returns:
        list: List of dataset dictionaries.
    """
    if dataset_name in DatasetCatalog:
        return DatasetCatalog.get(dataset_name)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not suuported yet")

def get_class_names(*, dataset_name: str, custom_classes: list = None):
    """
    Get the class names for a given dataset from the Detectron2 MetadataCatalog.
    Args:
        dataset_name (str): Name of the dataset registered in Detectron2 MetadataCatalog.
        custom_classes (list, optional): List of custom class names. If provided, these will be used instead of the dataset's default classes.
    Returns:
        list: List of class names for the dataset.
    """
    if custom_classes is not None and isinstance(custom_classes, list):
        class_names = custom_classes
    if dataset_name == 'cub200':
        if custom_classes == 'granularity_0':
            class_names =  CUB.GRANULARITY_0
        elif custom_classes == 'granularity_1':
            class_names =  CUB.GRANULARITY_1
        elif custom_classes == 'granularity_2':
            class_names = CUB.GRANULARITY_2
        elif custom_classes == 'all':
            class_names = CUB.GRANULARITY_0 + CUB.GRANULARITY_1 + CUB.GRANULARITY_2 + [get_class_names(dataset_name='ade20k_150_test_sem_seg')]
        else:
            class_names = ['background']
            print(f"Warning: No custom classes provided for CUB200 dataset. Using default class: {class_names}")
    else:
        catalog = MetadataCatalog.get(dataset_name)
        if "stuff_classes" in catalog.__dict__:  
            class_names = catalog.stuff_classes.copy()
        else:
            class_names = catalog.thing_classes.copy()
    return class_names