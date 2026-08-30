import random
from typing import List, Tuple

import torch
import sklearn.cluster as scikit_cluster

from src.model_wrapper import Cub200Model, Place365Model, DenseNetPlace365, CvT, MaxViTModel, ConvNext, EfficientViTModel
from utils import vecquantile

def get_layer_activations(cfg):
    """
    Get the activations of a specific layer of a model for a given dataset.
    Args:
        cfg (settings.Settings): configuration settings
    Returns:
        torch.Tensor: activations
    """

    # Extract parameters from cfg
    weights = cfg.get_weights()
    dataset_name = cfg.get_dataset_name()
    layer_name= cfg.get_layer_name()
    activation_dir = cfg.get_activation_dir()
    model_name = cfg.get_model_name()
    device = cfg.get_device()
    model_name = cfg.get_model_name()

    # Load the appropriate model wrapper based on the model name and pretrained setting
    if model_name == 'resnet_cub200':
        model_wrapper = Cub200Model(weights=weights)
    elif model_name == 'cvt':
        model_wrapper = CvT()
    elif model_name == 'maxvit':
        model_wrapper = MaxViTModel()
    elif model_name == 'convnext':
        model_wrapper = ConvNext()
    elif model_name == 'efficientvit':
        model_wrapper = EfficientViTModel()
    elif model_name == 'densenet161':
            model_wrapper = DenseNetPlace365(model_name=model_name, weights=weights)
    elif model_name in ['resnet18', 'alexnet']:
        model_wrapper = Place365Model(model_name=model_name, weights=weights)
    else:
        raise ValueError("Pretrained model not supported")

    # We fix the batch size to 1 because the probed models is likely not able to handle detectron2 data
    # Thus the parsing of the data are effectively done 1 sample at a time. 
    model_wrapper.set_loader(dataset_name, batch_size=1)

    # Get the activations for the specified layer
    layer_activations = model_wrapper.get_layer_activations(layer_name, activation_dir, device=device)

    return layer_activations

def extract_random_units(layer_activations, random_units=0):
    """
    Extract a specified number of random units from the layer activations.
    Args:
        layer_activations (torch.Tensor): activations of the layer
        random_units (int): number of random units to extract
    Returns:
        list: indices of the selected units
    """
    if random_units > 0:
        if random_units > layer_activations.shape[1]:
            raise ValueError(f"random_units ({random_units}) cannot be greater than the number of units in the layer ({layer_activations.shape[1]})")
        selected_units = random.sample(
            range(layer_activations.shape[1]), random_units)
    else:
        selected_units = range(layer_activations.shape[1])
    return selected_units

def build_ranges_from_clusters(
        activations: torch.Tensor, clusters: List[int],
        num_clusters: int) -> List[tuple]:
    """Build activation ranges from clusters.

    Args:
        activations (torch.Tensor): Activations of the unit.
        clusters (List[int]): Clusters indexes of the activations.
        num_clusters (int): Number of clusters.

    Returns:
        activation_ranges (List[tuple]): Activation ranges for each cluster.
    """

    activations_ranges = []
    for label in range(num_clusters):
        cluster_activations = activations[clusters == label]
        lower_bound = torch.min(cluster_activations)
        upper_bound = torch.max(cluster_activations)
        activations_ranges.append((lower_bound.item(), upper_bound.item()))
    return activations_ranges

def quantile_threshold(
        layer_activations: torch.Tensor, quantile: float, *,
        avoid_zero: bool, batch_size=64, seed=1) -> torch.Tensor:
    """
    Determine thresholds for neuron activations for each neuron.

    Args:
        layer_activations (torch.Tensor): Activations of the layer.
        quantile (float): Quantile to use.
        avoid_zero (bool): Whether to remove zeros from the activations.
        batch_size (int): Batch size to use.
        seed (int): Seed to use for the quantile vector.

    Returns:
        thresholds (torch.Tensor): Thresholds for each neuron.
    """
    quant = vecquantile.QuantileVector(depth=1, seed=seed)
    for i in range(0, layer_activations.shape[0], batch_size):
        batch = layer_activations[i:i + batch_size]
        batch = batch.flatten().reshape(-1, 1)
        if avoid_zero:
            batch = batch[batch != 0].reshape(-1, 1)
        quant.add(batch)
    thresholds = quant.readout(1000)[:, int(1000 * (1 - quantile) - 1)]
    return torch.tensor(thresholds)

def compute_activation_ranges(
        activations: torch.Tensor, num_clusters: int, quantile: float) -> List[Tuple]:
    """Compute activation ranges for each unit.

    Args:
        activations (torch.Tensor): Activations of the unit.
        num_clusters (int): Number of clusters.
        quantile (float): Quantile to use to threshold the activations when num_clusters is set to 1.

    Returns:
        activation_ranges (List[tuple]): Activation ranges for each unit.
    """
    if num_clusters == 1:
        # Case vanilla compositional and netdissect range
        # Avoid zero is set to false like in the compositional paper
        threshold = quantile_threshold(
            activations, quantile=quantile, avoid_zero=False
        )
        activation_ranges = [(threshold, torch.tensor(float("inf")))]
    else:
        activations = activations.reshape(-1, 1)
        # Remove zeros from activations if there is a relu activation
        if torch.all(activations >= 0):
            activations = activations[activations > 0]
            activations = activations.reshape(-1, 1)
        # Compute activation ranges
        clusters = scikit_cluster.KMeans(
            n_clusters=num_clusters, random_state=0
            ).fit(activations)
        activation_ranges = build_ranges_from_clusters(
            activations, clusters.labels_, num_clusters)
    return activation_ranges

def compute_bitmaps(
        activations: torch.Tensor, activation_range: Tuple,
        mask_shape: List[int]) -> torch.Tensor:
    """Get the bitmaps of the unit.

    This function upsamples the activations to the original size of the
    image and then binarize them.
    Args:
        activations (torch.Tensor): Activations of the unit.
        activation_range (Tuple): Activation range of the unit.
        mask_shape (List[int]): Shape of the mask.

    Returns:
        bitmaps (torch.Tensor): Bitmaps of the unit.
    """
    lower, upper = activation_range
    upsampled_activations = torch.nn.functional.interpolate(
        activations.unsqueeze(1),
        size=mask_shape, mode='bilinear')
    upsampled_activations = upsampled_activations.squeeze(1)
    bitmaps = torch.where(
        (upsampled_activations > lower) & (upsampled_activations < upper),
        True, False)
    bitmaps = bitmaps.reshape(bitmaps.shape[0], -1)
    return bitmaps