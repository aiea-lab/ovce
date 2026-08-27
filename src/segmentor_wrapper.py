import os
from tqdm import tqdm

import torch
torch.set_float32_matmul_precision("high") # Needed for CAT-Seg
import scipy.sparse as sparse
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer

from utils import dataset_utils
from utils import concept_mask_utils
from utils import config as config_collection


class Detectron2Segmentor:
    def __init__(self) -> None:
        # Supported Datasets: ade20k_150, ade20k_full, pascal20, cityscapes, mapillary, coco-stuff, pascal_context_459, cub200
        self.data_loader = None
        self.concept_labels = None

    def extract_segmentations(self, data):
        """
        Extracts the segmentations from the data returned by the dataloader.
        Args:
            data (list): List of dictionaries returned by the dataloader.
        Returns:
            torch.Tensor: Tensor of shape (batch_size, height, width) containing the segmentations.
        """
        raise NotImplementedError

    def get_dataset_segmentations(self):
        """
        Get the segmentations for the entire dataset.
        Returns:
            list: List of tensors containing the segmentations for each batch.
        """
        dataset_segmentations = []
        with torch.no_grad():
            for data in tqdm(self.data_loader, desc="Computing Whole Dataset Segmentations"):
                segmentations = self.extract_segmentations(data)
                dataset_segmentations.append(segmentations.cpu())
        return [dataset_segmentations]

    def set_data_loader(self, dataset_name, min_size, max_size):
        """
        Set the data loader for the model.
        Args:
            dataset_name (str): Name of the dataset to be used for the model.
            min_size (int): Minimum size of the input images.
            max_size (int): Maximum size of the input images.
        Returns:
            None
        """
        
        dataset = dataset_utils.get_dataset(dataset_name)
        augmentation = [T.ResizeShortestEdge(min_size, max_size,sample_style='choice')] 

        # Note that we use batch size 1 because the input and output (segmentations) may have different sizes. 
        # We want to avoid padding, and this choice makes easier the stacking in extract_segmentations
        self.data_loader = dataset_utils.get_data_loader(
            dataset, transforms=augmentation, batch_size=1,
        )

    def get_cached_batch_segmentations(self, segmentations_dir):
        """
        Get the segmentations for the entire dataset, yielding one batch at a time.
        Args:
            segmentations_dir (str): Directory where the segmentations will be saved.
        Yields:
            torch.Tensor: Tensor of shape (batch_size, height, width) containing the segmentations for the current batch.
        """
        cache_dir = os.path.join(segmentations_dir, "cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with torch.no_grad():
            for index, data in enumerate(tqdm(self.data_loader, desc="Computing Batch Segmentations")):
                batch_output_path = os.path.join(cache_dir, f"batch_{index}.pt")
                if os.path.exists(batch_output_path):
                    # Load batch
                    with open(batch_output_path, "rb") as f:
                        segmentations = torch.load(f)
                else:
                    # Compute batch
                    segmentations = self.extract_segmentations(data)
                    segmentations = segmentations.cpu()

                    # Save the segmentations for the current batch to disk
                    with open(batch_output_path, "wb") as f:
                        torch.save(segmentations, f)
                        f.flush()
                        os.fsync(f.fileno())
                yield [segmentations]

    def remove_cached_batch_segmentations(self, segmentations_dir):
        """
        Remove the cached segmentations for the entire dataset.
        Args:
            segmentations_dir (str): Directory where the segmentations are saved.
        Returns:
            None
        """
        cache_dir = os.path.join(segmentations_dir, "cache")
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(cache_dir)

    def save_segmentations(self, *, output_dir, parallel_concepts, mask_shape, batch_parsing=True):
        """
        Save the segmentor outputs (masks) to disk for the specified output directory, parallel concepts, mask shape, and batch parsing option.
        Args:
            output_dir (str): Directory where the segmentations will be saved.
            parallel_concepts (int): Number of concepts to parse in parallel.
            mask_shape (tuple): Shape of the masks to be saved.
            batch_parsing (bool): Whether to parse the dataset in batches or as a whole. Defaults to True.
        Returns:
            None
        """
        # Precompute segmentations for the entire dataset if batch parsing is disabled
        if batch_parsing is False:
            print("Precomputing segmentations for the entire dataset since batch parsing is disabled.")
            precomputed_segmentations = self.get_dataset_segmentations()

        # Total number of concepts
        num_concepts = len(self.concept_labels)

        # Concepts we will parse in parallel to speed up the computation
        num_parallel_concepts = min(parallel_concepts, num_concepts)  # Limit the number of parallel concepts to 8
        ranges = range(0, num_concepts, num_parallel_concepts)
        start = 0
        for start in tqdm(ranges, desc=f"Saving {num_parallel_concepts} Concept Masks per iteration. Tot: {num_concepts} concepts"):
            # This line distinguishes between batch parsing and dataset parsing. 
            # In batch parsing, we process one batch at a time, while in dataset parsing, we process the entire dataset at once.
            # Batch parsing is slower but uses less memory. Dataset parsing is faster but uses more memory. 
            if batch_parsing:
                parsed_segmentations = self.get_cached_batch_segmentations(output_dir)
            else:
                parsed_segmentations = precomputed_segmentations

            # Compute concepts to parse at this iteration
            end = min(start + num_parallel_concepts, num_concepts)
            concept_indices = list(range(start, end))
            concept_labels_batch = [self.concept_labels[i] for i in concept_indices]

            # Extract segmentation batches
            concept_masks_set = [ [] for _ in range(len(concept_indices)) ]
            with torch.no_grad():
                for segmentations_batch in parsed_segmentations:
                    concept_masks_batch = concept_mask_utils.compute_bulk_concept_masks(segmentations_batch, concept_indices=concept_indices, mask_shape=mask_shape)
                    for i, concept_mask in enumerate(concept_masks_batch):
                        concept_masks_set[i].append(concept_mask)

            del parsed_segmentations # Free memory after processing the segmentations for this iteration
                
            # Concatenate the concept masks for each concept and save them to disk
            for i, concept_label in enumerate(concept_labels_batch):
                concept_masks = torch.cat(concept_masks_set[i], 0)
                concept_masks = torch.reshape(
                        concept_masks, (concept_masks.shape[0], -1)).detach().cpu()
                sparse_masks = sparse.csr_matrix(concept_masks.numpy())
                with open(os.path.join(output_dir, f"{concept_label}.npz"), "wb") as f:
                    sparse.save_npz(f, sparse_masks)
            del concept_masks_set # Free memory after saving the concept masks to disk

        # Remove the cached segmentations after saving the concept masks to disk
        self.remove_cached_batch_segmentations(output_dir)
        
    def load_concept_masks(self, segmentations_dir):
        """
        Load the concept masks from disk.
        Args:
            output_dir (str): Directory where the concept masks are saved.
        Returns:
            list: List of torch.Tensor objects containing the concept masks.
            None if any concept mask is missing.
        """
        concept_labels = self.concept_labels

        # Check that the number of concept masks in the directory matches the number of concept labels. 
        # This is needed because removing concepts implies a difference in the distribution of the segmentors output. Therefore, we need to recompute them
        num_files_in_dir = len(os.listdir(segmentations_dir))
        if num_files_in_dir != len(concept_labels):
            return None

        print(f"Loading concept masks for {len(concept_labels)} concepts from {segmentations_dir}")
        # Try to load all concepts masks
        concept_masks = []
        # Check if all concept masks are already saved to disk. Otherwise, 
        for concept_label in concept_labels:
            output_path = os.path.join(segmentations_dir, f"{concept_label}.npz")
            if not os.path.exists(output_path):
                # If any concept mask is missing, return None to indicate that the concept masks need to be regenerated.
                return None
            else:
                with open(output_path, "rb") as f:
                    sparse_masks = sparse.load_npz(f)
                concept_masks.append(torch.from_numpy(sparse_masks.toarray()).bool())
        return concept_masks
        
class Detectron2GroundTruth(Detectron2Segmentor):
    """
    A class to handle ground truth segmentations from a dataset using Detectron2.
    This class extracts segmentations directly from the dataset without using any pre-trained model.
    """
    def __init__(self, dataset_name) -> None:

        self.set_data_loader(dataset_name, min_size=512, max_size=2048)
        self.concept_labels = dataset_utils.get_class_names(dataset_name=dataset_name)
              
        
    def extract_segmentations(self, data):
        """
        Extracts the ground truth segmentations from the data returned by the dataloader.
        Args:
            data (list): List of dictionaries returned by the dataloader.
        Returns:
            torch.Tensor: Tensor of shape (batch_size, height, width) containing the ground truth segmentations.
        """
        segmentations = torch.stack([data[0]["sem_seg"] for i in range(len(data)) ])
        return segmentations

class Detectron2Model(Detectron2Segmentor):
    """
    A class to handle segmentations from a pre-trained (open-vocabulary) model using Detectron2.
    This class extracts segmentations from the model's output for a given dataset."""
    def __init__(self, *,  dataset_name, mask_shape) -> None:
        super().__init__()
        self.mask_shape = mask_shape
        

    def extract_segmentations(self, data):
        """
        Extracts the segmentations from the data returned by the dataloader using the open-vocabulary model.
        Args:
            data (list): List of dictionaries returned by the dataloader.
        Returns:
            torch.Tensor: Tensor of shape (batch_size, height, width) containing the segmentations.
        """
        
        output = self.model(data)
        segmentations = torch.stack([output[i]["sem_seg"] for i in range(len(output)) ])
        segmentations = torch.argmax(segmentations, dim=1)

        del output # Remove the output to free up memory, as it is no longer needed after extracting the segmentations.
        return segmentations


    def set_model(self, cfg, device=torch.device("cpu")):
        """
        Set the model for the segmentor using the provided configuration.
        Args:
            cfg (detectron2.config.CfgNode): Configuration for the model.
            device (torch.device): Device to load the model onto. Defaults to CPU.
        """
        model_weights = cfg.MODEL.WEIGHTS
        directory = os.path.dirname(model_weights)
        self.model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(self.model, save_dir=directory).resume_or_load(
            model_weights, resume=False
        )
        self.model.eval()
        self.model = self.model.to(device)

    def init_from_config(self, cfg, *, dataset_name, custom_classes=None, device=torch.device("cpu")):
        """
        Initialize the segmentor from the provided configuration.
        Args:
            cfg (detectron2.config.CfgNode): Configuration for the model.
            dataset_name (str): Name of the dataset to be used for the model.
            custom_classes (list, optional): List of custom classes to be used for the model.
            device (torch.device): Device to load the model onto. Defaults to CPU.
        """
        # Set data
        self.set_data_loader(dataset_name, cfg.INPUT.MIN_SIZE, cfg.INPUT.MAX_SIZE)
        # Set the model
        self.set_model(cfg, device=device)
        class_names = dataset_utils.get_class_names(dataset_name=dataset_name, custom_classes=custom_classes)
        self.set_concept_labels(class_names)


    def set_concept_labels(self, class_names):
        raise NotImplementedError
    
class CATSeg(Detectron2Model):
    def __init__(self, dataset_name, *, mask_shape=None, custom_classes=None, device=torch.device("cpu")) -> None:
        # Register model
        import segmentors.cat_seg
        super().__init__(dataset_name=dataset_name, mask_shape=mask_shape)
        cfg = config_collection.cat_seg_config()
        self.init_from_config(cfg, dataset_name=dataset_name, custom_classes=custom_classes, device=device)

    def set_concept_labels(self, class_names):
        self.model.sem_seg_head.predictor.set_text_embedding(class_names)
        self.concept_labels = class_names

class Masqclip(Detectron2Model):
    def __init__(self, dataset_name, *, mask_shape=None, custom_classes=None, device=torch.device("cpu")) -> None:        
        # Register model
        import segmentors.masqclip.masq_tuning
        super().__init__(dataset_name=dataset_name, mask_shape=mask_shape)
        cfg = config_collection.masqclip_config()
        self.init_from_config(cfg, dataset_name=dataset_name, custom_classes=custom_classes, device=device)

        # Masqclip does not support deterministic algorithms due to cuDNN
        torch.use_deterministic_algorithms(False)

        # Move the masqclip model to the specified device after initialization
        self.model.masqclip = self.model.masqclip.to(device)
          
    def set_concept_labels(self, class_names):
        self.model.masqclip.set_text_embedding(class_names)
        self.concept_labels = class_names

class SED(Detectron2Model):
    def __init__(self, dataset_name, *, mask_shape=None, custom_classes=None, device=torch.device("cpu")) -> None:
        # Register model
        import segmentors.sed
        super().__init__(dataset_name=dataset_name, mask_shape=mask_shape)
        cfg = config_collection.sed_config()
        self.init_from_config(cfg, dataset_name=dataset_name, custom_classes=custom_classes, device=device)

        # SED does not support deterministic algorithms due to cuDNN
        torch.use_deterministic_algorithms(False)

    def set_concept_labels(self, class_names):
        self.model.sem_seg_head.predictor.set_text_embedding(class_names)
        self.concept_labels = class_names

class SCAN(Detectron2Model):
    def __init__(self, dataset_name, *, mask_shape=None, custom_classes=None, device=torch.device("cpu")) -> None:
        # Register model
        import segmentors.scan
        super().__init__(dataset_name=dataset_name, mask_shape=mask_shape)
        cfg = config_collection.scan_config()
        self.init_from_config(cfg, dataset_name=dataset_name, custom_classes=custom_classes, device=device)

        # SCAN does not support deterministic algorithms due to cuDNN
        torch.use_deterministic_algorithms(False)

    def set_concept_labels(self, class_names):
        self.model.set_text_embedding(class_names)
        self.concept_labels = class_names

class OpenSeeD(Detectron2Model):
    def __init__(self, dataset_name, *, mask_shape=None, custom_classes=None, device=torch.device("cpu")) -> None:
        # Register model
        import segmentors.openseed
        super().__init__(dataset_name=dataset_name, mask_shape=mask_shape)
        cfg = config_collection.openseed_config()
        # In the case of OpenSeeD, they use different max and min sizes for different datasets
        if 'city' in dataset_name:
            cfg.INPUT.MIN_SIZE = 1024
            cfg.INPUT.MAX_SIZE = 2048
        elif 'coco' in dataset_name:
            cfg.INPUT.MIN_SIZE = 800
            cfg.INPUT.MAX_SIZE = 1333
        else:
            cfg.INPUT.MIN_SIZE = 640
            cfg.INPUT.MAX_SIZE = 2560

        self.init_from_config(cfg, dataset_name=dataset_name, custom_classes=custom_classes, device=device)

        # OpenSeeD does not support deterministic algorithms due to cuDNN
        torch.use_deterministic_algorithms(False) 

    def set_model(self, cfg, device=torch.device("cpu")):
        import segmentors.openseed as openseed
        from segmentors.openseed.BaseModel import BaseModel as OpenSeeDBaseModel

        # OpenSeeD uses a different way to load the model, so we override the set_model method here.
        model_weights = cfg.MODEL.WEIGHTS
        directory = os.path.dirname(model_weights)
        self.model = OpenSeeDBaseModel(cfg, openseed.build_model(cfg))
        DetectionCheckpointer(self.model, save_dir=directory).resume_or_load(
            model_weights, resume=False
        )
        self.model.eval()
        self.model = self.model.to(device)

    def set_concept_labels(self, class_names):
        self.model.model.sem_seg_head.predictor.lang_encoder.set_text_embeddings(class_names)
        self.concept_labels = class_names

    def extract_segmentations(self, data):
        # OpenSeeD uses a different way to extract segmentations, so we override the extract_segmentations method here.
        device_type = 'cuda' if next(self.model.parameters()).is_cuda else 'cpu'
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            outputs = self.model(data, inference_task="sem_seg")
        segmentations = torch.stack([outputs[i]["sem_seg"] for i in range(len(outputs)) ])
        segmentations = torch.argmax(segmentations, dim=1)
        return segmentations

class Mask2Former(Detectron2Model):
    def __init__(self, dataset_name, *, mask_shape=None, custom_classes=None, device=torch.device("cpu")) -> None:
        # Register model
        import segmentors.mask2former
        super().__init__(dataset_name=dataset_name, mask_shape=mask_shape)
        cfg = config_collection.mask2former_config()
        self.init_from_config(cfg, dataset_name=dataset_name, custom_classes=custom_classes, device=device)

       # Mask2Former does not support deterministic algorithms due to cuDNN
        torch.use_deterministic_algorithms(False)

    def set_concept_labels(self, class_names):
        print("Warning: Mask2Former is a closed vocabulary model, so we ignore the custom classes and use the class names used for training as the concept labels.")
        # Because Mask2Former is a closed vocabulary model, we use the class names used for training as the concept labels. We ignore the custom classes provided by the user.
        self.concept_labels = dataset_utils.get_class_names(dataset_name='coco_2017_train_panoptic', custom_classes=[])
        print(f"Using concept labels: {self.concept_labels}")