"""Module containing the class for the settings."""
import os

import torch


class Settings:
    """
    Class that stores all the settings used in each run.
    """

    def __init__(
        self,
        *,
        dataset,
        model,
        layer,
        segmentor,
        configuration_name='std',
        parallel_concepts=80,
        batch_parsing=True,
        root_models="data/model",
        root_segmentations="data/cache/segmentations",
        root_activations="data/cache/activations",
        root_results="data/results",
        device="cuda",

    ):
        # This needs to be set before the other settings, as it is used to set the other settings
        self.__segmentor = segmentor
        self.__model_name = model
        self.__dataset_name = self.set_dataset_name(dataset)
        self.__root_segmentations = self.set_root_segmentations(root_segmentations)
        self.mask_config = self.set_mask_settings(mask_shape=self.get_mask_shape(), batch_parsing=batch_parsing, parallel_concepts=parallel_concepts)
        self.__root_activations = root_activations
        self.__root_results = root_results
        self.__root_models = root_models
        self.__device = self.set_device(device)
        self.__configuration_name = configuration_name
        self.__layer = self.set_layer(layer)


    
    def set_dataset_name(self, dataset):
        """
        Set the dataset name.
        Args:
            dataset (str): Name of the dataset.
        """
        if self.__segmentor == "human" and dataset == "cub200":
            dataset = "ade20k_150_test_sem_seg"
            print(f"Warning: Using {dataset} for human segmentor instead of cub200 since human does not support it.")
        return dataset

    def set_mask_settings(self, *, mask_shape, batch_parsing, parallel_concepts):
        return {
            "mask_shape": mask_shape,
            "batch_parsing": batch_parsing,
            "parallel_concepts": parallel_concepts
        }

    def set_layer(self, layer):
        """
        Set the layer to be used for the model.
        Args:
            layer (str): Name of the layer to be used for the model.
        """
        if self.get_model_name() in ['efficientvit', 'convnext', 'maxvit', 'cvt']:
            assert layer == 'stage3', f"Layer {layer} is not supported for model {self.get_model_name()}. Please use 'stage3' instead."

        return layer

    def get_mask_settings(self):
        return self.mask_config

    def get_configuration_name(self):
        """
        Returns the name of the configuration.
        """
        return self.__configuration_name

    def set_root_segmentations(self, root_segmentations):
        """
        Set the root directory where the segmentations are stored.
        Args:
            root_segmentations (str): Root directory where the segmentations are stored.
        """
        if os.path.exists(root_segmentations):
            return root_segmentations
        else:
            os.makedirs(root_segmentations, exist_ok=True)
            return root_segmentations

    def set_root_results(self, root_results):
        """
        Set the root directory where the results are stored.
        Args:
            root_results (str): Root directory where the results are stored.
        """
        if os.path.exists(root_results):
            return root_results
        else:
            os.makedirs(root_results, exist_ok=True)
            return root_results
        
    def set_root_activations(self, root_activations):
        """
        Set the root directory where the activations are stored.
        Args:
            root_activations (str): Root directory where the activations are stored.
        """
        if os.path.exists(root_activations):
            return root_activations
        else:
            os.makedirs(root_activations, exist_ok=True)
            return root_activations

    def get_activation_dir(self):
        """
        Returns the directory where the activations are stored.
        """
        path = os.path.join(self.__root_activations, self.__dataset_name, self.__model_name)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    def get_segmentations_dir(self):
        """
        Returns the directory where the segmentations are stored.
        """
        path = os.path.join(self.__root_segmentations, self.__dataset_name, self.__segmentor, self.__configuration_name, 'masks')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    def get_info_dir(self):
        """
        Returns the directory where the segmentations info is stored.
        """
        path = os.path.join(self.__root_segmentations, self.__dataset_name, self.__segmentor, self.__configuration_name, 'info')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    def get_results_dir(self):
        """
        Returns the directory where the results are stored.
        """
        path = os.path.join(self.__root_results, self.__dataset_name, self.__segmentor, self.__configuration_name, self.__model_name, self.__layer )
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    def get_info_dir(self):
        """
        Returns the directory where the segmentations info is stored.
        """
        path = os.path.join(self.__root_segmentations, self.__dataset_name, self.__segmentor, self.__configuration_name, 'info')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    def get_segmentor_name(self):
        """
        Returns the name of the segmentor.
        """
        return self.__segmentor

    def set_device(self, device):   
        """
        Set the device to be used for the model.
        Args:
            device (str): Device to be used for the model.
        """
        if "cuda" in device:
            if torch.cuda.is_available():
                return torch.device(device)
            else:
                raise ValueError("CUDA is not available. Please use CPU instead.")
        else:
            return torch.device(device)

    def get_weights(self):
        """
        Returns the path to the pretrained weights of the model.
        """

        if self.__model_name == "densenet161":
            model_file_name = (
                "whole_densenet161_places365_python36.pth.tar"
            )
            return self.__root_models + "/zoo/" + model_file_name
        elif self.__model_name in ["resnet18", "alexnet"]:
            model_file_name = f"{self.__model_name}_places365.pth.tar"
            return self.__root_models + "/zoo/" + model_file_name
        elif self.__model_name == "resnet_cub200":
            return self.__root_models + "/other/" + "bird_res50.tar"
        else:
            raise ValueError("Pretrained weights not available for this model. Please use a different model or provide your own weights.")

    def get_dataset_name(self):
        """
        Returns the name of the dataset.
        """
        return self.__dataset_name

    def get_device(self):
        """
        Returns the device to be used for the model.
        """
        return self.__device

    def get_model_name(self):
        """
        Returns the name of the model.
        """
        return self.__model_name

    def get_layer_name(self):
        """
        Returns the name of the layer to be used for the model.
        """
        return self.__layer


    def get_mask_shape(self):
        """
        Returns the shape of the mask.
        """
        return (112, 112)

