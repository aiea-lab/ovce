import os 

import torchvision
import torch
import numpy as np
import timm
from transformers import AutoFeatureExtractor, CvtForImageClassification

from utils import dataset_utils
from DifferentiableSVD.model_init import Newmodel as CUBmodel 
from DifferentiableSVD.src.representation import SEB

class ModelWrapper:
    """
    Abstract class that wraps a model and provides methods to compute activations.
    """
    
    def __init__(self, *args, **kwargs):
        self.model = None
        self.data_loader = None
        self.transformation = None
        self.iter_data = None

    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    # Reference: https://github.com/jayelm/compexp/blob/master/vision/loader/model_loader.py
    def hook(self, hook_fn, feature_names):
        """
        Register a hook to a model.

        Args:
            hook_fn (function): hook function
            feature_names (list): list of feature names

        Returns:
            list: list of handles
        """
        handles = []
        for name in feature_names:
            if isinstance(name, list):
                # Iteratively retrive the module
                hook_model = self.model
                for n in name:
                    hook_model = hook_model._modules.get(n)
            else:
                hook_model = self.model._modules.get(name)
            if hook_model is None:
                raise ValueError(f"Couldn't find feature {name}")
            handles.append(hook_model.register_forward_hook(hook_fn))
        return handles
       
    @torch.no_grad()
    def compute_activations( self,
            layer: str, device: torch.device):
        """Retrieve the activations of a given layer feeding the model with the
        images in the loader.

        Args:
            layer (str): name of the layer
            device (torch.device): device to use

        Returns:
            torch.Tensor: activations
        """
        # Move the model to the device
        original_device = next(self.model.parameters()).device
        self.model = self.model.to(device)
        temp_activations = []

        def hook_feature(module, inp, output):
            temp_activations.append(output.data.cpu())

        handles = self.hook(hook_feature, [layer])

        activations = []
        for data in self.data_loader:
            # Transformations
            images = self.iter_data(data)

            if len(images.shape) == 4:
                images = images.squeeze(0)

            if self.transformation is not None:
                images = self.transformation(images)
            if len(images.shape) == 3:
                images = images.unsqueeze(0)

            
            # Move to GPU
            images = images.to(device)

            # Forward pass
            _ = self.model(images)

            # collect data
            activations.append(temp_activations[0])

            # Empty the temp list
            del temp_activations[:]
            temp_activations = []

        # Move the model back the original device
        self.model = self.model.to(original_device)

        # Remove the hooks
        for handle in handles:
            handle.remove()

        
        activations =  torch.cat(activations)

        return activations

    def set_loader(self, dataset_name, batch_size):
        """
        Set the data loader for the model wrapper.
        Args:   
            dataset_name (str): name of the dataset
            batch_size (int): size of the batches to use
        """

        # Get dataset and data loader
        dataset = dataset_utils.get_dataset(dataset_name)
        self.data_loader = dataset_utils.get_data_loader(
            dataset, batch_size=batch_size)

        # Add transformations
        self.transformation = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((self.input_size, self.input_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Set the function to extract the images from the data loader. Since we use detectron2, we need to extract the image from the dict
        self.iter_data = lambda x: x[0]['image']


    
    def get_layer_activations(self,  layer, dir_activations, device):
        """
        Retrieve the activations of a given layer feeding the model with the
        images in the loader.
        Args:
            layer (str): name of the layer
            dir_activations (str): directory where the activations are stored or will be stored
            device (torch.device): device to use
        Returns:
            torch.Tensor: activations
        """
        if self.data_loader is None:
            raise ValueError("Data loader not set. Please set the data loader first (run wrapper_item.set_loader(args)).")
        layer_file = f"{dir_activations}/{layer}.npy"
        total_activations = []
        if os.path.exists(layer_file):
            # Load the activations from the file if they exist
            print(f"Loading activations for layer {layer} from {layer_file}")
            total_activations = np.load(layer_file)
        else:
            print(f"Computing activations for layer {layer}")
            activations = self.compute_activations(
               layer, device
            )
            # since the function checks one layer at a time
            activations = activations.numpy()
            np.save(f"{layer_file}", activations)
            total_activations = activations

        # Convert the activations to a torch tensor and return
        total_activations = torch.from_numpy(total_activations)
        return total_activations


class TimmModelWrapper(ModelWrapper):
    """
    Abstract class that wraps a model from the timm library and provides methods to compute activations.
    """

    def __init__(self):
        super().__init__()
        self.model = self.load_checkpoint()
        self.input_size = 224


    @torch.no_grad()
    def compute_activations( self,
            layer: str, device: torch.device):
        """Retrieve the activations of a given layer feeding the model with the
        images in the loader.

        Args:
            layer (str): name of the layer
            device (torch.device): device to use

        Returns:
            torch.Tensor: activations
        """

        assert layer == "stage3", f"Layer {layer} is not supported for Timm models. Please use 'stage3' instead."

        # Move the model to the device
        original_device = next(self.model.parameters()).device
        self.model = self.model.to(device)

        activations = []
        for data in self.data_loader:
            images = self.iter_data(data)

            if len(images.shape) == 4:
                images = images.squeeze(0)

            if self.transformation is not None:
                images = self.transformation(images)
            
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            
            images = images.to(device)

            feats = self.model(images)
            last_feat = feats[-1]
            activations.append(last_feat.detach().cpu())

        activations =   torch.cat(activations)

        # Move the model back the original device
        self.model = self.model.to(original_device)

        return activations

class ConvNext(TimmModelWrapper):
 
    def load_checkpoint(
        self,
    ):
        """
        Load the ConvNext model from the timm library.

        Returns:
            torch.nn.Module: model
        """

        print(f"Loading model: ConvNext\n\tfrom Timm library")
        
        model = timm.models.create_model("convnext_base.fb_in22k_ft_in1k",
                    pretrained=True,
                    features_only=True,
                    out_indices=(0, 1, 2, 3))
        model.eval()
        print(f"ConvNext loaded in memory. Modality: Evaluation")
        return model

class MaxViTModel(TimmModelWrapper):

 
    def load_checkpoint(
        self,
    ):
        """
        Load the MaxViT model from the timm library.

        Returns:
            torch.nn.Module: model
        """

        print(f"Loading model: MaxViT\n\tfrom Timm library")
        
        model = timm.models.create_model("maxvit_tiny_tf_224.in1k",
                    pretrained=True,
                    features_only=True,
                    out_indices=(0, 1, 2, 3))
        model.eval()
        print(f"MaxViT loaded in memory. Modality: Evaluation")
        return model
    
class EfficientViTModel(TimmModelWrapper):
 
    def load_checkpoint(
        self,
    ):
        """
        Load the EfficientViT model from the timm library.

        Returns:
            torch.nn.Module: model
        """

        print(f"Loading model: EfficientViT\n\tfrom Timm library")
        
        model = timm.models.create_model("efficientvit_b1.r224_in1k",
                    pretrained=True,
                    features_only=True,
                    out_indices=(0, 1, 2, 3))
        model.eval()
        print(f"EfficientViT loaded in memory. Modality: Evaluation")
        return model

class CvT(ModelWrapper):
    def __init__(self):
        """
        Initialize the CvT model wrapper.
        """
        super().__init__()
        self.model = self.load_checkpoint()
        self.input_size = 224
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/cvt-13")
 
    def load_checkpoint(
        self,
    ):
        """
        Load the CvT model from the HuggingFace library.
        Returns:
            torch.nn.Module: model
        """
        print(f"Loading model: CvT\n\tfrom HuggingFace library")
        model = CvtForImageClassification.from_pretrained("microsoft/cvt-13")
        model.eval()
        print(f"CvT loaded in memory. Modality: Evaluation")
        return model




    @torch.no_grad()
    def compute_activations(self, layer: str, device: torch.device):
        """
        Retrieve the activations of a given layer feeding the model with the images in the loader.
        Args:
            layer (str): name of the layer
            device (torch.device): device to use
        Returns:
            torch.Tensor: activations
        """

        assert layer == "stage3", f"Layer {layer} is not supported for CvT. Please use 'stage3' instead."

        original_device = next(self.model.parameters()).device
        self.model = self.model.to(device)
        temp_activations = []

        def hook_feature(module, inp, output):
            temp_activations.append(output.detach().cpu())

        handles = []
        layer = (
            self.model.cvt.encoder.stages[2]
            .layers[9]
            .attention.attention
            .convolution_projection_query
            .convolution_projection
            .convolution
        )
        handles.append(layer.register_forward_hook(hook_feature))

        activations = []

        for data in self.data_loader:
            images = self.iter_data(data)

            if len(images.shape) == 4:
                images = images.squeeze(0)

            images = self.feature_extractor(images=images, return_tensors="pt")

            images = images.to(device)

            _ = self.model(**images)

            activations.append(temp_activations[0])

            temp_activations.clear()

        # Move the model back the original device
        self.model = self.model.to(original_device)

        # Remove the hooks
        for handle in handles:
            handle.remove()

        activations = torch.cat(activations)

        return activations

class Place365Model(ModelWrapper):
    def __init__(self, model_name, weights):
        super().__init__()
        self.input_size = 227 if "alexnet" in model_name else 224
        self.model = self.load_checkpoint(model_name, weights)
    
    def load_checkpoint(
        self,
        model_name,
        weights,
    ):
        """
        Load the model from the given weights.
        Args:
            model_name (str): name of the model
            weights (str): path to the weights

        Returns:
            torch.nn.Module: model
        """

        print(f"Loading model:{model_name}\n\tfrom {weights}")

        model_fn = torchvision.models.__dict__[model_name]

        checkpoint = torch.load(weights, map_location='cpu')
        model = model_fn(num_classes=365)
        # the data parallel layer will add 'module' before each
        # layer name
        state_dict = {
            str.replace(k, "module.", ""): v
            for k, v in checkpoint["state_dict"].items()
        }

        model.load_state_dict(state_dict)
        model.eval()
        print(f"{model_name} loaded in memory. Modality: Evaluation")
        return model


    
class DenseNetPlace365(Place365Model):

    def __init__(self, model_name, weights):
        super().__init__(model_name, weights)

    def load_checkpoint(
        self,
        model_name,
        weights,
    ):
        """
        Load the DenseNet model from the given weights.
        Args:
            model_name (str): name of the model
            weights (str): path to the weights

        Returns:
            torch.nn.Module: model
        """
        def rep(k):
            for i in range(6):
                k = k.replace(f"norm.{i}", f"norm{i}")
                k = k.replace(f"relu.{i}", f"relu{i}")
                k = k.replace(f"conv.{i}", f"conv{i}")
            return k
        print(f"Loading model:{model_name}\n\tfrom {weights}")

        model_fn = torchvision.models.__dict__[model_name]

        checkpoint = torch.load(weights, map_location='cpu')
        model = model_fn(num_classes=365)
        # Fix old densenet pytorch names.        
        state_dict = checkpoint.state_dict()
        state_dict = {rep(k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        
        model.eval()
        print(f"{model_name} loaded in memory. Modality: Evaluation")
        return model
    
class Cub200Model(ModelWrapper):
    def __init__(self, weights):
        super().__init__()
        self.model = self.load_checkpoint('resnet50', weights)
        self.input_size = 448
    
    def load_checkpoint(
        self,
        model_name,
        weights,
    ):
        """
        Load the model from the given weights.
        Args:
            model_name (str): name of the model
            weights (str): path to the weights
        Returns:
            torch.nn.Module: model
        """

        print(f"Loading model:{model_name}")

        model_fn = CUBmodel
        representation = {'function':SEB,
                          'is_vec':True,
                          'input_dim':2048,
                          'dimension_reduction':256}
        model = model_fn(model_name, representation, 200, 0, pretrained=True)

        checkpoint = torch.load(weights)
        # layer name
        state_dict = {
            str.replace(k, "module.", ""): v
            for k, v in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"{model_name} loaded in memory. Modality: Evaluation")
        return model