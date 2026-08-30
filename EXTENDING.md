# Extending the Framework with New Segmentors, Datasets, and Models
This document provides instructions on how to extend the framework with new segmentors, datasets, and models.

## Segmentors
To add a new segmentor, you need to implement a new class in `src/segmentor_wrapper.py` that inherits from the `Detectron2Segmentor` class. You can refer to the `CATSeg` class as an example. The new class should implement the following methods:
- `init_from_config`: This method should initialize the segmentor from a configuration file. The configuration file should be located in the `configs` directory and should include both the configuration for Detectron2 and our custom paramaters. Specifically, the configuration file should include the following custom parameters:
  - `INPUT.MIN_SIZE`: The minimum size of the input image.
  - `INPUT.MAX_SIZE`: The maximum size of the input image.
  - `MODEL.WEIGHTS`: The path to the pre-trained weights of the segmentor.
- `set_concept_labels`: Given a set of class names, this method should set the concept labels for the segmentor. The segmentor should be able to compute the concept masks for the given concept labels. 

Additionally, you need to register the new segmentor in the `META_ARCH` registry of Detectron2. You can refer to the `CATSeg` class as an example. The current supported models are registered in the file `src/segmentor_wrapper.py` by calling the `__init__` method of the each segmentor.

## Datasets
The framework expects datasets to be registered with Detectron2 before they can be used by the explanation pipeline. The registration entry point is in [datasets/datasets_register/__init__.py](datasets/datasets_register/__init__.py), which imports the dataset-specific registration modules.

The dataset names used by the code are the same strings passed to the `--dataset` flag in the CLI. 

The dataset must follow the Detectron2 semantic-segmentation format expected by `load_sem_seg` and the `DatasetCatalog`/`MetadataCatalog` registration APIs. A typical workflow when supporting a new dataset is as follows:

1. Download or unzip the dataset into a directory under `DETECTRON2_DATASETS`.
2. Prepare the dataset by running the appropriate script in `scripts/prepare_datasets`. This will create the necessary directory structure and files for Detectron2 to recognize the dataset.
3. Register the dataset using a function similar to those in [datasets/datasets_register](datasets/datasets_register).
4. Use the registered dataset name in the CLI, for example `--dataset=ade20k_150_test_sem_seg` or `--dataset=cub200`.

Please refer to the official <a href=https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html>Detectron2 documentation</a> for more details on dataset registration and the expected format.

**Non-Detectron2 Datasets**: Currently, we do not provide a guide for integrating datasets that do not follow the Detectron2 format. This format is currently necessary for the segmentors we support, which rely on Detectron2 for their implementation. However, it is possible to integrate datasets that do not follow the Detectron2 format by implementing a custom segmentor that does not rely on Detectron2 and my modifying the utils in `utils/dataset_utils.py` to support the new dataset format. This is a more advanced use case and may require additional modifications to the framework in terms of data loading and processing.


## Probed Models

The framework supports probing new vision backbones by adding a wrapper in `src/model_wrapper.py` and registering it in the activation loader in `utils/activations_utils.py`.

At runtime, the probing pipeline does:

1. Build `Settings` from CLI flags (model, layer, dataset, device).
2. Resolve model weights path via `Settings.get_weights()` when needed.
3. Instantiate a model wrapper in `utils/activations_utils.py:get_layer_activations`.
4. Build a dataloader from the selected dataset and run a forward pass to cache activations.
5. Save activations to `data/cache/activations/<dataset>/<model>/<layer>.npy`.

### Requirements a New Probed Model Must Satisfy

- The wrapper must produce per-image spatial feature maps with shape `[N, C, H, W]` for the selected layer.
- The selected layer name must be compatible with the CLI `--layer` flag and with your wrapper logic.
- The base preprocessing must be defined so dataset images can be forwarded through the model.
- The model must run in evaluation mode (`model.eval()`).

Note: downstream code assumes 4D activations. In particular, unit-level processing uses indexing like `layer_activations[:, unit, :, :]`.

### Step 1: Add a Wrapper in `src/model_wrapper.py`

Current existing general wrappers include:
- `Place365Model` for models trained on Place365 and loaded from custom checkpoints.
- `TimmModelWrapper` for models created with `timm` and `features_only=True`.

Moreover, `CvT` uses hooks for HuggingFace models where hooks are used to capture an internal layer.

Minimal wrapper checklist:

- In `__init__`:
  - set `self.model = self.load_checkpoint(...)`
  - set `self.input_size` (used by default resize transform in `ModelWrapper.set_loader`)
- In `load_checkpoint`:
  - create/load model
  - load weights if needed (update `Settings.get_weights()` in `src/settings.py` in this case)
  - call `model.eval()`
  - return model
- Optionally override `compute_activations` when:
  - the model output is not directly the desired feature map
  - a hook is needed to capture an internal tensor
  - preprocessing differs from the default torchvision pipeline

If you rely on the base `ModelWrapper.compute_activations`, the value passed in `--layer` must match a module name available in `self.model._modules` (or nested names if passed as a list to `hook`).

### Step 2: Register the Model in `utils/activations_utils.py`

In `get_layer_activations(cfg)`, import your new wrapper and add a branch to instantiate your wrapper for the new model string (`cfg.get_model_name()`)

This is the entry point used by `run_fast.py`, `run_legacy.py`, and `evaluate.py`.

### Step 3: Expose the Model Name in CLI Flags

Update `src/common_flags.py` so `--model` help text includes your new model name.


### Optional: Model-Specific Preprocessing

By default, `ModelWrapper.set_loader` applies ImageNet-style normalization and a square resize to `self.input_size`.

If your backbone needs different preprocessing, override either:

- `set_loader(...)` to define custom transforms and dataloader behavior, or
- `compute_activations(...)` to apply custom feature-extractor logic per batch.


