import absl.flags

from datasets import datasets_register  # register all new datasets

# GENERAL flags
absl.flags.DEFINE_string(
    "dataset", "ade20k_150_test_sem_seg", "dataset to use. Values: [ade20k_150_test_sem_seg, cub200, ade20k_full_sem_seg_freq_val_all, mapillary_vistas_sem_seg_val, cityscapes_fine_sem_seg_val, context_459_test_sem_seg, coco_2017_test_stuff_all_sem_seg, voc_2012_test_sem_seg]",
)
absl.flags.DEFINE_string(
    "model",
    "resnet18",
    "model to use. Values:[resnet18, resnet_cub200, alexnet, densenet161, cvt, maxvit, convnext, efficientvit]",
)
absl.flags.DEFINE_string(
    "layer",
    "layer4",
    "Layer to analyze. Values: 'layer4' for resnet18, 'stage3' for cvt, maxvit, convnext, efficientvit, 'features' for densenet161,  alexnet, and resnet_cub200"
)
absl.flags.DEFINE_integer("random_units", 0, "number of units")
absl.flags.DEFINE_string(
    "configuration_name",
    "std",
    "Name of the configuration to use. This needs to be used to store the segmentations in a specific directory. If not specified, the default is 'std'.",
)
absl.flags.DEFINE_string("device", "cuda", "device to use to store the model")
absl.flags.DEFINE_integer("seed", 0, "seed to use to set reproducibility")

# SEGMENTORS FLAGS
absl.flags.DEFINE_string("segmentor", "human", "segmentor to use. Currently supported: [human, mask2former, catseg, masqclip, scan, sed, openseed]")
absl.flags.DEFINE_list("custom_classes", None, "Custom classes to use. It may be a list of string or nested lists of strings." )
absl.flags.DEFINE_string("predefined_concept_set", None, "Predefined concept set to use. Currently supported for CUB200: [granularity_0, granularity_1, granularity_2, all]")
absl.flags.DEFINE_boolean("batch_parsing", True, "Whether to parse the segmentations in batches. If False, the segmentations are parsed one by one. This is useful for large datasets that may not fit in memory.")
absl.flags.DEFINE_integer("parallel_concepts", 50, "Number of concepts to parse in parallel. This is useful for large datasets that may not fit in memory.")
absl.flags.DEFINE_list("ignore_concepts", [], "Concepts to ignore when parsing the segmentations. This is useful for concept sets that have background or null concepts")

# EXPLANATIONS FLUGS
absl.flags.DEFINE_integer("length", 3, "length of explanations")
absl.flags.DEFINE_integer("num_clusters", 5, "Number of clusters to use for clustering activations")
absl.flags.DEFINE_integer("beam_limit", 5, "beam limit")
absl.flags.DEFINE_float("quantile", 0.01, "quantile to use to threshold the activations when num_clusters is set to 1.")

# DIRECTORY FLAGS
absl.flags.DEFINE_string(
    "root_models", "data/model", "root directory for models"
)
absl.flags.DEFINE_string(
    "root_segmentations",
    "data/cache/segmentations",
    "root directory for segmentations",
)
absl.flags.DEFINE_string(
    "root_activations",
    "data/cache/activations",
    "root directory for activations",
)
absl.flags.DEFINE_string(
    "root_results", "data/results", "root directory for results"
)


