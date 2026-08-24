import os

from detectron2.data.datasets import register_coco_instances

root = os.getenv("DETECTRON2_DATASETS", "datasets")
path_cub= os.path.join(root, "CUB_200_2011/")
register_coco_instances("cub200", {}, path_cub +"cub200_coco.json", path_cub+"images/")