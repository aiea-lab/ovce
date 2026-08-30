import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Define paths
CUB_DATASET_ROOT = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "CUB_200_2011"


IMAGE_FOLDER = os.path.join(CUB_DATASET_ROOT, "images")
ANNOTATIONS_FILE = os.path.join(CUB_DATASET_ROOT, "bounding_boxes.txt")
LABELS_FILE = os.path.join(CUB_DATASET_ROOT, "image_class_labels.txt")
IMAGE_LIST_FILE = os.path.join(CUB_DATASET_ROOT, "images.txt")
CATEGORY_FILE = os.path.join(CUB_DATASET_ROOT, "classes.txt")
SPLIT_FILE = os.path.join(CUB_DATASET_ROOT, "train_test_split.txt")
OUTPUT_JSON = os.path.join(CUB_DATASET_ROOT,"cub200_coco.json")

# COCO structure
coco_format = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Load categories
def load_categories(category_file):
    categories = []
    with open(category_file, "r") as f:
        for line in f:
            class_id, class_name = line.strip().split(" ", 1)
            categories.append({
                "id": int(class_id),
                "name": class_name,
                "supercategory": "bird"
            })
    return categories

# Load bounding boxes
def load_bounding_boxes(bbox_file):
    bboxes = {}
    with open(bbox_file, "r") as f:
        for line in f:
            image_id, x, y, width, height = line.strip().split()
            bboxes[int(image_id)] = [float(x), float(y), float(width), float(height)]
    return bboxes

# Load labels
def load_labels(labels_file):
    labels = {}
    with open(labels_file, "r") as f:
        for line in f:
            image_id, class_id = line.strip().split()
            labels[int(image_id)] = int(class_id)
    return labels

# Load image list
def load_image_list(image_list_file):
    image_list = {}
    with open(image_list_file, "r") as f:
        for line in f:
            image_id, file_name = line.strip().split()
            image_list[int(image_id)] = file_name
    return image_list

def load_dataset_split(split_file):
    split = {}
    with open(split_file, "r") as f:
        for line in f:
            image_id, is_train = line.strip().split()
            split[int(image_id)] = int(is_train)
    return split

# Main conversion function
def convert_cub_to_coco():
    categories = load_categories(CATEGORY_FILE)
    bboxes = load_bounding_boxes(ANNOTATIONS_FILE)
    labels = load_labels(LABELS_FILE)
    image_list = load_image_list(IMAGE_LIST_FILE)

    annotation_id = 1

    # Populate categories
    coco_format["categories"] = categories

    split = load_dataset_split(SPLIT_FILE)
    # Process images and annotations
    for image_id, file_name in tqdm(image_list.items(), desc="Processing Images"):
        image_path = os.path.join(IMAGE_FOLDER, file_name)
        if not os.path.exists(image_path) or split[image_id] == 1:
            continue

        # Load image dimensions
        img = cv2.imread(image_path)
        height, width, _ = img.shape

        # Add image to COCO format
        coco_format["images"].append({
            "id": image_id,
            "file_name": file_name,
            "height": height,
            "width": width
        })

        # Add corresponding annotations
        if image_id in bboxes and image_id in labels:
            bbox = bboxes[image_id]
            class_id = labels[image_id]

            # COCO bbox format: [x, y, width, height]
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            annotation_id += 1

    # Save to JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(coco_format, f, indent=4)

if __name__ == "__main__":
    convert_cub_to_coco()
    print(f"COCO format annotations saved to {OUTPUT_JSON}")
