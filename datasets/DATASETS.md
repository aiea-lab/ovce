
# Prepare the Datasets
Different datasets require different procedures for download and preparation.

## ADE20K-150, ADE20K-847, COCO Stuff, PASCAL VOC, and CUB-200-2011
For the ADE20K-150, ADE20K-847, COCO Stuff, PASCAL VOC 2012, and CUB200 datasets, run the following command from the repository root:

```bash
bash scripts/download_datasets.sh <all|ade20k-150|ade20k-full|coco-stuff|voc2012|cub200>
```

The `--all` option downloads all of the datasets listed above, while the other options download only the specified dataset. The script will also prepare the datasets for use with Detectron2. By default, the datasets will be downloaded to the `data/datasets` directory. After downloading, you should set the `DETECTRON2_DATASETS` environment variable to point to the directory where the datasets are stored. For example, if you downloaded the datasets to `data/datasets`, you can set the environment variable as follows:

```bash
export DETECTRON2_DATASETS=absolute_path_to_/data/datasets
```

## Mapillary Vistas dataset
To use the Mapillary Vistas dataset, register at https://www.mapillary.com/dataset/vistas and download the file `mapillary-vistas-dataset_public_v1.2.zip`. After downloading, unzip the archive, rename the folder to `mapillary_vistas`, and move it to the directory where you store your datasets (that is, the one referenced by the `DETECTRON2_DATASETS` environment variable).

Your dataset folder should look like this:

```bash
mapillary_vistas/
  training/
    images/
    instances/
    labels/
    panoptic/
  validation/
    images/
    instances/
    labels/
    panoptic/
```

## Cityscapes dataset
To use the Cityscapes dataset, register at https://www.cityscapes-dataset.com/ and download `gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip`. After downloading, create a folder named `cityscapes`, unzip the files, and move the directory `gtFine` (stored in `gtFine_trainvaltest`) and the directory `leftImg8bit` (stored in the directory `leftImg8bit_trainvaltest`) into the `cityscapes` folder.
Your dataset folder should look like this:

```
cityscapes/
  gtFine/
    train/
    val/
    test/
  leftImg8bit/
    train/
    val/
    test/
```

Export the path where the Cityscapes dataset is stored:

```bash
export CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes
```

Then clone the following repository:

```bash
git clone https://github.com/mcordts/cityscapesScripts.git
```

Move into the `cityscapesScripts` directory and install the package:

```bash
cd cityscapesScripts
pip install -e .
```

Finally, run the Cityscapes script with:

```bash
python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```

Your final dataset folder should look like this:

```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
  leftImg8bit/
    train/
    val/
    test/
```

At this point, you can remove the `cityscapesScripts` directory if you want, along with the ZIP files you downloaded, as they are no longer needed.

## PASCAL Context dataset
For Pascal-Context, first download Pascal VOC 2010 and then download the auxiliary file needed for  Pascal-Context-459.

To download VOC2010, you can use the `download_datasets.sh` script with the special option `pascal-context`:

```bash
bash scripts/download_datasets.sh voc2010
```

Then download the file `pascalcontext_val.txt` from https://drive.google.com/file/d/1BCbiOKtLvozjVnlTJX51koIveUZHCcUh/view?usp=sharing and place it inside the folder `$DETECTRON2_DATASETS/VOCdevkit/VOC2010`.

To prepare Pascal-Context-459, run the following script from the repository root:

```bash
python datasets/prepare_datasets/prepare_pascal_context_459.py
```

Your final dataset folder should look like this:
```
VOCdevkit/
  VOC2010/
    Annotations/
    ImageSets/
    JPEGImages/
    SegmentationClass/
    SegmentationObject/
    trainval/
    labels.txt
    pascalcontext_val.txt
    annotations_detectron2/
      pc459_val
```