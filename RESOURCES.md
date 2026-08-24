
## Download and prepare probed models

To download the models pre-trained on Places365 (ResNet-18, AlexNet, DenseNet-161), run the following command:

```bash
bash scripts/download_models.sh
```


To download the model pre-trained on CUB-200-2011 (ResNet-50), download the file named `bird_res50.tar' from the following link

https://drive.google.com/drive/folders/1yDvm8ZFcJTBtv0ezOg17EQ4o6OPQ2HM8

and place it in the `data/model/other` directory (create it if it doesn't exist). This model is kindly offered by the <a href="https://github.com/KingJamesSong/DifferentiableSVD">repository</a> associated with the paper "Why Approximate Matrix Square Root Outperforms Accurate SVD in Global Covariance Pooling?" by Song et al. (ICCV 2021).

Finally, from the main directory of the repository, clone the following repository (necessary for the CUB-200-2011 model):
```
git clone https://github.com/KingJamesSong/DifferentiableSVD
```

## Download and prepare datasets
Follow the instructions contained in <a href="datasets/DATASETS.md">`datasets/DATSETS.md`</a>.

## Download and prepare segmentors
To download most of the segmentors (CATSeg), run the following command:
```bash
bash scripts/download_segmentors.sh
```

To download the **MasQCLIP segmentor**, download the model `cross_dataset.pth` from <a href="https://drive.google.com/drive/folders/1wGpl9k7lEYigvSiI2IMx_V_TBGrzfVNl">this link</a> and place it in the `data/model/segmentors` directory (create it if it doesn't exist). Name the file `masqclip_cross_dataset.pth`. 

To download the **SED segmentor**, download the model `sed_model_large.pth` from <a href="https://drive.google.com/file/d/1zAXE0QXy47n0cVn7j_2cSR85eqxdDGg8/view">this link</a> and place it in the `data/model/segmentors` directory (create it if it doesn't exist). Name the file `sed_model_large.pth`.

To download the **SCAN segmentor**, download the model  `SCAN.pth` from <a href="https://drive.google.com/drive/folders/1obgHGQngtQms0u5YUJRnwd4y1IzME-c8">this link</a> and place it in the `data/model/segmentors` directory (create it if it doesn't exist). Name the file `SCAN.pth`.

Some of the segmentors (Mask2Former, MasQCLIP, SCAN, OpenSeeD) require Mask2former to be compiled. To do so, run the following command after having downloaded the segmentors using the download_segmentor script:
```bash
bash scripts/prepare_mask2former.sh
```

**SED** and **SCAN** require also to install openclip

