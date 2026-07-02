# SCT-Net

This repository provides the implementation and supplementary materials for the manuscript:

**Road-Structure-Guided Sampling-Controlled Trajectory Aggregation for Road Extraction from High-Resolution Remote Sensing Imagery**

SCT-Net is designed for road extraction from high-resolution remote sensing imagery. It aims to improve road continuity under local road-cue insufficiency by organizing long-range road evidence along road-structure-guided sampling trajectories.

## Repository contents

* `model/`: implementation of SCT-Net and its related modules.
* `dataloader/`: data loading utilities used in the experiments.
* `utils/`: training, evaluation, and auxiliary utilities.
* `train.py`: training entry script.

## Datasets

The raw datasets used in this study are third-party public datasets and should be obtained from their original providers:

* DeepGlobe Road Extraction Dataset: http://deepglobe.org/
* CHN6-CUG Roads Dataset: https://github.com/CUG-URS/CHN6-CUG-Roads-Dataset
* Massachusetts Roads Dataset: https://www.cs.toronto.edu/~vmnih/data/

The raw images and annotations are not redistributed in this repository. Users should follow the licenses and access conditions of the original dataset providers.

## Experimental results

The main quantitative results reported in the manuscript are:

| Dataset       | Precision | Recall |    F1 |   IoU |  mIoU | clDice |
| ------------- | --------: | -----: | ----: | ----: | ----: | -----: |
| DeepGlobe     |     83.47 |  82.73 | 83.10 | 71.09 | 84.80 |  89.39 |
| CHN6-CUG      |     80.35 |  79.13 | 79.74 | 66.30 | 80.86 |  79.21 |
| Massachusetts |     81.57 |  77.70 | 79.58 | 66.09 | 82.10 |  86.93 |

## Usage

Please download the datasets from the original providers and organize them according to the paths used in your local configuration.

Training example:

```bash
python train.py
```

## Data availability

The data availability statement is provided in `DATA_AVAILABILITY.md`.
