# SCT-Net

This repository provides data availability information and supplementary materials for the manuscript:

**Road-Structure-Guided Sampling-Controlled Trajectory Aggregation for Road Extraction from High-Resolution Remote Sensing Imagery**

SCT-Net is designed for road extraction from high-resolution remote sensing imagery. It aims to improve road continuity under local road-cue insufficiency by organizing long-range road evidence along road-structure-guided sampling trajectories.

## Repository contents

* `DATA_AVAILABILITY.md`: data availability statement and public dataset access information.
* `paper_visualizations/`: visualization results used in the manuscript.
* `results/`: quantitative result summaries reported in the manuscript.
* `model/`: model implementation, to be released upon acceptance.
* `dataloader/`: data loading utilities, to be released upon acceptance.
* `utils/`: training, evaluation, and auxiliary utilities, to be released upon acceptance.
* `train.py`: training entry script, to be released upon acceptance.

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

The complete source code and training scripts will be released upon acceptance of the manuscript.

After release, users should download the datasets from the original providers and organize them according to the required directory structure.

## Data availability

The data availability statement is provided in `DATA_AVAILABILITY.md`.

## Code availability

The complete source code will be released upon acceptance of the manuscript.
