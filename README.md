<p align="center">
<img width="400" src="https://raw.githubusercontent.com/fchavelli/seg-eval/main/assets/tsseg-eval-logo-resize.png"/>
</p>


<h1 align="center">SegEval</h1>
<h2 align="center">Toward Interpretable Evaluation Measures for Time Series Segmentation</h2>
<h3 align="center">NeurIPS'25</h3>

<div align="center">
<p>
<img alt="GitHub issues" src="https://img.shields.io/github/issues/fchavelli/seg-eval">
</p>
</div>



Time series segmentation is a fundamental task in analyzing temporal data across various domains, from human activity recognition to energy monitoring. While numerous state-of-the-art methods have been developed to tackle this problem, the evaluation of their performance remains critically limited. Existing measures predominantly focus on change point accuracy or rely on point-based metrics such as Adjusted Rand Index (ARI), which fail to capture the quality of the detected segments, ignore the nature of errors, and offer limited interpretability. In this paper, we address these shortcomings by introducing two novel evaluation measures: WARI (Weighted Adjusted Rand Index), a temporal extension of ARI that accounts for the position of segmentation errors, and SMS (State Matching Score), a fine-grained metric that identifies and scores four distinct and fundamental types of segmentation errors while allowing error-specific weighting. We empirically validate WARI and SMS on synthetic and real-world benchmarks, showing that they not only provide a more accurate assessment of segmentation quality but also uncover insights, such as error provenance and type, that are inaccessible with traditional measures.

## References

If you use SMS or WARI in your project or research, please cite the following paper:

* [NeurIPS'25](https://arxiv.org/pdf/2510.23261)

> "Toward Interpretable Evaluation Measures for Time Series Segmentation"<br/>
> Félix Chavelli, Paul Boniol, Michaël Thomazo<br/>
> The Thirty-ninth Annual Conference on Neural Information Processing Systems **(NeurIPS'25)**<br/>

```bibtex
@inproceedings{
chavelli2025toward,
title={Toward Interpretable Evaluation Measures for Time Series Segmentation},
author={F{\'e}lix Chavelli and Paul Boniol and Micha{\"e}l Thomazo},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=Gz6dujD5j0}
}
```

## How to use SegEval

### Option 1: Use the metrics (PyPI package)

To use the evaluation measures (SMS, WARI, etc.) in your own project, you can install the (beta) lightweight package via pip:

```bash
$ pip install -i https://test.pypi.org/simple/ tsseg-eval==0.1.2
```

### Option 2: Reproduce the paper results

To reproduce the experiments and results presented in the paper, you need to clone this repository and install the full environment.

We recommend using a Python>3.9 virtual environment with the following dependencies.

```bash
git clone https://github.com/fchavelli/seg-eval.git
cd seg-eval
conda env create -f environment.yml
conda activate seg-eval
```

### Usage

```python
import numpy as np
import pandas as pd
from claspy.segmentation import BinaryClaSPSegmentation
from segeval import f1, covering, nmi, ari, wari, sms

def run_clasp(time_series):
    start_time = time.time()
    clasp = BinaryClaSPSegmentation()
    clasp.fit_predict(time_series)
    change_points = clasp.change_points.tolist()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return change_points, elapsed_time

# Run a segmentation method (Clasp here)
prediction, elapsed_time = run_clasp(data)


# For Change Point Detection
# Compute F-score
f1_score = f1(groundtruth, prediction)

# Compute coverage
cov_score = covering(groundtruth, prediction)

# For State Detection
# Compute Normalized Mutual Information
nmi_score = nmi(groundtruth, prediction)

# Compute Adjusted Rand Index
ari_score = ari(groundtruth, prediction)

# Compute Weighted Adjusted Rand Index
wari_score = wari(groundtruth, prediction)

# Compute State Matching Score
sms_score = sms(groundtruth, prediction)

```



## Reproduce the Paper

### Dataset Preparation
You can download the datasets used in the paper from the following links:

| Dataset   | Type               | Download Link |
|----------|----------|--------------------|
| MoCap   | Real-world | [download](https://drive.google.com/file/d/1Z3HRSxUUfjiPRMzGrOcGie63S1HXA8nf/view?usp=sharing) |
| ActRecTut| Real-world | [download](https://drive.google.com/file/d/1tU5EmxRUk37TzgvpkcgTMQSVG8DBGCUt/view?usp=sharing) |
| PAMAP2| Real-world | [download](https://drive.google.com/file/d/11zwi7PwJiRujncT7kt0NOGOo_GavSSo2/view?usp=sharing) |
| UscHad| Real-world | [download](https://drive.google.com/file/d/1kBHPZZCCN1zrZd7CoSGzG3_W0Jdsm9kF/view?usp=sharing) |
| UcrSeg| Real-world | [download](https://drive.google.com/file/d/1nGH-l3tkp18SauzUUR6P0FhlhEQDLTu2/view?usp=sharing) |

After downloading the datasets, move them to the '\data' directory, ensuring the following directory structure:

```
.
├── data
│   ├── ActRecTut
│   │   ├── subject1_walk
│   │   │   ├── S111.dat
│   │   │   ├── ...
│   │   ├── subject2_walk
│   │   │   ├── S111.dat
│   │   │   ├── ...
│   ├── MoCap
│   │   ├── 4d
│   │   │   ├── amc_86_01.4d
│   │   │   ├── ...
│   │   ├── raw
│   │   │   ├── amc_86_01.txt
│   │   │   ├── ...
│   ├── PAMAP2
│   │   ├── Protocol
│   │   │   ├── subject101.dat
│   │   │   ├── ...
│   ├── USC-HAD
│   │   ├── Subject1
│   │   ├── Subject2
│   │   ├── ...
│   ├── UCRSEG
│   │   ├── Cane_100_2345.txt
│   │   ├── DutchFactory_24_2184.txt
│   │   ├── ...

```
### Reproduce the experimental results

Run main experiment (flag `multivariate` also includes univariate dataset)

```bash
python src/experiments.py --t multivariate
```

Evaluate the algorithms
```bash
python src/evaluation.py multivariate
```

Code for some additional experiments is available in `src` folder.

Results will be saved in a `results/` directory

## Acknowledgements

This work leverages the [E2USd](https://github.com/AI4CTS/E2Usd/tree/main/Baselines) implementation as its foundation.

Our gratitude extends to the authors of the following studies for making their datasets publicly available:
- [UCRSEG](https://doi.org/10.1109/ICDM.2017.21)
- [MoCap](https://dl.acm.org/doi/abs/10.1145/2588555.2588556)
- [ActRecTut](https://dl.acm.org/doi/abs/10.1145/2499621)
- [PAMAP2](https://doi.org/10.1109/ISWC.2012.13)
- [USCHAD](https://doi.org/10.1145/2370216.2370438)
