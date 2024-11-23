# EHGS-Logo-Classification
EHGS-Logo-Classification: Optimizing VGG16 for Logo Classification Using Enhanced Hunger Games Search (EHGS)

# EHGS-Logo-Classification: Optimizing VGG16 for Logo Classification Using Enhanced Hunger Games Search (EHGS)

This repository contains the code and experiments conducted to enhance logo classification performance using the VGG16 deep learning model. The approach integrates an Enhanced Hunger Games Search (EHGS) algorithm for hyperparameter tuning, aimed at improving the robustness and accuracy of the model. The project is structured in three phases:

## Phase 1: Benchmark Comparison
This phase evaluates the performance of the HGS, EHGS, BA, HHO, and SCA algorithms on 30 real-valued benchmark functions from the IEEE CEC2014 suite. The optimization tasks in this phase are built using the **PyGMO framework**, which provides a unified interface for optimization algorithms and problems, supporting massively parallel environments for computational efficiency.  

For detailed information on setting up the PyGMO environment, please refer to the [PyGMO documentation](https://esa.github.io/pygmo2/install.html).

## Phase 2: Logo Classification
In this phase, various deep learning models are explored for logo classification using the **FlickrLogos-27** dataset, which is publicly available for research purposes. The models include VGG16 and several state-of-the-art architectures such as ResNet50V2 and MobileNetV2. The experiments are conducted in a **Colab environment**, where the necessary libraries are pre-installed, allowing for a seamless and easy setup for running deep learning tasks without the need for local installations.

The **FlickrLogos-27** dataset can be downloaded from the following link:  
[Download FlickrLogos-27](https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/flickrlogos/).



## Phase 3: EHGS for Hyperparameter Tuning
The final phase focuses on applying the EHGS algorithm to optimize hyperparameters for the VGG16 model, enhancing its performance on the logo classification task.

## Key Features:
- Comparison of swarm intelligence algorithms on benchmark functions.
- Exploration of deep learning models for logo classification.
- Hyperparameter optimization using EHGS for improved model performance.

