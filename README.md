# DFAN++: Enhanced Triple-branch Network for Generalized Zero-Shot Image Classification

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1%2B-orange)](https://pytorch.org)
[![Paper](https://img.shields.io/badge/Paper-Neurocomputing-green)](https://www.sciencedirect.com/journal/neurocomputing)

> Official PyTorch implementation of the paper:  
> **DFAN++: Enhanced Triple-branch Network for Generalized Zero-Shot Image Classification**  
> *Neurocomputing 2025*  

## 📊 Architecture Overview
![](framework/arc.png)  
*DFAN++ Framework: Integrating EAFL, CDM and CPP branches*

## 🚀 Performance Highlights
| Dataset | CZSL (acc) | GZSL (U) | GZSL (S) | GZSL (H) | 
|---------|------------|----------|----------|----------|
| **CUB** | 78.7       | 72.9     | 76.8     | 74.8     |
| **SUN** | 68.1       | 51.4     | 40.5     | 45.3     |
| **AWA2**| 71.7       | 66.2     | 84.5     | 74.2     |

## ⚙️ Installation
```bash
# Install dependencies
pip install -r requirements.txt                                             
```

## 📂 Dataset Preparation
| Dataset | Download Link | Attributes | Classes | Images |
|---------|------------|----------|----------|----------|
| **CUB** | [download link](https://www.vision.caltech.edu/datasets/cub_200_2011/)       | 312     | 200     | 11788     |
| **SUN** | [download link](https://cs.brown.edu/~gmpatter/sunattributes.html)       | 102     | 717     | 14340     |
| **AWA2**| [download link](https://cvml.ista.ac.at/AwA2/)       | 85     | 50     | 37322     |
| **aPY** | [download link](https://vision.cs.uiuc.edu/attributes/)       | 64     | 32     | 15399     |

we use standard splits from [*Xian et al.*](https://arxiv.org/abs/1707.00600).
We provide trained models on three different datasets: (https://pan.baidu.com/s/1cfN3lu-_fTOc8LaOhmv1Uw?pwd=1111) Extraction code：1111

## ✏️ Citation
```bibtex

@article{ZHU2025130880,
title = {DFAN++: Enhanced triple-branch network for generalized zero-shot image classification},
journal = {Neurocomputing},
volume = {653},
pages = {130880},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2025.130880},
url = {https://www.sciencedirect.com/science/article/pii/S0925231225015528},
author = {Yifan Zhu and Lei Xiang and Yuan Zhou and Haoran Duan and Yang Long}                                     
```

## 🙏 Acknowledgements
Parts of our codes based on:
* [FaisalAlamri0/ViT-ZSL](https://github.com/FaisalAlamri0/ViT-ZSL)
* [GAInuist/DFAN](https://github.com/GAInuist/DFAN)          

## 📧 Contact
If you have any questions about codes, please don't hesitate to contact us by yifanzhu12138@gmail.com.
