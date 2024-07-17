# GLA-Pytorch

The Generalized Logit Adjustment (GLA) algorithm implemented in Pytorch.

## Introduction

This study aims to directly estimate this ratio, for which a novel generalized logit adjustment (GLA) loss incorporating both the ratio of the class-conditional probability densities and the class priors is presented. A new GLT method, named Heuristic-GLA, is then proposed, which employs a $K$-neighborhood-based estimation approach. Several classical long-tail learning methods can be considered as special cases of Heuristic-GLA. Furthermore, given the strong performance of meta-learning, we propose a meta-learning-based estimation approach, resulting in another GLT method named Meta-GLA. 
<p align="center">
    <img src="GLT.jpg" width= "900">
</p>



## Get Started

Please obtain the specific implementations of Heuristic-GLA and Meta-GLA from folders "./Heuristic-GLA" and "./Meta-GLA".


## Results

- Image classification on CIFAR-LT
<p align="center">
    <img src="CIFAR-LT-res.jpg" width= "900">
</p>

- Image classification on ImageNet-GLT and MSCOCO-GLT
<p align="center">
    <img src="GLT-res.jpg" width= "900">
</p>

- Image classification on Subpopulation Shift Datasets
<p align="center">
    <img src="SUB-res.jpg" width= "900">
</p>
