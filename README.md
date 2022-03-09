# caption2image
This project is an implementation of a caption to image network which is trained to map images and captions of those images to the same vector space. This project contains networks for tokenized captions and raw text (character based prediction) captions. 

Important notice:
The code is my own work, using python and Pytorch. However some of the ideas and data are not:

The pretrained networks included in PyTorch (e.g. vgg16 vgg19 and resnet) are not trained or made by me but are freely available in PyTorch.
Please cite the original creators of any pretrained network you use. 

The speech2image neural networks were originally introduced by D. Harwath and J. Glass  (2016) in the paper called: Unsupervised Learning of Spoken Language with Visual Context. The basic neural network structure (the one in speech2im_net.py) and the use of the l2norm hinge loss function is a PyTorch based reproduction of the ideas and work described in that paper.

The NLE2019 branch is the version of the code used in "Learning semantic sentence representations from visually grounded language without lexical knowledge" The supplement_nle file is a supplement to this paper containing the exact numbers that were used to create fig.3 in the paper. Feel free to use this code but please cite the following paper and consider citing other relevant works used in this project.

@article{merkx2019NLE, 
title={Learning semantic sentence representations from visually grounded language without lexical knowledge}, 
volume={25}, 
number={4}, 
journal={Natural Language Engineering}, 
publisher={Cambridge University Press}, 
author={Merkx, Danny and Frank, Stefan L.},
year={2019}, pages={451–466}}
