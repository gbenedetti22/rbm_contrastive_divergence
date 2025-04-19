# Restricted Boltzmann Machine (RBM) for MNIST Digit Recognition

This repository contains a from-scratch implementation of a Restricted Boltzmann Machine (RBM) applied to the MNIST dataset, as required by Assignment 3 of the university course.

## Assignment Overview

The project implements an RBM model for unsupervised learning of representations of handwritten digit images from the MNIST dataset. The implementation includes:

1. An RBM built completely from scratch, both for the training and inference phases
2. A generalized implementation of the Contrastive Divergence (CD-K) learning algorithm
3. Two versions of the model with different K values for the Gibbs sampling Markov chain
4. A classifier that uses the representations learned by the RBM to recognize the digits

## Results

The project compares the performance of two versions of the RBM:
- RBM with CD-1 (K=1): standard version of the Contrastive Divergence algorithm
- RBM with CD-K (K=5): version with extended Gibbs sampling

The results show the differences in terms of:
- Quality of image reconstruction
- Classifier performance using the learned representations
- Training times and computational complexity

## Implementation Details

- The RBM is implemented as an energy-based model with binary units
- The CD-K algorithm is implemented with the ability to specify the number of steps K
- Activation functions use the sigmoid to calculate probabilities
- The final classifier is implemented using scikit-learn

## Notes

This project was developed as part of Assignment 3 for the machine learning/deep learning course.
Text of the assignment:

Implement from scratch an RBM and apply it to DSET2. The RBM should be implemented fully by you (both training and inference steps) but you are free to use library functions for the rest (e.g. image loading and management, etc.). 
Implement a generalization of the Contrastive Divergence (CD) learning algorithm that defines the number of steps K of the Gibbs sampling Markov chain runned before collecting the samples to estimate the model expectation. 
For instance the standard CD learning would be obtained with K=1. 
Test your models by training two versions of them, one with a small K and one with a medium K (I suggest you do not go over 10 steps), and discuss the differences in performance/behaviour (if any).

Outline of the assigment:

1.     Train an RBM with a number of hidden neurons selected by you (single layer) on the MNIST data (use the training set split provided by the website) using CD(K) with two choices of K.

2.     Use the trained RBMs to encode a selection of test images (e.g. using one per digit type) using the corresponding activation of the hidden neurons.

3.    Train a simple classifier (e.g. any simple classifier in scikit) to recognize the MNIST digits using as inputs their encoding obtained at step 2. Use the standard training/test split.

Show a performance metric of your choice in the presentation/handout and use it to confront the two versions of the RBM (obtained with different K).
