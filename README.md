# Introduction-to-Machine-Learning (IML)

This repository contains python scripts written during "Introduction to Machine Learning" ETH Zurich course. In total, there are four projects covering different machine learning topics:

- **K-Fold Cross-Validation and Ridge Regression:** The goal is to find the most optimal regularisation term for the linear regression implementing the k-fold cross-validation.

- **Image Classification:** ResNet50 neural network is implemented to extract features from images to finally create a binary classification based on them.

- **Feature Transformation and Ridge Regression:** Ridge Regression is applied on transformed features and combined with hyperparameter search to optimize model performance.

- **Transfer learning using Autoencoder:** a variation autoencoder (VAE) combined with a ridge regression from the latent space. This architecture is applied on a pre-train set that resembles the train set. The neural network is then fine-tuned with the actual training set. The final output are the values that come out of the regression part when the system is applied to the test set.
