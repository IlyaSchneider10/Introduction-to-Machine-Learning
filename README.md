# Introduction-to-Machine-Learning (IML)

This repository contains python scripts written during "Introduction to Machine Learning" ETH Zurich course. In total, there are four projects covering different machine learning topics. The projects were completed as a collaborative work of three students: Lucas Merlicek, Lukas Radtke and Ilya Schneider.

- **K-Fold Cross-Validation and Ridge Regression:** The goal is to find the most optimal regularisation term for the linear regression implementing the k-fold cross-validation. In case a the regression is applied on the original data, in case b the original data is mapped to a different feature space.

- **Image Classification:** ResNet50 neural network is implemented to extract features from images to finally create a binary classification based on them.

- **Data Imputation and Gaussian Regression:** Missing data in the train set is first imputed; the data set is transformed and cleaned up to apply Gaussian regression.

- **Transfer learning using Autoencoder:** a variation autoencoder (VAE) combined with a ridge regression from the latent space. This architecture is applied on a pre-train set that resembles the train set. The neural network is then fine-tuned with the actual training set. The final output are the values that come out of the regression part when the system is applied to the test set.
