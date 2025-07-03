# Fashion MNIST - CNN Classifier &  Autoencoder + Classifier 

This project explores two approaches to image classification using the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist):

1. **CNN Classifier using Keras**
2. **Autoencoder + Logistic Regression + SVM**

---

##  What’s Inside

- **CNN**: A separate Convolutional Neural Network trained directly on the images for comparison.
- **Autoencoder**: Learns compressed features (latent space) from fashion images.
- **Logistic Regression & SVM**: Trained on the encoded features to classify images.
- **Model Saving & Loading**: All models are saved using `.h5` and `.pkl` formats.
- **Reconstruction Visualization**: Original vs reconstructed image comparisons.

---

##  Models Used

###  CNN Classifier
- Trained directly on Fashion-MNIST image data
- Used as a performance benchmark

###  Autoencoder + Logistic Regression + SVM
- Input → Encoder → Latent Features → Decoder
- Latent features are used to train a logistic regression and svm classifier
- Combines **unsupervised learning** (autoencoder) + **supervised learning** (classifier)


---

##  Project Files

- `Fashion_mnist.ipynb`: Full notebook with code, training, evaluation & visualizations.
- `README.md`: This file.
- models/
  - fashion_mnist_autoencoder.h5
  - fashion_mnist_cnn.h5
  - fashion_mnist_enoder.h5
  - fashion_mnist_Log-Regression.pkl
  - fashion_mnist_SVM.pkl
---

## Sample Results

| Model                      | Accuracy   |
|---------------------------|------------|
| Autoencoder + Logistic + SVM   | ~80–83%    |
| CNN Classifier            | ~89–91%    |

---
##  Tools
- Python
- TensorFlow / Keras
- scikit-learn


