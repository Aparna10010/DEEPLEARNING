# Fashion MNIST - Autoencoder + Classifier & CNN Classifier

This project explores two approaches to image classification using the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist):

1. **CNN Classifier using Keras**
2. **Autoencoder + Logistic Regression**

---

##  What’s Inside

- **CNN**: A separate Convolutional Neural Network trained directly on the images for comparison.
- **Autoencoder**: Learns compressed features (latent space) from fashion images.
- **Logistic Regression**: Trained on the encoded features to classify images.
- **Model Saving & Loading**: All models are saved using `.h5` and `.pkl` formats.
- **Reconstruction Visualization**: Original vs reconstructed image comparisons.

---

##  Models Used

###  CNN Classifier
- Trained directly on Fashion-MNIST image data
- Used as a performance benchmark

###  Autoencoder + Logistic Regression
- Input → Encoder → Latent Features → Decoder
- Latent features are used to train a logistic regression classifier
- Combines **unsupervised learning** (autoencoder) + **supervised learning** (classifier)


---

##  Project Files

- `Fashion_mnist.ipynb`: Full notebook with code, training, evaluation & visualizations.
- `README.md`: This file.
- models/
  - fashion_autoencoder.h5
  - fashion_encoder.h5
  - cnn_fashion_mnist.h5
  - logistic_model.pkl
---

## Sample Results

| Model                      | Accuracy   |
|---------------------------|------------|
| Autoencoder + Logistic    | ~80–83%    |
| CNN Classifier            | ~89–91%    |

---
##  Tools
- Python
- TensorFlow / Keras
- scikit-learn


