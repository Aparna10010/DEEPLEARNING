# Fashion MNIST - Autoencoder + Classifier & CNN Classifier

This project explores two approaches to image classification using the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist):

1. **Autoencoder + Logistic Regression**
2. **CNN Classifier using Keras**

---

##  What’s Inside

- **Autoencoder**: Learns compressed features (latent space) from fashion images.
- **Logistic Regression**: Trained on the encoded features to classify images.
- **CNN**: A separate Convolutional Neural Network trained directly on the images for comparison.
- **Model Saving & Loading**: All models are saved using `.h5` and `.pkl` formats.
- **Reconstruction Visualization**: Original vs reconstructed image comparisons.

---

##  Models Used

###  Autoencoder + Logistic Regression
- Input → Encoder → Latent Features → Decoder
- Latent features are used to train a logistic regression classifier
- Combines **unsupervised learning** (autoencoder) + **supervised learning** (classifier)

###  CNN Classifier
- Trained directly on Fashion-MNIST image data
- Used as a performance benchmark

---

##  Project Files

- `FashionMNIST_Autoencoder_Classifier.ipynb`: Full notebook with code, training, evaluation & visualizations.
- `README.md`: This file.
- *(Optional)*: Saved model files (`.h5`, `.pkl`) can be added later if file size permits.

---

## Sample Results

| Model                      | Accuracy   |
|---------------------------|------------|
| Autoencoder + Logistic    | ~80–83%    |
| CNN Classifier            | ~89–91%    |

---

##  Requirements

Install dependencies using:

```bash
pip install -r requirements.txt

