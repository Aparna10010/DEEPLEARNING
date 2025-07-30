# 🔢 MNIST Handwritten Digit Classification using CNN

This project applies a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) from the MNIST dataset. It covers the complete deep learning workflow: preprocessing, data augmentation, training, saving/loading the model, and visualizing predictions.

---

## 📂 Dataset

- *Source*: MNIST (via Keras Datasets)
- *Size*: 70,000 grayscale images (28x28 pixels)
  - 60,000 training images
  - 10,000 test images
- *Classes*: Digits from 0 to 9 (10 classes)

---

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 🧠 Project Workflow

### 1. Data Loading & Preprocessing
- Loaded MNIST dataset using keras.datasets.mnist.load_data()
- Normalized pixel values to range [0,1]
- One-hot encoded class labels for 10 categories

### 2. Data Visualization
- Visualized sample handwritten digit images with their labels

### 3. Data Augmentation
- Applied augmentation using ImageDataGenerator:
  - Rotation
  - Zoom
  - Width/height shift

### 4. CNN Model Building
- Built a simple CNN with:
  - Convolutional + MaxPooling layers
  - Dropout for regularization
  - Dense layers for classification
- Final activation: softmax (for 10-class classification)

### 5. Compilation & Training
- Loss: categorical_crossentropy
- Optimizer: adam
- Metrics: accuracy
- Used *EarlyStopping* to avoid overfitting
- Trained model on augmented data
- Saved best model as mnist_cnn.h5

---

## 📈 Model Evaluation

- Evaluated the model on test set
- *Test Accuracy*: ~98% (update with your result)
- Plotted training & validation accuracy/loss curves

### 📊 Accuracy & Loss Curves

![Accuracy & Loss Curves]()

---

## ✅ Saved Model & Prediction

- *Saved model* using .save()  
- *Loaded model* using load_model()  
- Made predictions on test dataset  
- Visualized predictions alongside the actual digit image

### 🔍 Sample Prediction Visualization

Image: [28x28 image] True Label: 7
Predicted Label: 7 ✅

![Predicted Label]()
---

## ⚠️ Limitations

- Model performs very well on MNIST, but may not generalize to more complex handwriting styles.
- Augmentation helps slightly, but isn’t mandatory due to the simplicity of MNIST.

---

## 🛠️ Possible Enhancements

- Try deeper CNN architecture
- Add Batch Normalization
- Experiment with learning rate schedules
- Use custom handwritten digit images for robustness

---

## 📁 Folder Structure

```
mnist-cnn/ │
├── mnist_cnn.ipynb 
├── mnist_cnn.h5 
├── mnist_curves.png │
├── predictions_grid.png
└── README.md
```
---

## 🏁 Conclusion

This project demonstrates handwritten digit classification using CNNs on the MNIST dataset with over 98% accuracy. It includes saving/loading the model and visualizing predictions, forming a complete deep learning workflow.


---
