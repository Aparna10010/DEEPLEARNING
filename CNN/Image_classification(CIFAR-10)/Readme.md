---

# 🧠 CIFAR-10 Image Classification using CNN

This project applies a *Convolutional Neural Network (CNN)* to classify images from the *CIFAR-10 dataset* into 10 classes. It demonstrates a complete deep learning pipeline including preprocessing, data augmentation, model building, training, evaluation, and visualization.

---

## 📂 Dataset

- *Source*: CIFAR-10 (Keras built-in dataset)
- *Size*: 60,000 images (32x32 pixels RGB)
  - 50,000 for training, 10,000 for testing
- *Classes*: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn

---

## 🧠 Project Workflow

### 1. Data Loading and Preprocessing
- Loaded CIFAR-10 using keras.datasets
- Normalized image pixel values 
- Applied *one-hot encoding* to labels

### 2. Data Visualization
- Displayed a grid of sample training images with their labels
![Trainig Images](https://github.com/Aparna10010/DEEPLEARNING/blob/main/CNN/Image_classification(CIFAR-10)/Original_img_preview.png)

---

### 3. Data Augmentation
- Real-time augmentation applied to training data using ImageDataGenerator:
  - Rotation, zoom, shift, and horizontal flip

### 4. Model Building
- Designed a CNN architecture:
  - Multiple Conv2D + MaxPooling2D layers
  - Flatten + Dense layers
  - Final layer: Dense(10, activation='softmax')
- Compiled with:
  - Loss: categorical_crossentropy
  - Optimizer: adam
  - Metrics: accuracy
- Used *EarlyStopping* to prevent overfitting

### 5. Model Training
- Trained over multiple epochs on augmented data
- Saved best model as cnn_model.h5

---

## 📈 Model Evaluation

- Evaluated on the test set
- *Final Test Accuracy: *~74% (update with your exact value)
- Visualized:
  - Training vs. Validation *Accuracy*
  - Training vs. Validation *Loss*

### 📉 Accuracy & Loss Curves

![Accuracy and Loss Curves](https://github.com/Aparna10010/DEEPLEARNING/blob/main/CNN/Image_classification(CIFAR-10)/loss_accuracy_curve.png)

---

## 🧾 Classification Report
![Classification Report](https://github.com/Aparna10010/DEEPLEARNING/blob/main/CNN/Image_classification(CIFAR-10)/Classification%20Report.png)

---

### 🧮 Confusion Matrix
![Confusion Matrix](https://github.com/Aparna10010/DEEPLEARNING/blob/main/CNN/Image_classification(CIFAR-10)/ConfusionMatrix.png)

---

## 🛠️ Improvements & Next Steps

- Add *Dropout* and *Batch Normalization*
- Try *Transfer Learning* using pretrained CNNs (VGG16, ResNet, etc.)
- Perform *Hyperparameter Tuning*
- Use *ensemble models* for better generalization
- Improve class-wise precision/recall (especially for underperforming classes)

---

## 📁 Folder Structure

cifar10-cnn/ 
├── CIFAR-10_NN.ipynb
├── cifar10_cnn_model.h5 
├── results/
│   ├── loss_accuracy_plot.png
│   ├── confusion_matrix.png 
│   └── classification_report.png
└── README.md


---

## 🏁 Conclusion

This project demonstrates how to build, train, and evaluate a CNN on a multiclass image dataset. While initial results show ~74% accuracy, further improvements like transfer learning can enhance model performance significantly.


---
