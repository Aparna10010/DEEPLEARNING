# 🫁 Chest X-ray Pneumonia Classification using CNN

This project uses a Convolutional Neural Network (CNN) to classify chest X-ray images as either *Pneumonia* or *Normal*. It demonstrates end-to-end deep learning on a medical imaging dataset, covering everything from data loading to evaluation and visualization.

---

## 📂 Dataset

- *Source*: [Kaggle – Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Contains three folders: train, val, and test
- Classes: PNEUMONIA and NORMAL
- Downloaded using kaggle.json API and unzipped locally

---

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Kaggle API

---

## 🧠 Project Workflow

### 1. *Library Imports and Dataset Loading*
- Used kaggle.json to download dataset directly
- Unzipped using zipfile
- Loaded training, validation, and test data using image_dataset_from_directory

### 2. *Image Preprocessing*
- Visualized sample X-ray images
- Rescaled pixel values 
- Applied *data augmentation* using Keras preprocessing layers:
  - Horizontal flip
  - Rotation
  - Zoom
- Batched and shuffled the dataset for training

### 3. *CNN Model Building*
- Sequential CNN architecture with:
  - Conv2D → MaxPooling2D layers
  - Dropout layers for regularization
  - Flatten → Dense → Output layer with sigmoid activation

### 4. *Model Training*
- Compiled using binary_crossentropy, Adam optimizer, and accuracy metric
- Trained for multiple epochs with training and validation datasets
- Tracked performance using loss and accuracy plots

### 5. *Model Evaluation*
- Evaluated model on the test set
- Final *Test Accuracy: **74%*
- Plotted training vs. validation *loss and accuracy curves*

---

## 📊 Results

- The model was able to learn distinguishing features from chest X-rays.
- *Training accuracy increased* consistently, while validation accuracy *plateaued*.
- *Validation loss* started increasing after a few epochs — a sign of *overfitting*.
- Used *early stopping* to avoid unnecessary training.

---

## 📉 Performance Curves

> Training vs Validation Loss and Accuracy

![Loss and Accuracy](results/loss_accuracy_plot.png)

---

## ⚠️ Current Limitations

- Validation accuracy plateaued around 74%
- Overfitting observed after a few epochs
- Model may not generalize well to unseen or external datasets

---

## 🚀 Future Improvements

- Add *Dropout* and *Batch Normalization* layers to improve generalization
- Try *Transfer Learning* using pretrained CNNs like:
  - VGG16, ResNet50, or EfficientNet
- Apply *Learning Rate Scheduling* and *EarlyStopping*
- Evaluate with:
  - *Confusion Matrix*
  - *Classification Report*
  - *ROC-AUC Curve*

---

## 📁 Folder Structure

chest-xray-cnn/
│
- ├── kaggle.json 
- ├── chest_xray.zip 
- ├── chest_xray/                 
## Unzipped dataset (train/val/test) 
- ├── model_training.ipynb 
- ├── results/ │  
- ├── loss_accuracy_plot.png
- └── README.md

---

## 🏁 Conclusion

This project demonstrates the use of CNNs in healthcare imaging. While the model achieves 74% accuracy, it lays the foundation for more robust approaches using transfer learning and regularization.


---
