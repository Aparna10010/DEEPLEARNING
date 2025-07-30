
---

# 📁 Cotton Disease Classification using CNN (MobileNetV2)

## 🔍 Project Overview

This project applies **transfer learning** with **MobileNetV2** to classify cotton leaf images into four categories:

* **Disease Cotton Leaf**
* **Disease Cotton Plant**
* **Fresh Cotton Leaf**
* **Fresh Cotton Plant**

🎯 *Goal:* Detect plant diseases early to support timely interventions and help farmers improve crop yield.

---

## 🧠 What I Did

* Loaded and preprocessed a labeled dataset of cotton leaf images
* Applied **MobileNetV2** (pretrained on ImageNet) with custom classification layers
* Trained two versions:

  * **Baseline model** (no regularization)
  * **Regularized model** (with Dropout + EarlyStopping)
* Evaluated models using:

  * Accuracy/Loss curves
  * Confusion matrix
  * Classification report

---

## 📊 Final Model Performance

| Metric        | Value          |
| ------------- | -------------- |
| Test Accuracy | **97.1%**      |
| Test Loss     | **0.0818**     |
| Model Size    | \~14MB (`.h5`) |
| Classes       | 4              |

---

## 🔬 Key Findings

* The **baseline model** showed smoother accuracy/loss curves and better generalization.
* The **regularized model** slightly underfit (less confident predictions and noisy learning curves).
* ✅ **Final Decision:** *Baseline model selected for deployment.*

---

## 📦 Files Included

* `cotton_disease_model.h5` – Trained model
* `training_metrics.csv` – Accuracy/loss logs
* `confusion_matrix.png` – Evaluation matrix visualization
* `classification_report.txt` – Precision / Recall / F1
* `notebook.ipynb` – Full training & evaluation code

---

## ✅ Tech Stack

* Python 🐍
* TensorFlow / Keras
* NumPy, Pandas, Matplotlib, Scikit-learn
* Google Colab (training & evaluation)

---

## 🚀 How to Use

1. **Clone** the repo or download the `.h5` model
2. **Load** the model in any TensorFlow/Keras environment:

   ```python
   from tensorflow.keras.models import load_model
   model = load_model('cotton_disease_model.h5')
   ```
3. **Preprocess** your input image (resize, normalize like training data)
4. **Predict** using:

   ```python
   predictions = model.predict(image)
   ```

---

## 🙋‍♀ Author

**Aparna Sharmaa**
Aspiring Data Analyst | M.Com | Passionate about ML & Social Impact

---

