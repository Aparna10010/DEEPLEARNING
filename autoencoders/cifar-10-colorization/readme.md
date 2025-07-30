---

# 🎨 CIFAR-10 Image Colorization using Dense Autoencoder

This project demonstrates how to colorize grayscale images using a *Dense Autoencoder* on the CIFAR-10 dataset. The goal is to train a model that takes grayscale images as input and reconstructs their color versions.

---

## 🧰 Tech Stack

- *Python*
- *TensorFlow / Keras*
- *NumPy*
- *Matplotlib*

---

## 🧠 Project Workflow

1. *Data Loading & Preprocessing*
   - Loaded CIFAR-10 dataset from Keras datasets.
   - Visualized original RGB images.
   - Normalized image pixel values between 0 and 1.
   - Converted color images to grayscale using luminance formula.
   - Visualized grayscale vs. original images.
   - Flattened images for feeding into the dense network.

2. *Model Architecture*
   - Built a Dense Autoencoder with a bottleneck layer (size 32).
   - Encoder: Fully connected layers reducing to bottleneck size.
   - Decoder: Fully connected layers expanding back to original shape.

3. *Training*
   - Trained the autoencoder for 5 epochs with batch size of 256.
   - Used MSE loss and Adam optimizer.
   - Tracked and visualized training and validation loss.

4. *Evaluation & Visualization*
   - Reconstructed test grayscale images back to color.
   - Reshaped flat outputs back to 32x32 image shape.
   - Compared and visualized original, grayscale, and reconstructed images.
   - Plotted loss curves for both training and validation.

---

## 📊 Results

- Successfully reconstructed color images from grayscale input using a fully connected autoencoder.
- Achieved validation loss as low as *~0.0139*.
- Loss curves show steady convergence over epochs.

---

## 📌 Folder Structure

📁 cifar10_colorization/
├── model_training.ipynb               # Jupyter notebook for training the model
├── loss_curve_plot.png                # Plot of training/validation loss
├── results_visualization/            # Folder containing visual output comparisons
│   └── original_vs_grayscale_vs_reconstructed.png
└── README.md                          # Project documentation
---

## 🚀 Future Improvements

- Use *Convolutional Autoencoders* for better spatial feature extraction.
- Add *skip connections* for sharper reconstruction.
- Perform *hyperparameter tuning* (learning rate, bottleneck size).
- Try *transfer learning* on grayscale images from other datasets.

---

## 🏁 Final Notes

This project is an excellent demonstration of how autoencoders can learn to reconstruct missing color information from grayscale data. It also introduces key concepts in unsupervised learning and image processing.


---
