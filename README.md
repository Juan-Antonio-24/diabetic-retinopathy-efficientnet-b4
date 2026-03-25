# diabetic-retinopathy-efficientnet-b4
# Diabetic Retinopathy Detection using Deep Learning (EfficientNet-B4)

## Overview

This project focuses on building an automated system to detect **Diabetic Retinopathy (DR)** from retinal fundus images using deep learning.

The goal is to classify images into 5 severity levels:

* **0** → No DR
* **1** → Mild
* **2** → Moderate
* **3** → Severe
* **4** → Proliferative DR

The model is trained using a high-resolution medical imaging dataset with real-world challenges such as class imbalance, noise, and variability in acquisition conditions.

---

## Model Architecture

* Backbone: **EfficientNet-B4**
* Framework: **PyTorch**
* Loss Function: **CrossEntropyLoss**
* Metric: **Quadratic Weighted Kappa (QWK)**

---

## Training Strategy

### Key Features:

* Transfer learning with EfficientNet-B4
* Fine-tuning (unfreezing backbone)
* Early stopping
* Checkpointing (best + latest model)
* GPU acceleration

---

## Data Preparation & Split

The original dataset provided by Kaggle contained a single training directory with all images.

A custom preprocessing pipeline was applied:

* Images resized to **224×224**
* Dataset split into:

  * **80% Training set**
  * **20% Validation set**
* Images organized into class folders (0–4)

The split preserved the original class distribution, ensuring a realistic evaluation under imbalanced conditions.

---

## Class Distribution

### Training Set

| Class | Images |
| ----- | ------ |
| 0     | 20648  |
| 1     | 1954   |
| 2     | 4234   |
| 3     | 698    |
| 4     | 566    |

### Validation Set

| Class | Images |
| ----- | ------ |
| 0     | 5162   |
| 1     | 489    |
| 2     | 1058   |
| 3     | 175    |
| 4     | 142    |

> The dataset is highly imbalanced, with ~70% of samples belonging to class 0.

---

## Handling Class Imbalance

To mitigate class imbalance, a **WeightedRandomSampler** was implemented:

* Class weights computed as:
  `weight = 1 / sqrt(class_count)`
* This approach increases exposure to minority classes while avoiding overcompensation and instability.

---

## Results

* **Best QWK:** ~0.66
* **Validation Accuracy:** ~74–76%

### Observations:

* Strong performance on majority class (0)
* Moderate performance on class 2
* Difficulty distinguishing:

  * Mild (1) vs No DR (0)
  * Severe (3–4) vs Moderate (2)

---

## Error Analysis

### Key Challenges:

1. **Severe class imbalance**

   * Model biased toward class 0 (~70% of data)

2. **Ordinal nature of labels**

   * Misclassifications between adjacent classes are common

3. **Noisy data**

   * Variability in image quality (blur, exposure, artifacts)
   * Subjective clinical labeling

---

## Confusion Matrix Insights

* Class 1 frequently misclassified as 0
* Classes 3 and 4 often confused with class 2
* Model tends to predict intermediate classes when uncertain

---

## Future Improvements

* Apply **class-weighted loss**
* Explore **Focal Loss**
* Reformulate as **regression problem (0–4 continuous)**
* Use **Test Time Augmentation (TTA)**
* Improve data preprocessing and augmentation

---

## Key Learnings

* Handling imbalanced datasets is critical in medical AI
* Accuracy alone is not sufficient — QWK provides better insight
* Proper data splitting is essential for reliable evaluation
* Iterative experimentation is key to improving model performance

---

## Project Structure
