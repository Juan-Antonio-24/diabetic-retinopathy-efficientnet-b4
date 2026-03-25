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
  \`weight = 1 / sqrt(class_count)\`

---

## Results

* **Best QWK:** ~0.66
* **Validation Accuracy:** ~74–76%

---

## Error Analysis

### Key Challenges:

1. Severe class imbalance  
2. Ordinal nature of labels  
3. Noisy data  

---

## Future Improvements

* Class-weighted loss  
* Focal Loss  
* Regression formulation  
* Test Time Augmentation  
* Better preprocessing  

---

## Key Learnings

* Imbalance handling is critical  
* Accuracy is not enough — QWK matters  
* Proper data splitting is essential  
* Iteration improves performance  

---

## How to Run

### 1. Install dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2. Run training

Open Jupyter Notebook:

\`\`\`bash
jupyter notebook
\`\`\`

Then open and run:

\`\`\`
dataset_training_V2.ipynb
\`\`\`

> Note: This project uses PyTorch. For GPU support, install the appropriate version from https://pytorch.org

---

## Author

Juan Antonio Barreda Mendez
