# Brain MRI Tumor Classification using DenseNet and Transfer Learning

## Project Overview

This project aims to develop a **deep learning model to classify brain tumors** from MRI images into **four tumor types**. Accurate classification of tumors is crucial for diagnosis and treatment planning. The project demonstrates the use of **CNNs, transfer learning, and fine-tuning** on real-world medical imaging data.

---

## Dataset

* MRI scans of brains categorized into **four folders**, each representing a tumor type.
* Each class contains ~3,000 images (~12,000 total).
* Images were resized to **224x224** and normalized based on pretrained model requirements.

---

## Key Concepts

1. **Convolutional Neural Networks (CNNs)**

   * Automatically extract spatial features from MRI images.
   * DenseNet121 chosen for **dense connectivity**, feature reuse, and alleviating vanishing gradients.

2. **Transfer Learning**

   * Pretrained DenseNet121 on ImageNet.
   * Initially, **only classifier layers were trained**, freezing the convolutional backbone.

3. **Fine-Tuning**

   * Unfroze last dense block + classifier.
   * Adapted high-level features to MRI-specific patterns.

4. **Loss & Optimization**

   * **CrossEntropyLoss** for multi-class classification.
   * **Adam optimizer** with small learning rate during fine-tuning.

5. **Evaluation Metrics**

   * Training loss per batch, epoch-wise average loss.
   * Validation accuracy.
   * Precision, recall, F1-score per class.
   * Confusion matrix.

6. **Data Handling (PyTorch)**

   * Used `torchvision.datasets.ImageFolder` and `DataLoader`.
   * 80-20 train-test split per class to avoid imbalance.
   * Normalization applied for pretrained DenseNet.

---

## Technologies Used

* **Language:** Python
* **Framework:** PyTorch
* **Model:** DenseNet121 (pretrained)
* **Libraries:** torchvision, sklearn, numpy, matplotlib
* **Environment:** Google Colab with GPU
* **Storage:** Google Drive for model checkpoints

---

## Workflow

1. **Data Preparation**

   * Normalized images, split per class to avoid imbalance.
   * Loaded using PyTorch DataLoaders with batching.

2. **Model Setup**

   * Loaded DenseNet121, replaced classifier with Dropout + Linear layer for 4 classes.
   * Trained classifier only for 10 epochs.

3. **Fine-Tuning**

   * Unfroze last dense block, continued training for 5 epochs.
   * Optimized with smaller learning rate.
   * Continued training for additional epochs for performance improvement.

4. **Evaluation**

   * Calculated validation loss, accuracy.
   * Computed per-class precision, recall, F1-score.
   * Analyzed confusion matrix.

---

## Results

* **Per-class Precision:** `[0.983, 0.906, 0.961, 0.965]`
* **Per-class Recall:** `[0.947, 0.930, 0.983, 0.980]`
* **Per-class F1-score:** `[0.965, 0.918, 0.972, 0.973]`
* **Macro Averages:** Precision = 0.954, Recall = 0.960, F1-score = 0.957
* **Confusion Matrix:**

```
[[711  30   6   4]
 [ 10 436   8  15]
 [  1   5 346   0]
 [  1  10   0 531]]
```

✅ High performance across all classes indicates robust classification.

---

## Learnings

* Transfer learning reduces training time and improves accuracy.
* Fine-tuning top layers balances adaptation with generalization.
* Class-wise metrics are crucial for medical datasets.
* Proper train-test splits prevent class imbalance.
* PyTorch DataLoaders and GPU acceleration streamline deep learning workflows.

---

## Conclusion

This project demonstrates an **end-to-end pipeline** for multi-class MRI tumor classification using **DenseNet, transfer learning, and fine-tuning**. The model achieves **high accuracy and balanced performance**, suitable for medical imaging applications.
