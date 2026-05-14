# Detection of Patronizing and Condescending Language (PCL) using Multi-Task Transformer Architectures

This repository contains an NLP framework for detecting Patronizing and Condescending Language (PCL) using Transformer-based architectures. The project implements a multi-task learning approach to address class imbalance and linguistic nuance. 
The objective of this research is to develop a robust classifier capable of identifying PCL. 

## Methodology

### 1. Baseline Model
The baseline consists of a **RoBERTa-Base** model fine-tuned for binary classification. This implementation uses a single linear classification head and standard cross-entropy loss, providing a benchmark for performance without specific architectural novelties.

### 2. Proposed Custom Model
The final system utilizes a **RoBERTa-Large** backbone with the following architectural and procedural enhancements:

***Multi-Head Architecture:** The model simultaneously optimizes for binary classification (PCL vs. No PCL) and a seven-class categorization task. 
* **Weighted Random Sampling:** To address extreme class imbalance, a weighted sampling strategy was implemented during the training phase. This ensures the model is exposed to a representative distribution of minority-class (PCL) instances in every batch.
* **Threshold Optimization:** The classification threshold was analytically tuned on the development set to maximize the F1-score, moving beyond the default 0.5 cutoff to better balance precision and recall.


## Experimental Results

The models were evaluated on the official development set. The proposed architecture demonstrated a significant performance gain over the baseline.

| Configuration | Architecture | F1 Score (Dev) |
| :--- | :--- | :--- |
| Baseline | RoBERTa-Base (Single Head) | 0.541 |
| **Proposed** | **RoBERTa-Large (Multi-Head + Sampling)** | **0.617** |

---

## Repository Organization

### Core Implementation
* **`best_model.ipynb`**: The primary execution file used to train and evaluate the custom multi-head model. It includes the logic for weighted sampling and final threshold tuning.
* **`best_roberta_large_model_v2.pt`**: Saved state dictionary containing the optimized weights for the high-performance model.
* **`base_model.ipynb`**: Implementation of the standard RoBERTa-Base baseline without architectural modifications.

### Analysis and Documentation
* **`eda.ipynb`**: Contains extensive exploratory data analysis, including label distributions, sequence length profiling, and linguistic analysis of the corpus.
* **`error_analysis.ipynb`**: Post-run evaluations involving confusion matrices, precision-recall curves, and qualitative review of misclassifications to identify model limitations.

### Data and Outputs
* **`/data`**: Directory containing official training, development, and test splits (including both 7-category labels and binary labels).
* **`/Output`**: Contains final model predictions in the required format.
    * **`dev.txt` / `test.txt`**: One binary prediction (0 or 1) per line.
    * **Alignment**: `test.txt` contains exactly 3,832 lines, matching the official test input size.



**Environment**: Requires Python 3.8+ with `torch` and Hugging Face `transformers` installed. A GPU with 16GB VRAM or more is recommended for the RoBERTa-Large architecture.
