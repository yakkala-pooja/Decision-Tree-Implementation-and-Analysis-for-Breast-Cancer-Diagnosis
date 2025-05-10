
# Decision Tree Classifier from Scratch

## Overview

This project implements a top-down, recursive **Decision Tree Classifier** from scratch, supporting both **Information Gain (Entropy)** and **Gini Index** as splitting criteria. The classifier is applied to the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset to predict whether tumors are malignant or benign. It includes preprocessing, training, evaluation, pruning, and comparison with scikit-learn’s implementation.

---

## Project Structure

```
.
├── data/
│   ├── wdbc_train.csv
│   ├── wdbc_dev.csv
│   ├── wdbc_dev_normalized.csv
│   ├── wdbc_dev_raw.csv
│   ├── wdbc_test.csv
│   ├── wdbc_test_normalized.csv
│   ├── wdbc_test_raw.csv
│   └── wdbc_train_raw.csv
├── notebooks/
│   ├── Decision_Tree.ipynb
│   ├── ChiSquarePruning.ipynb
│   └── WDBC_Decision_Tree.ipynb
├── results/
│   └── report.tex
├── requirements.txt
└── README.md
```

---

## Features

- Custom Decision Tree implementation from scratch
- Supports Entropy and Gini Index as splitting criteria
- χ² pruning for generalization
- Discretization of continuous features using Z-score normalization
- Evaluation with Accuracy, Error Rate, Precision, and Recall
- Comparison with scikit-learn’s `DecisionTreeClassifier`

---

## Dataset

The **WDBC** dataset contains 30 real-valued features derived from digitized images of fine needle aspirate (FNA) of breast masses. Each sample is labeled as either:

- **M** – Malignant  
- **B** – Benign

### Discretization (Z-score Normalization + Binning)

Each feature is standardized using:

```
Zij = (xij - μj) / σj
```

Then binned into levels:

- l1: Zij < -2σ  
- l2: -2σ ≤ Zij < -σ  
- l3: -σ ≤ Zij < 0  
- l4: 0 ≤ Zij < σ  
- l5: σ ≤ Zij < 2σ  
- l6: Zij ≥ 2σ  

---

## Installation

```bash
git clone https://github.com/yourusername/decision-tree-classifier.git
cd decision-tree-classifier
python -m venv env
source env/bin/activate  # or use env\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Usage

### Train the model

```bash
python Decision_Tree.py --criterion [entropy|gini] --prune [True|False]
```

### Example

```bash
python Decision_Tree.py --criterion entropy --prune True
```

---

## Evaluation

After training, the model is evaluated on the test set and the following metrics are computed:

- **Accuracy** = Correct predictions / Total predictions
- **Error Rate** = Incorrect predictions / Total predictions
- **Precision** = True Positives / (True Positives + False Positives)
- **Recall** = True Positives / (True Positives + False Negatives)

Results are discussed in:

```
results/report.tex
```

---

## Comparison with scikit-learn

Use the `notebooks/scikit_comparison.ipynb` notebook to:

- Compare accuracy and performance
- Visualize decision trees
- Understand differences in splitting behavior

---

## Notes

- **χ² Pruning**: Prunes statistically insignificant branches to prevent overfitting.
- **Discretization**: ID3-style trees require categorical inputs, so continuous features must be discretized.
- **Medical Relevance**: High precision and recall are critical for cancer diagnostics to minimize misclassifications.

---

## References

- Quinlan, J. R. (1986). *Induction of decision trees*. Machine learning, 1(1), 81-106.
- [Wisconsin Diagnostic Breast Cancer (WDBC) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [scikit-learn Decision Tree Documentation](https://scikit-learn.org/stable/modules/tree.html)

---

## Acknowledgments

This project was developed as part of an academic machine learning course to deepen understanding of decision tree algorithms and practical ML workflow.
