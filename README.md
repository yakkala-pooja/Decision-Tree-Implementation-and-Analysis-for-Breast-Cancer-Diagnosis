# Decision Tree Classifier for Binary Classification

## Overview

This project implements a **Decision Tree Classifier** to solve a binary classification task. It explores different configurations of the model by tuning key hyperparameters such as `max_depth`, splitting criterion (`gini` vs. `entropy`), and pruning strategies. The model is evaluated using a variety of performance metrics and validated using a separate development dataset to ensure generalization.

## Features

- Decision tree classifier using `scikit-learn`
- Hyperparameter tuning (`max_depth`, `criterion`)
- Model pruning for reducing overfitting
- Confusion matrix and performance metrics (accuracy, precision, recall, F1-score, specificity)
- Evaluation on both training and development datasets
- Performance comparison against majority class baseline

## Dataset

The dataset used for training and evaluation consists of:
- **Numerical and categorical features** (preprocessed using scaling and encoding)
- **Target label**: Binary (0 or 1)

Data was preprocessed to handle:
- Missing values
- Irrelevant columns
- Imbalanced classes (using appropriate sampling or evaluation techniques)

## Model Training & Evaluation

### Hyperparameters Tuned
- `max_depth`: 3 to 25
- `criterion`: `gini` or `entropy`
- `pruning`: Active vs. No pruning

### Performance Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **Specificity**
- **F1 Score**
- **Confusion Matrix**

### Best Configuration
- `max_depth`: 5  
- `criterion`: entropy  
- `pruning`: active  

#### Best Model Results (on dev dataset):
- **Accuracy**: 92.98%
- **Recall**: 90.70%
- **Specificity**: 94.37%
- **Precision**: 90.70%
- **F1 Score**: 90.70%

## Results Summary

| Criterion                 | Accuracy | Precision | Recall | F1 Score |
|--------------------------|----------|-----------|--------|----------|
| **Gini**                 | 92.98%   | 90.70%    | 90.70% | 90.70%   |
| **Entropy (Information Gain)** | 92.98%   | 90.70%    | 90.70% | 90.70%   |
| **Majority Class Baseline**   | 62.28%   | 0.00%     | 0.00%  | 0.00%    |

## Key Takeaways

- Models with `max_depth` > 5 showed diminishing returns.
- Both `gini` and `entropy` performed similarly, though `entropy` was slightly better in some cases.
- Active pruning improved generalization by reducing overfitting.
- The decision tree significantly outperforms the majority class baseline.

## Future Work

- Implement cross-validation for more robust evaluation.
- Explore other classifiers like Random Forest, SVM, or XGBoost.
- Apply SHAP/LIME for model interpretability.
- Perform advanced feature engineering for better signal extraction.

## Installation

### Requirements

- Python 3.x
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn

### Install Dependencies

```bash
pip install -r requirements.txt
