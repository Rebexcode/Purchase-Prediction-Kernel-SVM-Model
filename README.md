# Purchase Prediction using Kernel SVM

## Overview
This project builds a purchase prediction model using a Kernel Support Vector Machine (SVM) classifier.  
The objective is to predict whether a customer will purchase a product based on selected input features, using a non-linear decision boundary enabled by the kernel trick.

The repository demonstrates a standard supervised machine learning workflow:
- data loading and preprocessing
- train/test split
- feature scaling
- model training with Kernel SVM
- prediction and evaluation
- decision boundary visualization

## Problem Statement
Given customer-related input variables, predict a binary target:
- 1: customer is likely to purchase
- 0: customer is unlikely to purchase

This is a binary classification task where Kernel SVM is used to capture potentially non-linear relationships between features and target outcomes.

## Model
The model used is a Support Vector Machine classifier with a kernel (commonly RBF in this type of workflow).

Why Kernel SVM:
- effective in high-dimensional spaces
- suitable for non-linear class separation
- robust margin-based classifier for binary outcomes

## Project Structure
A typical structure for this project is:

- `Purchase_Prediction_Kernel_SVM_Model.ipynb` or `kernel_svm.py`: training and evaluation workflow
- `data.csv` (or similarly named dataset): source data
- `README.md`: project documentation

If your filenames differ, update this section to match the repository exactly.

## Workflow
1. Import libraries
2. Load dataset
3. Select features and target
4. Split data into training and test sets
5. Apply feature scaling
6. Train Kernel SVM classifier
7. Generate predictions
8. Evaluate with confusion matrix and accuracy
9. Visualize training/test decision boundaries

## Installation
Clone the repository:

```bash
git clone https://github.com/Rebexcode/Purchase-Prediction-Kernel-SVM-Model.git
cd Purchase-Prediction-Kernel-SVM-Model
```

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not yet included, install the core packages manually:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage
If using a Python script:

```bash
python kernel_svm.py
```

If using a notebook:

```bash
jupyter notebook
```

Then open the notebook file and run cells in order.

## Expected Output
The project should produce:
- predicted labels on the test set
- confusion matrix
- classification accuracy
- visualization of decision boundaries (for training and test sets)

## Evaluation
Common evaluation metrics for this model include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

If only accuracy and confusion matrix are currently implemented, you can extend evaluation with `classification_report` from scikit-learn.

## Example: Additional Metrics
```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

## Reproducibility Notes
To ensure reproducible results:
- set a fixed `random_state` in train/test split
- keep preprocessing steps identical between training and inference
- scale features using the scaler fitted on training data only

## Potential Improvements
- hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- cross-validation for more robust performance estimates
- class imbalance handling (if applicable)
- model comparison with Logistic Regression, Decision Tree, Random Forest, and XGBoost
- deployment as a simple web app or API endpoint

## License
Add a license file if you want others to use or modify this project.  
A common choice is the MIT License.

## Author
Rebecca Akinlade  
GitHub: https://github.com/Rebexcode
