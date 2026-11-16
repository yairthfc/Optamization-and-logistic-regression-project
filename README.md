# Optimization & Logistic Regression From Scratch: Gradient Descent, Learning Rates & Regularization

This project implements a **full optimization framework from scratch**, including:

- Gradient Descent engine (with callbacks, tolerance, multiple output modes)
- Fixed & exponential learning-rate schedules
- Base module system for objective functions (L1, L2, logistic loss, regularized objectives)
- Full Logistic Regression implementation (with L1/L2 regularization)
- Cross‑validation from scratch
- ROC curve evaluation + optimal threshold
- Analysis of different learning rates on L1/L2 objectives

Everything is implemented by hand with NumPy. **No scikit-learn models** (only a metrics/ROC plotting helper).

---

# Project Structure

```
optimization-from-scratch/
│
├── src/
│   ├── base_estimator.py            # scikit-learn-style estimator API
│   ├── base_learning_rate.py        # Base class for LR strategies
│   ├── base_module.py               # Base class for differentiable objective functions
│   ├── learning_rate.py             # FixedLR, ExponentialLR implementations
│   ├── gradient_descent.py          # Full GD implementation with callbacks
│   ├── cross_validate.py            # Manual K-fold cross validation
│   ├── logistic_regression.py       # Logistic regression using gradient descent
│   ├── loss_functions.py            # MSE, cross entropy, misclassification, softmax
│   ├── gradient_descent_investigation.py
│   └── __init__.py
│
├── docs/
│   └── Answers.pdf                  # Theory, derivations & written answers
│
├── requirements.txt
└── README.md
```

---

# Implemented Components

## BaseEstimator  
Located in: `src/base_estimator.py`

Provides a scikit-learn-style interface:

- `fit(X, y)`
- `predict(X)`
- `loss(X, y)`
- `fit_predict(X, y)`

All real functionality is implemented through `_fit`, `_predict`, `_loss`.  

---

## BaseModule (Objective Function API)
Located in: `src/base_module.py`  
Defines:

- `compute_output(X, y)` → objective value  
- `compute_jacobian(X, y)` → gradient w.r.t. parameters  
- Accessible `.weights` property  

Used for: logistic regression, L1/L2 modules, regularized modules.  

---

## Learning Rate Strategies  
Located in: `src/base_learning_rate.py` + `src/learning_rate.py`

Includes:

### **FixedLR**: constant learning rate  

### **ExponentialLR**: decaying step size  
Implements:  
\[
\eta_t = \eta_0 \cdot \gamma^t
\]

---

## Gradient Descent Engine  
Located in: `src/gradient_descent.py`  

Features:

- Custom LR strategies (`lr_step`)
- Tolerance‑based stopping
- Max iteration control
- Output modes:
  - `"last"`
  - `"best"`
  - `"average"`
- Full callback system (records value, gradient, weights, learning rate, etc.)
- Tracks best solution during training

This is the backbone used by logistic regression.

---

## Cross‑Validation (Manual)  
Located in: `src/cross_validate.py`  

Implements K‑fold CV:

- Random folding
- Fit‑predict inside each fold
- Computes average train + validation score

---

## Logistic Regression (With Regularization)
Located in: `src/logistic_regression.py`  

Features:

- Gradient descent optimization  
- Optional intercept  
- Regularization penalties: `"none"`, `"l1"`, `"l2"`  
- Custom modules:
  - `LogisticModule`
  - `RegularizedModule`
  - `L1`, `L2`
- Outputs:
  - `predict`
  - `predict_proba`
  - `loss` (misclassification)

Used heavily in `gradient_descent_investigation.py`.

---

## Loss Functions  
Located in: `src/loss_functions.py`  

Includes:

- Mean Square Error  
- Misclassification Error  
- Accuracy  
- Cross‑Entropy  
- Softmax  

---

# Analysis & Experiments

## 1️⃣ Gradient Descent on L1 and L2 objectives

The script `gradient_descent_investigation.py` explores:

- Different fixed learning rates
- Convergence rates
- Descent path visualization (using Plotly)
- GD’s behavior on nonsmooth (L1) vs smooth (L2) objectives

It produces:

- Descent path figures  
- Convergence curve per learning rate  
- Objective value evolution  


---

## 2️⃣ Logistic Regression on the SA Heart Dataset

The investigation includes:

- Fitting logistic regression with GD
- ROC curve and AUC
- Finding optimal threshold maximizing TPR − FPR
- Train/test evaluation
- Cross‑validation over regularization parameter λ

Outputs:

- ROC plot  
- Optimal threshold  
- Best λ selection  
- Test error with optimal λ  

---

# Theoretical PDF

Found in: `docs/Answers.pdf`

Includes:

- GD derivations  
- Regularization theory  
- Logistic model likelihood  
- Gradient computation  
- ROC theory  
- Cross‑validation justification  
- Plots used in the assignment  

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Usage

### Run Gradient Descent Investigation:
```bash
python src/gradient_descent_investigation.py
```

### Fit Logistic Regression:
Inside `gradient_descent_investigation.py`, run:
```python
fit_logistic_regression()
```

### Or import manually:
```python
from logistic_regression import LogisticRegression
from gradient_descent import GradientDescent
from learning_rate import FixedLR

model = LogisticRegression(solver=GradientDescent(FixedLR(1e-3)))
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

---

# Technologies Used

- Python  
- NumPy  
- Pandas  
- Plotly  
- Matplotlib  
- scikit‑learn (only for roc_curve/auc helper)  

---

# Learning Outcomes

This project demonstrates:

- Implementing full optimization pipelines  
- Custom gradient descent framework  
- Learning rate schedules  
- Cross‑validation from scratch  
- Logistic regression theory & implementation  
- Regularization (L1 vs L2)  
- Practical convergence analysis  
- Callback‑based solver design  

---

# License

MIT License.

---

# Author  
**Yair Mahfud**
