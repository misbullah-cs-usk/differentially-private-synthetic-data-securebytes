# Differentially Private Synthetic Data Generation: A Comparison of rmckenna-style and DPSyn-style Methods on the Adult Dataset

## Course Name: Data Privacy and Security (1142CS5164701)
### Group Name: SecureBytes
Member:
1. Alim Misbullah D11415803	
2. Laina Farsiah D11415802
3. Stenly Ibrahim Adam D11215809
4. Aurelio Naufal Effendy M11415802

## Project Overview
This project studies **Differentially Private Synthetic Data (DPSD)** using the **Adult Income Dataset** and compares two synthetic-data generation approaches:
1. **R-McKenna Method**  
   A graphical-model based differentially private synthetic data generator inspired by the NIST repository and Private-PGM framework.

2. **DPSyn Method**  
   A marginal-based differentially private synthetic data generator based on noisy marginals, consistency repair, and synthetic record reconstruction.
   
The goal of this project is to compare both methods in terms of:
- Privacy protection
- Data utility
- Machine learning performance
- Practical usability
- Comparison with previous homework:
  - **HW1:** k-anonymity anonymization
  - **HW2:** DP-SGD during model training
---

## Problem Statement

Traditional anonymization methods such as **k-anonymity** and **l-diversity** modify original records but may still be vulnerable to linkage attacks or background knowledge attacks.

Differential Privacy (DP) offers a stronger mathematical guarantee by limiting the influence of any individual record on the released output.

Instead of releasing modified real records, this project releases:

> **Synthetic datasets generated under differential privacy**

These datasets preserve useful statistical patterns while protecting individual privacy.

---

## Dataset

### Adult Income Dataset

Predict whether annual income exceeds \$50K using census-style attributes.

Features include:

- age
- education
- occupation
- marital-status
- relationship
- race
- sex
- hours-per-week
- native-country
- capital-gain
- capital-loss
- income (target)

---

## Compared Methods

| Method | Type | Privacy Stage | Output |
|------|------|------|------|
| HW1 | k-anonymity | Data release | Generalized real records |
| HW2 | DP-SGD | Model training | Private trained model |
| HW3-A | R-McKenna | Synthetic data generation | DP synthetic dataset |
| HW3-B | DPSyn | Synthetic data generation | DP synthetic dataset |

---

## Privacy Parameters

We evaluate multiple privacy budgets:

```
epsilon = 0.1, 0.5, 1.0, 2.0, 5.0
delta   = 1e-5
```

Interpretation:
* Smaller epsilon → stronger privacy
* Larger epsilon → better utility

## Evaluation Models

Synthetic datasets are used to train:
* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)
* Multi-layer Perceptron (MLP)

Metrics:
* Accuracy
* Misclassification Rate
* Precision
* Recall
* AUC

Repository Structure
```
differentially-private-sythetic-data-securebytes/
│── rmckenna_dp_synth_adult.py
│── dpsyn_adult.py
│── compare_hw1_hw2_hw3_all.py
│── adult.csv
│── hw3_outputs/
│── hw3_dpsyn_outputs/
│── final_all_outputs/
│── README.md
```


References
NIST Differentially Private Synthetic Data
Private-PGM / McKenna et al.
DPSyn papers
Adult Dataset (UCI)
