# Heart Disease Prediction

A machine learning project that predicts the risk of cardiovascular disease from basic health measurements, with a web app built using Streamlit.

**Live demo:** https://heart-disease-prediction321.streamlit.app/
<img width="1092" height="700" alt="image" src="https://github.com/user-attachments/assets/95cb8116-90cf-449e-bc82-3bad73099cc2" />
<img width="967" height="202" alt="image" src="https://github.com/user-attachments/assets/e6958f02-4c83-4a52-a88d-e34ed02c933a" />



>  This is a student project for learning purposes. It is **not** a medical diagnosis tool.

## Dataset

- 70,000 patient records (age, gender, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol, activity).
- Target: `cardio` (0 = no disease, 1 = disease).
- After cleaning (removing impossible blood pressure, height and weight values), **68,610 records** were used (1,390 removed).
- Split: 80% training (54,888) and 20% testing (13,722), stratified.

## What was done

1. **Data cleaning:** converted age from days to years and removed unrealistic values.
2. **Feature engineering:** created `bmi` and `pulse_pressure` (systolic - diastolic).
3. **Feature selection:** 4 methods (ANOVA F-test, Mutual Information, Random Forest importance, RFE) vote on the best features. It is done on training data only, to avoid data leakage.
4. **Models compared:** Logistic Regression, KNN, Decision Tree, Random Forest and a Stacking ensemble.
5. **Evaluation:** Accuracy, F1, MCC, AUC, confusion matrix and ROC curves on a held-out test set.

## Selected features

8 of 13 features were kept (chosen by at least 2 of the 4 methods):

`age`, `ap_hi`, `ap_lo`, `weight`, `bmi`, `pulse_pressure`, `cholesterol`, `gluc`

Dropped: `gender`, `height`, `smoke`, `alco`, `active`.

Seven features were chosen by all 4 methods. `gluc` (glucose) was chosen by 2 methods and kept because blood sugar is medically linked to heart disease.

## Results

| Model | Test Accuracy | F1 | MCC | AUC |
|---|---|---|---|---|
| **Stacking** | **73.4%** | 0.718 | 0.469 | **0.799** |
| Random Forest | 73.3% | 0.715 | 0.467 | 0.798 |
| Decision Tree | 73.0% | 0.714 | 0.462 | 0.793 |
| Logistic Regression | 72.7% | 0.708 | 0.456 | 0.791 |
| KNN | 72.2% | 0.715 | 0.444 | 0.779 |

Best model: **Stacking** (highest test AUC). Its training accuracy is 74.4% and test accuracy is 73.4%, so the model is not overfitting.

The differences between the top models are very small, which suggests that the available features limit the accuracy, not the choice of model.

**Limitation:** for the disease class, precision is 0.75 and recall is 0.69. This means the model misses about 31% of patients who actually have the disease, so it must never be used for real medical decisions.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `cardio_model.joblib` | Trained model and the list of features it uses |
| `requirements.txt` | Python packages needed |
| `main.ipynb` | Full notebook: cleaning, feature selection, training and evaluation |

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit.
