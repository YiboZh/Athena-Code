from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

df = pd.read_csv(PROJECT_ROOT / 'data' / 'vaccine' / 'vaccine_processed_3_1.csv')

print(df['choice_idx'].value_counts())
print("Data shape:", df.shape)

X = df.drop(['choice_idx', 'No.'], axis=1)
y = df['choice_idx']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3
)

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest':        RandomForestClassifier(),
    'XGBoost':              XGBClassifier(),
    'Linear Regression':    LinearRegression(),
}

results = []
for name, model in models.items():
    if name == 'Linear Regression':
        model.fit(X_train, y_train)
        y_pred = (model.predict(X_test) >= 0.5).astype(int)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)


    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        ce = log_loss(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='micro')
    else:
        ce = None
        auc = None

    if ce is not None:
        print(f"{name} — Accuracy: {acc:.3f}, Precision (macro): {prec:.3f}, Recall (macro): {rec:.3f}, F1 (macro): {f1:.3f}, Cross Entropy: {ce:.3f}, AUC: {auc:.3f}")
    else:
        print(f"{name} — Accuracy: {acc:.3f}, Precision (macro): {prec:.3f}, Recall (macro): {rec:.3f}, F1 (macro): {f1:.3f}")
    results.append({
        'model':     name,
        'accuracy':  acc,
        'precision': prec,
        'recall':    rec,
        'f1':        f1,
        'cross_entropy': ce,
        'auc': auc,
    })

metrics_df = pd.DataFrame(results)
metrics_df.to_csv('model_metrics.csv', index=False)
print("\nAll metrics have been saved to model_metrics.csv")

with open('model_details.txt', 'w', encoding='utf-8') as f:
    feature_names = X.columns

    # Linear Regression
    lin = models['Linear Regression']
    lin_coeffs    = lin.coef_
    lin_intercept = lin.intercept_
    f.write("=== Linear Regression ===\n")
    f.write("Intercept:\n")
    f.write(f"  {lin_intercept:.4f}\n")
    f.write("Coefficients:\n")
    for feat, coef in sorted(zip(feature_names, lin_coeffs), key=lambda x: abs(x[1]), reverse=True):
        f.write(f"  {feat}: {coef:.4f}\n")
    terms = " + ".join(f"{coef:.4f}*{name}" for coef, name in zip(lin_coeffs, feature_names))
    f.write("Equation:\n")
    f.write(f"  y = {lin_intercept:.4f} + {terms}\n\n")

    # Logistic Regression (multiclass)
    log = models['Logistic Regression']
    f.write("=== Logistic Regression (multinomial) ===\n")
    n_classes = log.coef_.shape[0]
    for k in range(n_classes):
        intercept_k = log.intercept_[k]
        coefs_k = log.coef_[k]
        f.write(f"[Class {k}] Intercept:\n")
        f.write(f"  {intercept_k:.4f}\n")
        f.write("Coefficients:\n")
        for feat, coef in sorted(zip(feature_names, coefs_k), key=lambda x: abs(x[1]), reverse=True):
            f.write(f"  {feat}: {coef:.4f}\n")
        terms_k = " + ".join(f"{coef:.4f}*{name}" for coef, name in zip(coefs_k, feature_names))
        f.write("Equation:\n")
        f.write(f"  logit(p_class_{k}) = {intercept_k:.4f} + {terms_k}\n\n")

    # Random Forest
    rf = models['Random Forest']
    f.write("=== Random Forest Feature Importances ===\n")
    for feat, imp in sorted(zip(feature_names, rf.feature_importances_), key=lambda x: x[1], reverse=True):
        f.write(f"  {feat}: {imp:.4f}\n")
    f.write("\n")

    # XGBoost
    xgb = models['XGBoost']
    f.write("=== XGBoost Feature Importances ===\n")
    for feat, imp in sorted(zip(feature_names, xgb.feature_importances_), key=lambda x: x[1], reverse=True):
        f.write(f"  {feat}: {imp:.4f}\n")
    f.write("\n")
