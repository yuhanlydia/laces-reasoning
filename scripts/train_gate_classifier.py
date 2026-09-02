#!/usr/bin/env python3
"""P0: Train confidence gate classifier for state injection.

Predicts whether injecting Z into 0.4B will improve or hurt acceptance.
Lightweight MLP, fast training, can run on CPU.
"""
import json, sys, pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

data = json.load(open("outputs_eval/gate_training_data.json"))

# Features
X = np.array([[d["z_norm"]] for d in data])
y_help = np.array([d["gate_label"] for d in data])
y_cat  = np.array([d["catastrophic"] for d in data])

# Train/test split
n = len(data)
split = int(n * 0.8)
perm = np.random.RandomState(42).permutation(n)
X_tr, X_te = X[perm[:split]], X[perm[split:]]
y_tr_h, y_te_h = y_help[perm[:split]], y_help[perm[split:]]
y_tr_c, y_te_c = y_cat[perm[:split]], y_cat[perm[split:]]

print(f"Train: {split}, Test: {n - split}")
print(f"Help rate: {y_help.mean():.1%}, Cat rate: {y_cat.mean():.1%}")

# Simple threshold baseline: predict based on z_norm
for thresh in [25, 28, 30, 32, 35]:
    preds = (X[:, 0] < thresh).astype(int)
    acc = (preds == y_help).mean()
    print(f"  z_norm < {thresh}: acc={acc:.3f}")

# Logistic regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_tr, y_tr_h)
preds = lr.predict(X_te)
print(f"\nLogisticRegression (predict help):")
print(f"  acc={accuracy_score(y_te_h, preds):.3f} f1={f1_score(y_te_h, preds):.3f}")
print(f"  coef={lr.coef_[0][0]:.4f} intercept={lr.intercept_[0]:.4f}")

# Random forest with more features
X2 = np.array([[d["z_norm"], d["raw_top5"], d["inj_top5"]] for d in data])
X2_tr, X2_te = X2[perm[:split]], X2[perm[split:]]

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X2_tr, y_tr_h)
preds = rf.predict(X2_te)
print(f"\nRandomForest (predict help, 3 features):")
print(f"  acc={accuracy_score(y_te_h, preds):.3f} f1={f1_score(y_te_h, preds):.3f}")

# Predict catastrophic (more important!)
rf_cat = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_cat.fit(X2_tr, y_tr_c)
preds_cat = rf_cat.predict(X2_te)
print(f"\nRandomForest (predict catastrophic):")
print(f"  acc={accuracy_score(y_te_c, preds_cat):.3f} f1={f1_score(y_te_c, preds_cat):.3f}")
print(classification_report(y_te_c, preds_cat, target_names=["safe", "catastrophic"]))

# Save models
Path("outputs_eval").mkdir(exist_ok=True)
pickle.dump({"model": rf_cat, "features": ["z_norm", "raw_top5", "inj_top5"]},
            open("outputs_eval/gate_catastrophic_model.pkl", "wb"))
pickle.dump({"model": rf, "features": ["z_norm", "raw_top5", "inj_top5"]},
            open("outputs_eval/gate_help_model.pkl", "wb"))
print("\nModels saved: gate_catastrophic_model.pkl, gate_help_model.pkl")
