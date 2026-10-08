import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load Data
data = pd.read_csv('enhanced_feature_set.csv')
X = data.drop(["class"], axis=1)
y = data["class"]

# Convert labels to binary (defacement=1, benign=0) for ROC
y_bin = (y == 'defacement').astype(int)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

# Train Model
forest = RandomForestClassifier(random_state=42)
forest.fit(X_train, y_train)

# Predict probabilities
y_pred_proba = forest.predict_proba(X_test)[::,1]

# Calculate ROC and AUC
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300)
plt.close()

print("ROC curve generated successfully: roc_curve.png")
