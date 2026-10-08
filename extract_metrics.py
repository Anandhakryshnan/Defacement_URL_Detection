import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load dataset
data = pd.read_csv('enhanced_feature_set.csv')
X = data.drop(["class"], axis=1)
y = data["class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
forest = RandomForestClassifier(random_state=42)
forest.fit(X_train, y_train)

# Evaluate
y_test_pred = forest.predict(X_test)
pos_label = 'defacement'

print("Accuracy:", metrics.accuracy_score(y_test, y_test_pred))
print("Precision:", metrics.precision_score(y_test, y_test_pred, pos_label=pos_label))
print("Recall:", metrics.recall_score(y_test, y_test_pred, pos_label=pos_label))
print("F1-Score:", metrics.f1_score(y_test, y_test_pred, pos_label=pos_label))

print("\nClassification Report:")
print(metrics.classification_report(y_test, y_test_pred))

print("\nFeature Importances:")
importances = forest.feature_importances_
feature_names = X.columns
for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance:.4f}")
