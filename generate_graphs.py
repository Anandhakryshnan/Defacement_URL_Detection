import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree

# 1. Load Data
data = pd.read_csv('enhanced_feature_set.csv')
X = data.drop(["class"], axis=1)
y = data["class"]

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
forest = RandomForestClassifier(random_state=42)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

# --- GRAPH 1: Feature Importances ---
importances = forest.feature_importances_
feature_names = X.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10) # Top 10

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, hue='Feature', palette='viridis', legend=False)
plt.title('Top 10 Feature Importances in Random Forest')
plt.xlabel('Gini Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()

# --- GRAPH 2: Confusion Matrix ---
cm = metrics.confusion_matrix(y_test, y_pred, labels=['benign', 'defacement'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Defacement'], yticklabels=['Benign', 'Defacement'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()

# --- GRAPH 3: Single Decision Tree (Simplified for visibility) ---
individual_tree = forest.estimators_[0]
plt.figure(figsize=(24, 10))
tree.plot_tree(individual_tree, max_depth=2, filled=True, feature_names=X.columns, class_names=['benign', 'defacement'], fontsize=12, rounded=True, precision=2)
plt.title('Sample Decision Tree from Random Forest (Depth 2)')
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=300)
plt.close()

print("Graphs successfully generated: feature_importance.png, confusion_matrix.png, decision_tree.png")
