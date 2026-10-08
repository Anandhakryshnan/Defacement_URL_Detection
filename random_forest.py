import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 


# Loading data into dataframe
data = pd.read_csv('enhanced_feature_set.csv')
data.head()

# Splitting the dataset into dependent and independent features
X = data.drop(["class"], axis=1)
y = data["class"]

# Splitting the dataset into train and test sets: 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
forest = RandomForestClassifier()

# Fit the model
forest.fit(X_train, y_train)

# Save the trained Random Forest model to a .pkl file
model_path = 'random_forest_model.pkl'
joblib.dump(forest, model_path)

# Load the saved Random Forest model
loaded_model = joblib.load('random_forest_model.pkl')

# Predicting the target value from the model for the samples
y_train_forest = loaded_model.predict(X_train)
y_test_forest = loaded_model.predict(X_test)

# Specify the positive label explicitly
pos_label = 'defacement'

# Computing the accuracy, f1_score, recall, precision of the model performance
acc_train_forest = metrics.accuracy_score(y_train, y_train_forest)
acc_test_forest = metrics.accuracy_score(y_test, y_test_forest)
print("Random Forest: Accuracy on training Data: {:.3f}".format(acc_train_forest))
print("Random Forest: Accuracy on test Data: {:.3f}".format(acc_test_forest))
print()

f1_score_train_forest = metrics.f1_score(y_train, y_train_forest, pos_label=pos_label)
f1_score_test_forest = metrics.f1_score(y_test, y_test_forest, pos_label=pos_label)
print("Random Forest: f1_score on training Data: {:.3f}".format(f1_score_train_forest))
print("Random Forest: f1_score on test Data: {:.3f}".format(f1_score_test_forest))
print()

recall_score_train_forest = metrics.recall_score(y_train, y_train_forest, pos_label=pos_label)
recall_score_test_forest = metrics.recall_score(y_test, y_test_forest, pos_label=pos_label)
print("Random Forest: Recall on training Data: {:.3f}".format(recall_score_train_forest))
print("Random Forest: Recall on test Data: {:.3f}".format(recall_score_test_forest))
print()

precision_score_train_forest = metrics.precision_score(y_train, y_train_forest, pos_label=pos_label)
precision_score_test_forest = metrics.precision_score(y_test, y_test_forest, pos_label=pos_label)
print("Random Forest: precision on training Data: {:.3f}".format(precision_score_train_forest))
print("Random Forest: precision on test Data: {:.3f}".format(precision_score_test_forest))

# Computing the classification report of the model
print(metrics.classification_report(y_test, y_test_forest, target_names=['benign', 'defacement']))

# Visualizing an individual tree from the Random Forest
individual_tree = loaded_model.estimators_[0]  # Selecting the first tree
plt.figure(figsize=(20, 10))
tree.plot_tree(individual_tree, filled=True, feature_names=X.columns, class_names=['benign', 'defacement'])
plt.show()
