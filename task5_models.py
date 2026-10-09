import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_predictions)

plt.figure(figsize=(20, 10))
plot_tree(
    tree_model,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=7
)
plt.title("Decision Tree Visualization (first three levels)")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=200)
plt.close()

depth_values = list(range(1, 16))
train_accuracies = []
test_accuracies = []

for depth in depth_values:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_accuracies.append(accuracy_score(y_train, model.predict(X_train)))
    test_accuracies.append(accuracy_score(y_test, model.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.plot(depth_values, train_accuracies, marker="o", label="Training accuracy")
plt.plot(depth_values, test_accuracies, marker="s", label="Testing accuracy")
plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Depth and Overfitting")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tree_depth_comparison.png", dpi=200)
plt.close()

forest_model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)
forest_accuracy = accuracy_score(y_test, forest_predictions)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tree_cv_scores = cross_val_score(
    DecisionTreeClassifier(max_depth=5, random_state=42),
    X, y, cv=cv, scoring="accuracy"
)
forest_cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    X, y, cv=cv, scoring="accuracy"
)

importance = pd.Series(
    forest_model.feature_importances_, index=data.feature_names
).sort_values(ascending=False)
top_importance = importance.head(10).sort_values()

plt.figure(figsize=(9, 6))
top_importance.plot(kind="barh")
plt.xlabel("Feature Importance")
plt.title("Top 10 Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=200)
plt.close()

print("Task 5: Decision Trees and Random Forests")
print("-" * 45)
print(f"Decision Tree test accuracy: {tree_accuracy:.4f}")
print(f"Random Forest test accuracy: {forest_accuracy:.4f}")
print(f"Decision Tree 5-fold CV mean accuracy: {tree_cv_scores.mean():.4f}")
print(f"Random Forest 5-fold CV mean accuracy: {forest_cv_scores.mean():.4f}")
print("\nDecision Tree Classification Report")
print(classification_report(y_test, tree_predictions, target_names=data.target_names))
print("Random Forest Classification Report")
print(classification_report(y_test, forest_predictions, target_names=data.target_names))
print("Random Forest Confusion Matrix")
print(confusion_matrix(y_test, forest_predictions))
print("\nTop 10 Important Features")
for feature, score in importance.head(10).items():
    print(f"{feature}: {score:.4f}")

ConfusionMatrixDisplay.from_predictions(
    y_test, forest_predictions, display_labels=data.target_names
)
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("random_forest_confusion_matrix.png", dpi=200)
plt.close()
