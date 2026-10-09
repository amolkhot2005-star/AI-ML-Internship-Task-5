# Task 5: Decision Trees and Random Forests

This project trains and compares a Decision Tree Classifier and a Random Forest Classifier using the Breast Cancer Wisconsin dataset included with scikit-learn.

## Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the program:

```bash
python task5_models.py
```

## What it does

- Splits the dataset into training and testing sets.
- Trains a Decision Tree and visualizes its first three levels.
- Compares accuracy at different tree depths to demonstrate overfitting.
- Trains a Random Forest and compares its accuracy with the Decision Tree.
- Prints classification metrics and a confusion matrix.
- Plots the top 10 Random Forest feature importances.
- Evaluates both models using 5-fold cross-validation.

## Output files

The program creates `decision_tree.png`, `tree_depth_comparison.png`, `feature_importance.png`, and `random_forest_confusion_matrix.png`. Evaluation results are printed in the terminal.
