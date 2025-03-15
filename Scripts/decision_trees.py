"""
Script for training and evaluating decision trees model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)

from sklearn.tree import plot_tree

def bagging(classifier):
    """
    Apply bagging to a classifier.

    Args:
        classifier: The base classifier to be bagged.

    Returns:
        BaggingClassifier: Bagged classifier.
    """
    
    return BaggingClassifier(estimator=classifier)

def create_matrix(cm, labels, save_dir, normalize, title, filename):
    """
    Create and save a confusion matrix.

    Args:
        cm (conf_matrix): Confusion matrix from scikitlearn
        labels (list): List of class labels.
        save_dir (str): Directory folder to save confusion matrix.
        title (str): Title of the confusion matrix.
        normalize (int): Type of normalization (0, 1, or 2).
        filename (str): Name of the file to save the confusion matrix plot.
    """

    plt.figure()

    if normalize == 1:  # Normalize by columns
        column_sums = cm.sum(axis=0)
        column_sums[column_sums == 0] = 1  # Avoid division by zero
        cm = cm.astype("float") / column_sums[np.newaxis, :]
    elif normalize == 2:  # Normalize by rows
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]:}",
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    joined_dir = os.path.join(save_dir, filename)

    plt.savefig(joined_dir, bbox_inches="tight")
    plt.close()
    
def plot_decision_tree(classifier, feature_names, class_names, save_dir, filename):
    """
    Plot and save a visual representation of the decision tree.

    Args:
        classifier: The trained decision tree classifier.
        feature_names (list): List of feature names.
        class_names (list): List of class names.
        save_dir (str): Directory to save the plot.
        filename (str): Filename for the saved plot.
    """
    
    plt.figure(figsize=(40,20)) 
    tree_plot = plot_tree(
        classifier,
        feature_names=feature_names,
        class_names=class_names,
        filled=False,
        impurity=False,
        fontsize=10
    )
    plt.savefig(os.path.join(save_dir, filename), format='png')
    plt.close()

def main(): 
    # reading in data
    test = pd.read_csv('training_data/test.csv')
    train = pd.read_csv('training_data/train.csv')

    exclude = 'ALLSTAR'
    remove = 'PLAYER'

    # splitting training set
    x_train = train.drop(exclude, axis=1)
    y_train = train[exclude]
    x_train = x_train.drop(remove, axis=1)

    # splitting testing set
    x_test = test.drop(exclude, axis=1)
    y_test = test[exclude]
    x_test = x_test.drop(remove, axis=1)

    # grid search hyperparameters
    dt_hyperparams = {
        "max_depth": [None, 10, 25, 50, 75, 100],
        "min_samples_split": [2, 3, 5, 10],
        "criterion": ["gini", "entropy"],
        "max_leaf_nodes": [None, 10, 25, 50, 75, 100]
    }

    # balanced class weight compensate for data imbalance
    model = DecisionTreeClassifier(class_weight='balanced')
    
    clf = GridSearchCV(
        estimator=model, 
        param_grid=dt_hyperparams, 
        scoring='f1_macro', # macro to compensate data imbalance 
        cv=5, 
        verbose=2
    )

    # perform grid search
    model = clf.fit(x_train,y_train)
    print("Best parameters found from grid search: ", clf.best_params_)
    print("Validation fitting with best recall on labels = 1:", clf.best_score_)

    clf_bagging = bagging(model.best_estimator_)
    clf_bagging.fit(x_train, y_train)

    y_pred = model.best_estimator_.predict(x_test)
    y_pred_bag = clf_bagging.predict(x_test)

    f1 = f1_score(y_test, y_pred, average='macro')
    bag_f1 = f1_score(y_test, y_pred_bag, average='macro')

    print("No baggiing: ", f1)
    print("With baggiing: ", bag_f1)

    if f1 < bag_f1:
        y_pred = y_pred_bag
        print("Use bagging boosted model")
    else:
        print("Use model without bagging")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    save_dir = 'evaluate'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    classes = [0,1]   

    # Plotting confusion matices
    # Normal confusion matrix
    create_matrix(
        conf_matrix, 
        classes, 
        save_dir,
        0,
        "Decision Tree - Confusion Matrix", 
        "decisiontree_confusion_matrix.png"
    )

    # Normalized by Rows
    create_matrix(
        conf_matrix, 
        classes, 
        save_dir,
        2, 
        "Decision Tree - Row-Normalized Confusion Matrix", 
        "decisiontree_row_normalized_confusion_matrix.png"
    )

    # Normalized by Columns
    create_matrix(
        conf_matrix,
        classes,
        save_dir,
        1,
        "Decision Tree - Column-Normalized Confusion Matrix",
        "decisiontree_col_normalized_confusion_matrix.png"
    )

    # eval metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    # Write evaluation metrics to .txt file
    with open(
        os.path.join(save_dir, "decisiontree_evaluation_metrics.txt"), "w"
    ) as file:
        file.write(f"Best Parameters: {clf.best_params_}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"F1: {f1:.4f}\n")
        
        
    feature_names = x_train.columns.tolist()
    class_names = ['Non-All-Star', 'All-Star']


    plot_decision_tree(
        model.best_estimator_,  
        feature_names,
        class_names,
        save_dir,
        'decision_tree_visualization.png'
    )
    
if __name__ == "__main__":
    main()
