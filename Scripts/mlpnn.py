"""
Script for training and evaluating neural network model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)

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
        normalize (int): Type of normalization (0, 1, or 2).
        title (str): Title of the confusion matrix.
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

    # hyperparameter grid
    mlp_hyperparams = {
        'hidden_layer_sizes': [(50,), (100,), (150,), # 1 hidden layer
                            (70, 50), (50, 70), (60, 60), # 2 hidden layers
                            (50, 30, 20), (40, 40, 30), # 3 hidden layers
                            (50, 35, 25, 15)], # 4 hidden layers
        'activation': ['tanh', 'relu', 'logistic'],
        'learning_rate': ['constant', 'adaptive'],
    }

    # initialize model
    mlp = MLPClassifier()

    # Set up GridSearchCV
    clf = GridSearchCV(
        estimator=mlp, 
        param_grid=mlp_hyperparams, 
        scoring='f1_macro', # macro to compensate data imbalance 
        cv=5, 
        verbose=2
    )

    # Fit the model with training data
    clf.fit(x_train, y_train)

    # Best model's parameters
    print("Best parameters:", clf.best_params_)
    print("Best Label 1 Recall", clf.best_score_)

    best_mlp = MLPClassifier(**clf.best_params_)
    best_mlp.fit(x_train, y_train)

    save_dir = 'evaluate'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plotting the loss curve
    plt.figure()
    plt.plot(best_mlp.loss_curve_)
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # Peform bagging on best model
    clf_bagging = bagging(best_mlp)
    clf_bagging.fit(x_train, y_train)

    y_pred = best_mlp.predict(x_test)

    y_pred_bag = clf_bagging.predict(x_test)

    f1 = f1_score(y_test, y_pred, average='macro')
    bag_f1 = f1_score(y_test, y_pred_bag, average='macro')

    print("No baggiing: ", f1)
    print("With baggiing: ", bag_f1)

    # select which model to plot confusion matrices on
    if f1 < bag_f1:
        y_pred = y_pred_bag
        print("Use bagging boosted model")
    else:
        print("Use model without bagging")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
        
    classes = [0,1]
    
    # Normal confusion matrix   
    create_matrix(
        conf_matrix, 
        classes,
        save_dir, 
        0, 
        "Neural Network - Confusion Matrix", 
        "nn_confusion_matrix.png"
    )

    # Normalized by Rows
    create_matrix(
        conf_matrix, 
        classes,
        save_dir, 
        2,
        "Neural Network - Row-Normalized Confusion Matrix",  
        "nn_row_normalized_confusion_matrix.png"
    )

    # Normalized by Columns
    create_matrix(
        conf_matrix,
        classes,
        save_dir,
        1,
        "Neural Network - Column-Normalized Confusion Matrix",
        "nn_col_normalized_confusion_matrix.png"
    )

    # eval metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    # Write evaluation metrics to .txt file
    with open(
        os.path.join(save_dir, "nn_evaluation_metrics.txt"), "w"
    ) as file:
        file.write(f"Best Parameters: {clf.best_params_}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"F1: {f1:.4f}\n")
    
if __name__ == "__main__":
    main()
