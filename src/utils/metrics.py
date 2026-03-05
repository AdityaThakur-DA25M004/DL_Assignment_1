# This file contains function needed for evaluation

import  numpy as np
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

def calculate_accuracy(y_pred,y_true):
    return np.mean(y_pred==y_true)

def calculate_precision(y_pred,y_true,average="weighted"):
    return precision_score(y_true,y_pred,average=average,zero_division=0)

def calculate_recall(y_pred,y_true,average="weighted"):
    return recall_score(y_true,y_pred,average=average,zero_division=0)


def calculate_f1(y_pred,y_true,average="weighted"):
    return f1_score(y_true,y_pred,average=average,zero_division=0)

def get_confusion_matrix(y_pred,y_true,num_classes=10):
    cm = confusion_matrix(y_true,y_pred,labels=np.arange(num_classes))
    return cm

def evaluate_model(y_pred,y_true,num_classes=10,average="weighted"):
    acc = calculate_accuracy(y_pred,y_true)
    prec = calculate_precision(y_pred,y_true,average=average)
    rec = calculate_recall(y_pred,y_true,average=average)
    f1 = calculate_f1(y_pred,y_true,average=average)
    cm = get_confusion_matrix(y_pred,y_true,num_classes)
    metrics = {
        "accuracy":acc,
        "precision":prec,
        "recall":rec,
        "f1":f1,
        "confusion_matrix":cm
    }
    return metrics

def print_metrics(metrics,prefix=""):
    print(f"\n{prefix} Metrics:")
    print(f" Accuracy:  {metrics['accuracy']:.4f}")
    print(f" Precision: {metrics['precision']:.4f}")
    print(f" Recall:    {metrics['recall']:.4f}")
    print(f" F1-Score:  {metrics['f1']:.4f}")

def per_class_metrics(Y_pred, Y_true, num_classes=10):
    per_class_results = {}

    for class_id in range(num_classes):

        # Convert multi-class problem into binary:
        # Current class vs all other classes
        y_pred_binary = (Y_pred == class_id).astype(int)
        y_true_binary = (Y_true == class_id).astype(int)

        # Compute metrics safely (avoid division-by-zero errors)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)

        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)

        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        # Store results
        per_class_results[class_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return per_class_results