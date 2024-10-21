import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_file(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

def calculate_metrics(y_true, y_pred):
    classes = sorted(set(y_true + y_pred))
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t]][class_to_idx[p]] += 1
    
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1_score = np.zeros(n_classes)
    
    for i in range(n_classes):
        precision[i] = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() != 0 else 0
        recall[i] = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() != 0 else 0
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0
    
    accuracy = np.trace(cm) / np.sum(cm)
    
    return classes, precision, recall, f1_score, accuracy, cm

def print_metrics(classes, precision, recall, f1_score, accuracy, total):
    print(f"{'':>10} {'precision':>10} {'recall':>10} {'f1-score':>10} {'total':>10}")
    for c, p, r, f, t in zip(classes, precision, recall, f1_score, total):
        print(f"{c:>10} {p:>10.2f} {r:>10.2f} {f:>10.2f} {t:>10d}")
    print(f"{'accuracy':>10} {accuracy:>10.2f} {sum(total):>10d}")

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) != 3:
        print("Usage: python Confusion_Matrix.py predictions.txt truth.txt")
        sys.exit(1)

    predictions = read_file(sys.argv[1])
    truth = read_file(sys.argv[2])

    if len(predictions) != len(truth):
        print("Error: Number of predictions and truth values do not match.")
        sys.exit(1)

    classes, precision, recall, f1_score, accuracy, cm = calculate_metrics(truth, predictions)
    total = cm.sum(axis=1)

    print_metrics(classes, precision, recall, f1_score, accuracy, total)
    print("\nConfusion Matrix:")
    print(cm)

    plot_confusion_matrix(cm, classes)

if __name__ == "__main__":
    main()
