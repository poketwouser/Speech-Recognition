import os
import pickle
import numpy as np
from gmm import GMM
from hmm import HMM

def evaluate(models, X_data, y_data):
    correct = 0
    total = len(y_data)
    
    classes = sorted(list(models.keys()))
    n_classes = len(classes)
    
    max_digit = max(classes)
    confusion_matrix = np.zeros((max_digit + 1, max_digit + 1), dtype=int)
    
    for X, y_true in zip(X_data, y_data):
        best_score = -np.inf
        predicted_digit = -1
        
        for digit, model in models.items():
            score = model.score(X)
            if score > best_score:
                best_score = score
                predicted_digit = digit
                
        if predicted_digit == y_true:
            correct += 1
            
        confusion_matrix[y_true, predicted_digit] += 1
        
    accuracy = correct / total if total > 0 else 0
    
    confusion_matrix = confusion_matrix[classes][:, classes]
    
    return accuracy, confusion_matrix

def print_metrics(name, algo_name, accuracy, cm, classes):
    print(f"\n{algo_name} {name} Results")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(" True\\Pred | " + " ".join([f"{c:4d}" for c in classes]))
    print("-" * (12 + 5 * len(classes)))
    
    per_digit_acc = {}
    for i, true_digit in enumerate(classes):
        row = cm[i]
        total_class = np.sum(row)
        acc_class = row[i] / total_class if total_class > 0 else 0
        per_digit_acc[true_digit] = acc_class
        
        row_str = " ".join([f"{val:4d}" for val in row])
        print(f"{true_digit:11d} | {row_str}")
        
    print("\nPer-digit Accuracy:")
    for digit, acc in per_digit_acc.items():
        print(f"Digit {digit}: {acc * 100:.2f}%")

def main():
    # Load test data
    test_feat_path = './data/test_features.pkl'
    with open(test_feat_path, 'rb') as f:
        test_data = pickle.load(f)
        
    X_test, y_test = test_data['features'], test_data['labels']
    classes = sorted(list(np.unique(y_test)))
    
    # Load val data
    val_feat_path = './data/val_features.pkl'
    with open(val_feat_path, 'rb') as f:
        val_data = pickle.load(f)
    X_val, y_val = val_data['X_val'], val_data['y_val']
        
    # Load Models
    with open('./data/gmms.pkl', 'rb') as f:
        gmms = pickle.load(f)
        
    with open('./data/hmms.pkl', 'rb') as f:
        hmms = pickle.load(f)

    print("Evaluating GMMs")
    gmm_val_acc, gmm_val_cm = evaluate(gmms, X_val, y_val)
    print_metrics("Validation", "GMM", gmm_val_acc, gmm_val_cm, classes)
    
    gmm_test_acc, gmm_test_cm = evaluate(gmms, X_test, y_test)
    print_metrics("Test", "GMM", gmm_test_acc, gmm_test_cm, classes)
    
    print("Evaluating HMMs")
    hmm_val_acc, hmm_val_cm = evaluate(hmms, X_val, y_val)
    print_metrics("Validation", "HMM", hmm_val_acc, hmm_val_cm, classes)
        
    hmm_test_acc, hmm_test_cm = evaluate(hmms, X_test, y_test)
    print_metrics("Test", "HMM", hmm_test_acc, hmm_test_cm, classes)

if __name__ == "__main__":
    main()
