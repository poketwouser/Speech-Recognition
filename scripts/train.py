import os
import pickle
import numpy as np
from gmm import GMM
from hmm import HMM

def train_test_split_stratified(features, labels, test_ratio=0.2):
    classes = np.unique(labels)
    X_train, X_val = [], []
    y_train, y_val = [], []
    
    for c in classes:
        idx_c = [i for i, lbl in enumerate(labels) if lbl == c]
        np.random.shuffle(idx_c)
        
        n_val = int(len(idx_c) * test_ratio)
        idx_val = idx_c[:n_val]
        idx_train = idx_c[n_val:]
        
        for i in idx_train:
            X_train.append(features[i])
            y_train.append(labels[i])
            
        for i in idx_val:
            X_val.append(features[i])
            y_val.append(labels[i])
            
    return X_train, X_val, y_train, y_val

def main():
    train_feat_path = './data/train_features.pkl'
    
    print("Loading training data")
    with open(train_feat_path, 'rb') as f:
        data = pickle.load(f)
        
    features = data['features']
    labels = data['labels']
    
    print("Splitting data into 80/20 train/validation sets")
    np.random.seed(42) # For reproducibility
    X_train, X_val, y_train, y_val = train_test_split_stratified(features, labels, test_ratio=0.2)
    
    # Save the split
    val_data_path = './data/val_features.pkl'
    with open(val_data_path, 'wb') as f:
        pickle.dump({'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}, f)
    
    classes = sorted(list(np.unique(y_train)))
    
    gmms = {}
    hmms = {}
    
    print("\nTraining GMMs")
    for c in classes:
        print(f"Training GMM for digit {c}")
        # One GMM Model for every class
        X_c = [X for X, y in zip(X_train, y_train) if y == c]
        gmm = GMM(n_components=8, n_iter=50)
        gmm.fit(X_c)
        gmms[c] = gmm
        
    print("\nTraining HMMs")
    for c in classes:
        print(f"Training HMM for digit {c}")
        # One HMM Model for every class
        X_c = [X for X, y in zip(X_train, y_train) if y == c]
        hmm = HMM(n_states=5, n_iter=50)
        hmm.fit(X_c)
        hmms[c] = hmm
        
    print("\nSaving models")
    with open('./data/gmms.pkl', 'wb') as f:
        pickle.dump(gmms, f)
        
    with open('./data/hmms.pkl', 'wb') as f:
        pickle.dump(hmms, f)
        
    print("Training complete")

if __name__ == "__main__":
    main()
