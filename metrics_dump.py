import os
import pickle
import numpy as np
import json
from gmm import GMM
from hmm import HMM
from inference import evaluate

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    with open(os.path.join(data_dir, 'test_features.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    X_test, y_test = test_data['features'], test_data['labels']
    
    with open(os.path.join(data_dir, 'val_features.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    X_val, y_val = val_data['X_val'], val_data['y_val']
    
    with open(os.path.join(data_dir, 'gmms.pkl'), 'rb') as f:
        gmms = pickle.load(f)
        
    with open(os.path.join(data_dir, 'hmms.pkl'), 'rb') as f:
        hmms = pickle.load(f)

    gmm_val_acc, _ = evaluate(gmms, X_val, y_val)
    gmm_test_acc, _ = evaluate(gmms, X_test, y_test)
    hmm_val_acc, _ = evaluate(hmms, X_val, y_val)
    hmm_test_acc, _ = evaluate(hmms, X_test, y_test)
    
    with open('metrics.json', 'w') as f:
        json.dump({
            "GMM_Val_Acc": gmm_val_acc,
            "GMM_Test_Acc": gmm_test_acc,
            "HMM_Val_Acc": hmm_val_acc,
            "HMM_Test_Acc": hmm_test_acc
        }, f)

if __name__ == "__main__":
    main()
