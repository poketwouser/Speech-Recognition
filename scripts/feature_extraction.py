import os
import librosa
import numpy as np
import pickle

# Hypertune parameters, n_mfcc=13, n_fft=n_fft, hop_length=hop_length

def extract_features_from_dir(audio_dir):
    """
    Iterates through .wav files in the given directory and extracts MFCC features.
    
    Args:
        audio_dir (str): Path to the directory containing .wav files.
        
    Returns:
        tuple: (list of MFCC sequences, list of labels)
    """
    mfcc_sequences = []
    labels = []
    extension = ('.wav')
        
    # Get all files in directory and sort for consistency
    files = sorted([f for f in os.listdir(audio_dir) if f.endswith(extension)])
    
    print(f"Processing {len(files)} files in {audio_dir}")
    
    for filename in files:
        file_path = os.path.join(audio_dir, filename)
        
        # Load the audio signal
        # Use sr=None to preserve native sampling rate
        y, sr = librosa.load(file_path, sr=16000)
            
        # Framing parameters
        # 25 ms duration for window length, 10 ms for hop length
        n_fft = int(0.025 * sr) 
        hop_length = int(0.010 * sr)
            
        # 13 MFCCs per frame
        # Use n_mfcc=13 per frame. n_fft handles the 25ms window.
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        mfccs_transpose = mfccs.T
            
        # Digit label
        digit_label = int(filename.split('_')[0])
            
        mfcc_sequences.append(mfccs_transpose.astype(np.float32))
        labels.append(digit_label)
            
    return mfcc_sequences, labels

def save_dataset(data, labels, filepath):
    """Saves the extracted features and labels to a pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump({'features': data, 'labels': labels}, f)
    print(f"Dataset saved to {filepath}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    recordings_dir = os.path.join(base_dir, 'recordings')
    train_dir = os.path.join(recordings_dir, 'train')
    test_dir = os.path.join(recordings_dir, 'test')
    
    data_output_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_output_dir, exist_ok=True)
    
    print("Extracting Features")

    X_train, y_train = extract_features_from_dir(train_dir)
    save_dataset(X_train, y_train, os.path.join(data_output_dir, 'train_features.pkl'))
    
    X_test, y_test = extract_features_from_dir(test_dir)
    save_dataset(X_test, y_test, os.path.join(data_output_dir, 'test_features.pkl'))
    
    print(f"Training sequences: {len(X_train)}")
    print(f"Sequence shape example: {X_train[0].shape} (T, 13)")
    print(f"Testing sequences: {len(X_test)}")
    print(f"Sequence shape example: {X_test[0].shape} (T, 13)")

if __name__ == "__main__":
    main()
