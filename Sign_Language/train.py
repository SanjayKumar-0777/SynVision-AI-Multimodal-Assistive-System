import os
import pickle
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.hand_utils import detect_and_process_hand

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

def train_model():
    print("Loading data...")
    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print("Dataset directory is empty or missing!")
        return

    data = []
    labels = []
    class_names = sorted(os.listdir(DATASET_DIR))
    
    # Save class names mapping
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    with open(os.path.join(MODEL_DIR, "labels.txt"), "w") as f:
        for name in class_names:
            f.write(name + "\n")
            
    print(f"Classes found: {class_names}")

    for idx, name in enumerate(class_names):
        path = os.path.join(DATASET_DIR, name)
        print(f"Processing class: {name}")
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                
                # Extract landmarks
                # detect_and_process_hand now returns (landmarks_list, annotated_frame)
                landmarks, _ = detect_and_process_hand(img)
                
                if landmarks is not None:
                    data.append(landmarks)
                    labels.append(idx)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                pass

        print(f"Collected samples so far: {len(data)}")
                
    if len(data) == 0:
        print("No valid hand landmarks found in dataset! Make sure images contain visible hands.")
        return

    data = np.asarray(data, dtype=np.float32)
    labels = np.asarray(labels)

    # Remove any NaN or Inf samples (if they sneaked in)
    valid_mask = np.isfinite(data).all(axis=1)
    data = data[valid_mask]
    labels = labels[valid_mask]

    print(f"Training on {len(data)} base samples with {len(class_names)} classes.")

    # Data Augmentation (simulating camera noise and slight orientation variance)
    # We create copies of the data and add small Gaussian noise to the points
    noise_factor = 0.005 # Small noise relative to normalized coordinates
    augmented_data = data + np.random.normal(0, noise_factor, data.shape)
    
    noisy_data = np.vstack((data, augmented_data))
    noisy_labels = np.concatenate((labels, labels))
    
    print(f"Total samples after augmentation: {len(noisy_data)}")

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(noisy_data, noisy_labels, test_size=0.2, shuffle=True, stratify=noisy_labels)

    # Train Random Forest
    # n_estimators=200 for robustness, class_weight='balanced' handles imbalanced datasets gracefully
    # max_depth=15 prevents trees from memorizing the exact training noise (overfitting)
    model = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42)
    model.fit(x_train, y_train)

    # Evaluate
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print(f"Model Accuracy on Augmented Validation Set: {score * 100:.2f}%")

    # Save model
    with open(os.path.join(MODEL_DIR, "model.p"), "wb") as f:
        pickle.dump({'model': model}, f)
        
    print("Model saved successfully as model.p!")

if __name__ == "__main__":
    train_model()
