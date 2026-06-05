"""
SignVox — Model Training Script
Trains a Random Forest classifier on collected hand landmark data.
"""

import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

DATA_DIR = "data"
MODEL_DIR = "models"


def main():
    # Load collected data
    print("=" * 55)
    print("  SignVox - Model Training")
    print("=" * 55)

    landmarks_path = os.path.join(DATA_DIR, "landmarks.npy")
    labels_path = os.path.join(DATA_DIR, "labels.npy")

    if not os.path.exists(landmarks_path) or not os.path.exists(labels_path):
        print("  ERROR: No training data found!")
        print(f"  Run collect_data.py first to collect data.")
        return

    # Load data
    X = np.load(landmarks_path)
    y = np.load(labels_path)

    print(f"\n  Data loaded successfully!")
    print(f"  Total samples: {len(X)}")
    print(f"  Features per sample: {X.shape[1]}")
    print(f"  Letters in dataset: {sorted(set(y))}")
    print(f"  Number of letters: {len(set(y))}")

    # Show samples per letter
    print("\n  Samples per letter:")
    for letter in sorted(set(y)):
        count = list(y).count(letter)
        bar = "█" * (count // 10)
        print(f"    {letter}: {count:>4} {bar}")

    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Testing samples:  {len(X_test)}")

    # Train Random Forest model
    print("\n  Training model... ", end="", flush=True)
    model = RandomForestClassifier(
        n_estimators=200,       # 200 decision trees
        max_depth=20,
        random_state=42,
        n_jobs=-1               # Use all CPU cores
    )
    model.fit(X_train, y_train)
    print("DONE!")

    # Evaluate on test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n  ╔══════════════════════════════╗")
    print(f"  ║   MODEL ACCURACY: {accuracy * 100:.1f}%     ║")
    print(f"  ╚══════════════════════════════╝")

    # Detailed report
    print("\n  Detailed Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "signvox_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Also save the label list
    labels_list = sorted(set(y))
    labels_path = os.path.join(MODEL_DIR, "labels.pkl")
    with open(labels_path, "wb") as f:
        pickle.dump(labels_list, f)

    print(f"\n  Model saved to: {model_path}")
    print(f"  Labels saved to: {labels_path}")
    print(f"\n  You can now run signvox.py to test it live!")
    print("=" * 55)


if __name__ == "__main__":
    main()