"""
SignVox — Data Collection Script (IMPROVED)
Collects hand landmark data for sign language letters.
Now APPENDS to existing data instead of overwriting!
"""

import cv2
import numpy as np
import os
import time
from hand_detector import HandDetector

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SAMPLES_PER_LETTER = 200
DATA_DIR = "data"


def load_existing_data():
    """Load existing data if available."""
    landmarks_path = os.path.join(DATA_DIR, "landmarks.npy")
    labels_path = os.path.join(DATA_DIR, "labels.npy")

    if os.path.exists(landmarks_path) and os.path.exists(labels_path):
        landmarks = list(np.load(landmarks_path))
        labels = list(np.load(labels_path))
        print(f"  Loaded existing data: {len(landmarks)} samples")
        print(f"  Letters already recorded: {sorted(set(labels))}")
        return landmarks, labels
    else:
        print("  No existing data found. Starting fresh.")
        return [], []


def save_data(all_landmarks, all_labels):
    """Save data to files."""
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, "landmarks.npy"), np.array(all_landmarks))
    np.save(os.path.join(DATA_DIR, "labels.npy"), np.array(all_labels))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot access webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandDetector(model_path="hand_landmarker.task", max_hands=1)

    cv2.namedWindow("SignVox - Data Collection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("SignVox - Data Collection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("=" * 55)
    print("  SignVox - Data Collection (IMPROVED)")
    print("=" * 55)

    # Load existing data first!
    all_landmarks, all_labels = load_existing_data()

    # Show which letters already have data
    existing_letters = {}
    for label in all_labels:
        existing_letters[label] = existing_letters.get(label, 0) + 1

    print(f"\n  Collecting {SAMPLES_PER_LETTER} samples per letter")
    print("  Controls:")
    print("    SPACE  = Start/Pause recording")
    print("    N      = Skip to next letter")
    print("    R      = Redo current letter (deletes old data for it)")
    print("    Q      = Quit and save")
    print("=" * 55)

    letter_index = 0

    while letter_index < len(LABELS):
        current_letter = LABELS[letter_index]

        # Check if this letter already has data
        existing_count = existing_letters.get(current_letter, 0)
        collected = 0
        recording = False

        if existing_count >= SAMPLES_PER_LETTER:
            print(f"\n  {current_letter}: Already has {existing_count} samples. Press N to skip or R to redo.")
        else:
            print(f"\n  {current_letter}: Has {existing_count} existing samples. Recording more...")

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame = detector.find_hands(frame)
            h, w, _ = frame.shape

            # ---- Draw UI ----
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Current letter
            cv2.putText(frame, current_letter, (w // 2 - 40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 5)

            # Progress info
            total_for_letter = existing_count + collected
            progress_text = f"Letter: {current_letter}  |  Existing: {existing_count}  |  New: {collected}  |  Total: {total_for_letter}/{SAMPLES_PER_LETTER}"
            cv2.putText(frame, progress_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Overall progress
            overall = f"Overall: {letter_index + 1}/{len(LABELS)} letters  |  Total samples: {len(all_landmarks)}"
            cv2.putText(frame, overall, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Recording status
            if recording:
                if int(time.time() * 3) % 2 == 0:
                    cv2.circle(frame, (w - 40, 30), 12, (0, 0, 255), -1)
                cv2.putText(frame, "RECORDING", (w - 170, 37),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "PAUSED - Press SPACE to record", (w // 2 - 220, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            # Progress bar
            bar_progress = min(total_for_letter / SAMPLES_PER_LETTER, 1.0)
            bar_width = int(bar_progress * (w - 40))
            cv2.rectangle(frame, (20, 110), (w - 20, 125), (50, 50, 50), -1)
            if bar_width > 0:
                bar_color = (0, 255, 0) if total_for_letter >= SAMPLES_PER_LETTER else (0, 200, 255)
                cv2.rectangle(frame, (20, 110), (20 + bar_width, 125), bar_color, -1)

            # Controls
            cv2.putText(frame, "SPACE=Record | N=Next | R=Redo | Q=Quit",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # ---- Collect data ----
            if recording and total_for_letter < SAMPLES_PER_LETTER:
                landmarks = detector.get_landmarks(frame)
                if landmarks is not None:
                    all_landmarks.append(landmarks)
                    all_labels.append(current_letter)
                    collected += 1
                    total_for_letter = existing_count + collected

                    cv2.rectangle(frame, (0, 0), (10, h), (0, 255, 0), -1)
                    cv2.rectangle(frame, (w - 10, 0), (w, h), (0, 255, 0), -1)

            # Done with this letter
            if total_for_letter >= SAMPLES_PER_LETTER:
                cv2.putText(frame, "DONE! Press N for next letter",
                            (w // 2 - 250, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                recording = False

            cv2.imshow("SignVox - Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                recording = not recording

            elif key == ord('n'):
                # Auto-save when moving to next letter
                save_data(all_landmarks, all_labels)
                print(f"  Saved! {current_letter}: {total_for_letter} samples")
                letter_index += 1
                break

            elif key == ord('r'):
                # Remove ALL data for this letter
                new_landmarks = []
                new_labels = []
                for lm, lb in zip(all_landmarks, all_labels):
                    if lb != current_letter:
                        new_landmarks.append(lm)
                        new_labels.append(lb)
                all_landmarks = new_landmarks
                all_labels = new_labels
                existing_count = 0
                collected = 0
                recording = False
                existing_letters[current_letter] = 0
                print(f"  Reset {current_letter}. Starting over...")

            elif key == ord('q'):
                letter_index = len(LABELS)
                break

    # ---- Final Save ----
    cap.release()
    cv2.destroyAllWindows()

    if len(all_landmarks) > 0:
        save_data(all_landmarks, all_labels)

        print("\n" + "=" * 55)
        print("  DATA COLLECTION COMPLETE!")
        print("=" * 55)
        print(f"  Total samples: {len(all_landmarks)}")
        print(f"  Letters: {len(set(all_labels))}")

        print("\n  Summary:")
        for letter in LABELS:
            count = all_labels.count(letter)
            if count > 0:
                bar = "█" * (count // 10)
                print(f"    {letter}: {count:>4} samples {bar}")
        print("=" * 55)
    else:
        print("\n  No data collected.")


if __name__ == "__main__":
    main()