"""
SignVox — Real-Time Sign Language to Speech
Detects sign language and speaks it out loud!
Fixed: Backspace, clear, and clear voice settings.
"""

import cv2
import numpy as np
import pickle
import pyttsx3
import time
import os
import threading
from hand_detector import HandDetector

MODEL_DIR = "models"


def get_best_voice():
    """Find the clearest voice available on the system."""
    eng = pyttsx3.init()
    voices = eng.getProperty('voices')

    print("\n  Available voices:")
    for i, voice in enumerate(voices):
        print(f"    [{i}] {voice.name}")

    # Prefer Zira (female, very clear) or David (male, clear) on Windows
    best_voice = None
    for voice in voices:
        name = voice.name.lower() if voice.name else ""
        if "zira" in name:       # Microsoft Zira — very clear female voice
            best_voice = voice.id
            print(f"\n  Selected voice: {voice.name} (clearest)")
            break
        elif "david" in name:    # Microsoft David — clear male voice
            best_voice = voice.id
            print(f"\n  Selected voice: {voice.name}")

    if best_voice is None and voices:
        best_voice = voices[0].id
        print(f"\n  Selected voice: {voices[0].name} (default)")

    eng.stop()
    return best_voice


# Find best voice at startup
BEST_VOICE = get_best_voice()


def speak_text(text):
    """Speak text in a separate thread with a fresh engine each time."""
    def _speak():
        try:
            eng = pyttsx3.init()
            if BEST_VOICE:
                eng.setProperty('voice', BEST_VOICE)
            eng.setProperty('rate', 130)      # Slower = clearer
            eng.setProperty('volume', 1.0)    # Max volume
            eng.say(text)
            eng.runAndWait()
            eng.stop()
        except:
            pass
    threading.Thread(target=_speak, daemon=True).start()


def main():
    # Load trained model
    model_path = os.path.join(MODEL_DIR, "signvox_model.pkl")
    labels_path = os.path.join(MODEL_DIR, "labels.pkl")

    if not os.path.exists(model_path):
        print("ERROR: No trained model found! Run train_model.py first.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)

    print(f"  Model loaded! Recognizes: {labels}")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot access webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize hand detector
    detector = HandDetector(model_path="hand_landmarker.task", max_hands=1)

    # Fullscreen
    cv2.namedWindow("SignVox", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("SignVox", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("=" * 55)
    print("  SignVox - Live Sign Language to Speech")
    print("=" * 55)
    print("  Controls:")
    print("    SPACE     = Speak the full word")
    print("    BACKSPACE = Delete last letter")
    print("    C         = Clear entire word")
    print("    Q         = Quit")
    print("=" * 55)

    # State variables
    current_word = ""
    last_letter = ""
    stable_count = 0
    STABLE_THRESHOLD = 15
    last_spoken_letter = ""
    confidence_val = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = detector.find_hands(frame)
        h, w, _ = frame.shape

        # Get landmarks
        landmarks = detector.get_landmarks(frame)

        detected_letter = ""

        if landmarks is not None:
            # Predict the letter
            landmarks_array = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(landmarks_array)[0]
            probabilities = model.predict_proba(landmarks_array)[0]
            confidence_val = max(probabilities) * 100

            if confidence_val > 60:
                detected_letter = prediction

                # Stability check
                if detected_letter == last_letter:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_letter = detected_letter

                # If stable enough and different from last spoken
                if stable_count >= STABLE_THRESHOLD and detected_letter != last_spoken_letter:
                    current_word += detected_letter
                    last_spoken_letter = detected_letter
                    stable_count = 0
                    print(f"  Detected: {detected_letter}  |  Word: {current_word}")
                    speak_text(detected_letter)

            # Draw bounding box
            bbox = detector.get_bounding_box(frame)
            if bbox:
                x, y, bw, bh = bbox
                color = (0, 255, 0) if confidence_val > 60 else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

                if detected_letter:
                    cv2.putText(frame, detected_letter, (x + bw // 2 - 20, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)

        else:
            stable_count = 0
            last_letter = ""
            last_spoken_letter = ""

        # ---- Draw UI ----

        # Top dark bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Bottom dark bar
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 100), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0)

        # Title
        cv2.putText(frame, "SignVox", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # Confidence bar
        if detected_letter:
            cv2.putText(frame, f"Confidence: {confidence_val:.0f}%", (250, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            bar_w = int((confidence_val / 100) * 200)
            bar_color = (0, 255, 0) if confidence_val > 60 else (0, 165, 255)
            cv2.rectangle(frame, (500, 25), (500 + bar_w, 50), bar_color, -1)
            cv2.rectangle(frame, (500, 25), (700, 50), (100, 100, 100), 2)

        # Current word display
        display_word = current_word if current_word else "Show a sign..."
        cv2.putText(frame, f"Word: {display_word}", (20, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        # Controls
        cv2.putText(frame, "SPACE=Speak | BACKSPACE=Delete | C=Clear All | Q=Quit",
                    (w - 600, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("SignVox", frame)

        # ---- Key controls ----
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):          # SPACE = speak full word
            if current_word:
                print(f"\n  Speaking: {current_word}")
                speak_text(current_word)

        elif key == 8:               # BACKSPACE = delete last letter
            if current_word:
                removed = current_word[-1]
                current_word = current_word[:-1]
                last_spoken_letter = ""
                print(f"  Deleted '{removed}'. Word: {current_word}")

        elif key == ord('c'):        # C = clear entire word
            current_word = ""
            last_spoken_letter = ""
            print("  Word cleared.")

        elif key == ord('q'):        # Q = quit
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nSignVox closed. Goodbye!")


if __name__ == "__main__":
    main()