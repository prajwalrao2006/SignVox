"""
SignVox — Hand Detection Module (New MediaPipe Tasks API)
Detects hands in real-time and extracts 21 landmarks.
Full screen + Both hands support.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time


# Hand landmark connections for drawing the skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm
]

# Colors for each hand
HAND_COLORS = [
    (0, 255, 0),      # Green for hand 1
    (255, 165, 0)     # Orange for hand 2
]


class HandDetector:
    """Detects hands and extracts landmark coordinates using MediaPipe Tasks API."""

    def __init__(self, model_path="hand_landmarker.task", max_hands=2,
                 detection_confidence=0.7, tracking_confidence=0.7):

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            running_mode=vision.RunningMode.VIDEO
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.latest_result = None

    def find_hands(self, frame, draw=True):
        """Detect hands in a frame and optionally draw landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        self.latest_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if self.latest_result.hand_landmarks and draw:
            for i, hand_landmarks in enumerate(self.latest_result.hand_landmarks):
                color = HAND_COLORS[i % len(HAND_COLORS)]
                self._draw_landmarks(frame, hand_landmarks, color)

        return frame

    def _draw_landmarks(self, frame, hand_landmarks, color=(0, 255, 0)):
        """Draw hand landmarks and connections on the frame."""
        height, width, _ = frame.shape
        points = []

        for lm in hand_landmarks:
            px = int(lm.x * width)
            py = int(lm.y * height)
            points.append((px, py))

        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx], color, 2)

        # Draw landmark points
        for i, point in enumerate(points):
            if i in [4, 8, 12, 16, 20]:  # Fingertips
                cv2.circle(frame, point, 7, (0, 200, 255), -1)
                cv2.circle(frame, point, 7, (255, 255, 255), 1)
            else:
                cv2.circle(frame, point, 4, color, -1)
                cv2.circle(frame, point, 4, (255, 255, 255), 1)

    def get_hand_count(self):
        """Return the number of hands currently detected."""
        if self.latest_result and self.latest_result.hand_landmarks:
            return len(self.latest_result.hand_landmarks)
        return 0

    def get_landmarks(self, frame, hand_index=0):
        """
        Extract hand landmark coordinates as a flat list.
        Returns a list of 42 values (21 landmarks x 2 coordinates: x, y)
        Returns
                None if no hand is detected.
        """
        if (self.latest_result and self.latest_result.hand_landmarks
                and hand_index < len(self.latest_result.hand_landmarks)):
            hand = self.latest_result.hand_landmarks[hand_index]
            landmarks = []

            x_coords = [lm.x for lm in hand]
            y_coords = [lm.y for lm in hand]

            min_x, min_y = min(x_coords), min(y_coords)

            for lm in hand:
                landmarks.append(lm.x - min_x)
                landmarks.append(lm.y - min_y)

            return landmarks

        return None

    def get_bounding_box(self, frame, hand_index=0):
        """
        Get the bounding box around a specific detected hand.
        Returns (x, y, w, h) or None if hand not detected.
        """
        if (self.latest_result and self.latest_result.hand_landmarks
                and hand_index < len(self.latest_result.hand_landmarks)):
            hand = self.latest_result.hand_landmarks[hand_index]
            height, width, _ = frame.shape

            x_coords = [lm.x for lm in hand]
            y_coords = [lm.y for lm in hand]

            padding = 20
            x_min = max(0, int(min(x_coords) * width) - padding)
            y_min = max(0, int(min(y_coords) * height) - padding)
            x_max = min(width, int(max(x_coords) * width) + padding)
            y_max = min(height, int(max(y_coords) * height) + padding)

            return (x_min, y_min, x_max - x_min, y_max - y_min)

        return None


def main():
    """Test the hand detector with live webcam feed."""

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot access webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize hand detector (2 hands)
    detector = HandDetector(model_path="hand_landmarker.task", max_hands=2)

    # Set window to FULLSCREEN
    cv2.namedWindow("SignVox - Hand Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("SignVox - Hand Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("=" * 50)
    print("  SignVox - Hand Detection Test")
    print("=" * 50)
    print("  Show your hands to the camera!")
    print("  Press 'Q' to quit")
    print("=" * 50)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = detector.find_hands(frame)

        num_hands = detector.get_hand_count()

        if num_hands > 0:
            for i in range(num_hands):
                bbox = detector.get_bounding_box(frame, hand_index=i)
                if bbox:
                    x, y, w, h = bbox
                    color = HAND_COLORS[i % len(HAND_COLORS)]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"Hand {i+1}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, f"HANDS DETECTED: {num_hands}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected - show your hands!", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Title at bottom
        h = frame.shape[0]
        cv2.putText(frame, "SignVox | Hand Detection | Press 'Q' to quit", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("SignVox - Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nHand detection test complete!")


if __name__ == "__main__":
    main()