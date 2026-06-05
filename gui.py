"""
SignVox — Beautiful GUI Version (COMPLETE)
Dark mode, live camera, speech, space for sentences, backspace, clear.
"""

import cv2
import numpy as np
import pickle
import pyttsx3
import time
import os
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import threading
from hand_detector import HandDetector


# ============== COLORS ==============
BG_DARK = "#0D1117"
BG_CARD = "#161B22"
ACCENT = "#00FFC8"
ACCENT_2 = "#7C3AED"
TEXT_PRIMARY = "#E6EDF3"
TEXT_SECONDARY = "#8B949E"
BORDER = "#30363D"
RED = "#FF4757"
GREEN = "#00FF88"
YELLOW = "#FFD93D"
ORANGE = "#FF9F43"


def get_best_voice():
    """Find the clearest voice available."""
    eng = pyttsx3.init()
    voices = eng.getProperty('voices')
    best_voice = None
    for voice in voices:
        name = voice.name.lower() if voice.name else ""
        if "zira" in name:
            best_voice = voice.id
            break
        elif "david" in name:
            best_voice = voice.id
    if best_voice is None and voices:
        best_voice = voices[0].id
    eng.stop()
    return best_voice


BEST_VOICE = get_best_voice()


def speak_text(text):
    """Speak text in a separate thread with a fresh engine."""
    def _speak():
        try:
            eng = pyttsx3.init()
            if BEST_VOICE:
                eng.setProperty('voice', BEST_VOICE)
            eng.setProperty('rate', 130)
            eng.setProperty('volume', 1.0)
            eng.say(text)
            eng.runAndWait()
            eng.stop()
        except:
            pass
    threading.Thread(target=_speak, daemon=True).start()


class SignVoxGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SignVox — AI Sign Language Translator")
        self.root.configure(bg=BG_DARK)
        self.root.attributes('-fullscreen', True)

        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()

        # Fixed sidebar width
        self.SIDEBAR_W = 350

        # ---- Fonts ----
        self.font_title = tkfont.Font(family="Segoe UI", size=22, weight="bold")
        self.font_subtitle = tkfont.Font(family="Segoe UI", size=10)
        self.font_large = tkfont.Font(family="Segoe UI", size=48, weight="bold")
        self.font_medium = tkfont.Font(family="Segoe UI", size=16)
        self.font_small = tkfont.Font(family="Segoe UI", size=11)
        self.font_word = tkfont.Font(family="Consolas", size=28, weight="bold")
        self.font_letter = tkfont.Font(family="Segoe UI", size=64, weight="bold")
        self.font_btn = tkfont.Font(family="Segoe UI", size=13, weight="bold")

        # ---- State ----
        self.current_word = ""
        self.last_letter = ""
        self.stable_count = 0
        self.STABLE_THRESHOLD = 15
        self.last_spoken_letter = ""
        self.detected_letter = ""
        self.confidence = 0.0
        self.is_running = True
        self.history = []
        self.hand_detected = False

        # ---- Load Model ----
        self.model = None
        self.labels = None
        self.model_loaded = False
        self.load_model()

        # ---- Camera & Detector ----
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.detector = HandDetector(model_path="hand_landmarker.task", max_hands=1)

        # ---- Build GUI ----
        self.build_gui()

        # ---- Key Bindings ----
        self.root.bind('<space>', lambda e: self.speak_word())
        self.root.bind('<s>', lambda e: self.add_space())
        self.root.bind('<c>', lambda e: self.clear_word())
        self.root.bind('<BackSpace>', lambda e: self.delete_last())
        self.root.bind('<Escape>', lambda e: self.quit_app())
        self.root.bind('<q>', lambda e: self.quit_app())

        # ---- Start Camera ----
        self.update_frame()

    def load_model(self):
        """Load the trained model."""
        model_path = os.path.join("models", "signvox_model.pkl")
        labels_path = os.path.join("models", "labels.pkl")
        if os.path.exists(model_path) and os.path.exists(labels_path):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(labels_path, "rb") as f:
                self.labels = pickle.load(f)
            self.model_loaded = True
        else:
            self.model_loaded = False

    def build_gui(self):
        """Build the dark mode GUI with fixed layout."""

        # ============ TOP BAR ============
        top_bar = tk.Frame(self.root, bg=BG_CARD, height=55)
        top_bar.pack(fill=tk.X)
        top_bar.pack_propagate(False)

        # Left - Logo
        left_top = tk.Frame(top_bar, bg=BG_CARD)
        left_top.pack(side=tk.LEFT, padx=15)

        tk.Label(left_top, text="🤟 SignVox", font=self.font_title,
                 fg=ACCENT, bg=BG_CARD).pack(side=tk.LEFT)
        tk.Label(left_top, text="  AI Sign Language Translator",
                 font=self.font_subtitle, fg=TEXT_SECONDARY, bg=BG_CARD).pack(side=tk.LEFT, pady=(5, 0))

        # Right - Status
        right_top = tk.Frame(top_bar, bg=BG_CARD)
        right_top.pack(side=tk.RIGHT, padx=15)

        self.status_label = tk.Label(right_top, text="● LIVE", font=self.font_small,
                                     fg=GREEN, bg=BG_CARD)
        self.status_label.pack(side=tk.RIGHT, padx=(10, 0))

        model_text = "✓ Model Loaded" if self.model_loaded else "✗ No Model"
        model_color = GREEN if self.model_loaded else RED
        tk.Label(right_top, text=model_text, font=self.font_small,
                 fg=model_color, bg=BG_CARD).pack(side=tk.RIGHT)

        # Separator
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X)

        # ============ MAIN AREA ============
        main_frame = tk.Frame(self.root, bg=BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ---- LEFT: Camera ----
        left_frame = tk.Frame(main_frame, bg=BG_DARK)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 4), pady=8)

        cam_container = tk.Frame(left_frame, bg=BORDER, highlightbackground=BORDER,
                                 highlightthickness=1)
        cam_container.pack(fill=tk.BOTH, expand=True)

        cam_header = tk.Frame(cam_container, bg=BG_CARD, height=35)
        cam_header.pack(fill=tk.X)
        cam_header.pack_propagate(False)
        tk.Label(cam_header, text="  📷 Live Camera", font=self.font_small,
                 fg=TEXT_PRIMARY, bg=BG_CARD).pack(side=tk.LEFT, pady=5)

        self.fps_label = tk.Label(cam_header, text="", font=self.font_small,
                                  fg=TEXT_SECONDARY, bg=BG_CARD)
        self.fps_label.pack(side=tk.RIGHT, padx=10)

        self.camera_label = tk.Label(cam_container, bg="#000000")
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        # Separator
        tk.Frame(main_frame, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)

        # ---- RIGHT: Sidebar (FIXED width) ----
        right_frame = tk.Frame(main_frame, bg=BG_DARK, width=self.SIDEBAR_W)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 8), pady=8)
        right_frame.pack_propagate(False)

        # ---- Detected Letter ----
        self.letter_card = tk.Frame(right_frame, bg=BG_CARD, highlightbackground=BORDER,
                                    highlightthickness=1)
        self.letter_card.pack(fill=tk.X, pady=(0, 4))

        tk.Label(self.letter_card, text="DETECTED SIGN", font=self.font_small,
                 fg=TEXT_SECONDARY, bg=BG_CARD).pack(pady=(12, 0))

        self.letter_display = tk.Label(self.letter_card, text="—", font=self.font_letter,
                                       fg=ACCENT, bg=BG_CARD)
        self.letter_display.pack(pady=(0, 5))

        conf_frame = tk.Frame(self.letter_card, bg=BG_CARD)
        conf_frame.pack(fill=tk.X, padx=15, pady=(0, 5))
        tk.Label(conf_frame, text="Confidence", font=self.font_small,
                 fg=TEXT_SECONDARY, bg=BG_CARD).pack(side=tk.LEFT)
        self.conf_label = tk.Label(conf_frame, text="0%", font=self.font_small,
                                   fg=TEXT_PRIMARY, bg=BG_CARD)
        self.conf_label.pack(side=tk.RIGHT)

        self.conf_canvas = tk.Canvas(self.letter_card, height=6, bg=BG_DARK,
                                     highlightthickness=0)
        self.conf_canvas.pack(fill=tk.X, padx=15, pady=(0, 12))

        # ---- Current Word / Sentence ----
        word_card = tk.Frame(right_frame, bg=BG_CARD, highlightbackground=BORDER,
                             highlightthickness=1)
        word_card.pack(fill=tk.X, pady=4)

        tk.Label(word_card, text="CURRENT TEXT", font=self.font_small,
                 fg=TEXT_SECONDARY, bg=BG_CARD).pack(pady=(12, 0))

        self.word_display = tk.Label(word_card, text="...", font=self.font_word,
                                     fg=TEXT_PRIMARY, bg=BG_CARD, wraplength=310,
                                     justify=tk.LEFT)
        self.word_display.pack(pady=(5, 12), padx=10)

        # ---- Buttons ----
        btn_frame = tk.Frame(right_frame, bg=BG_DARK)
        btn_frame.pack(fill=tk.X, pady=4)

        # Speak button (big, purple)
        tk.Button(btn_frame, text="🔊 SPEAK", font=self.font_btn,
                  bg=ACCENT_2, fg="white", relief=tk.FLAT,
                  activebackground="#6D28D9", cursor="hand2",
                  command=self.speak_word, pady=10).pack(fill=tk.X, pady=(0, 3))

        # Button row: Space, Delete, Clear
        btn_row = tk.Frame(btn_frame, bg=BG_DARK)
        btn_row.pack(fill=tk.X)

        tk.Button(btn_row, text="␣ Space", font=self.font_small,
                  bg="#1A5C3A", fg=TEXT_PRIMARY, relief=tk.FLAT,
                  activebackground="#22764A", cursor="hand2",
                  command=self.add_space, pady=6).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        tk.Button(btn_row, text="⌫ Delete", font=self.font_small,
                  bg="#21262D", fg=TEXT_PRIMARY, relief=tk.FLAT,
                  activebackground="#30363D", cursor="hand2",
                  command=self.delete_last, pady=6).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        tk.Button(btn_row, text="🗑 Clear", font=self.font_small,
                  bg="#21262D", fg=TEXT_PRIMARY, relief=tk.FLAT,
                  activebackground="#30363D", cursor="hand2",
                  command=self.clear_word, pady=6).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # ---- History ----
        history_card = tk.Frame(right_frame, bg=BG_CARD, highlightbackground=BORDER,
                                highlightthickness=1)
        history_card.pack(fill=tk.BOTH, expand=True, pady=4)

        tk.Label(history_card, text="📜 HISTORY", font=self.font_small,
                 fg=TEXT_SECONDARY, bg=BG_CARD).pack(pady=(10, 5), anchor=tk.W, padx=12)

        self.history_text = tk.Text(history_card, font=self.font_small, bg=BG_DARK,
                                    fg=TEXT_SECONDARY, relief=tk.FLAT, height=4,
                                    wrap=tk.WORD, padx=8, pady=5)
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.history_text.config(state=tk.DISABLED)

        # ---- Controls Help ----
        controls_card = tk.Frame(right_frame, bg=BG_CARD, highlightbackground=BORDER,
                                 highlightthickness=1)
        controls_card.pack(fill=tk.X, pady=(4, 0))

        tk.Label(controls_card, text="SPACE=Speak | S=Space | ⌫=Delete | C=Clear | ESC=Quit",
                 font=tkfont.Font(family="Segoe UI", size=9),
                 fg=TEXT_SECONDARY, bg=BG_CARD).pack(pady=6)

    def update_frame(self):
        """Main loop — captures frame, detects, updates GUI."""
        if not self.is_running:
            return

        start_time = time.time()

        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            frame = self.detector.find_hands(frame)

            self.detected_letter = ""
            self.hand_detected = False

            if self.model_loaded:
                landmarks = self.detector.get_landmarks(frame)

                if landmarks is not None:
                    self.hand_detected = True
                    landmarks_array = np.array(landmarks).reshape(1, -1)
                    prediction = self.model.predict(landmarks_array)[0]
                    probabilities = self.model.predict_proba(landmarks_array)[0]
                    self.confidence = max(probabilities) * 100

                    if self.confidence > 60:
                        self.detected_letter = prediction

                        if self.detected_letter == self.last_letter:
                            self.stable_count += 1
                        else:
                            self.stable_count = 0
                            self.last_letter = self.detected_letter

                        if (self.stable_count >= self.STABLE_THRESHOLD
                                and self.detected_letter != self.last_spoken_letter):
                            self.current_word += self.detected_letter
                            self.last_spoken_letter = self.detected_letter
                            self.stable_count = 0
                            speak_text(self.detected_letter)

                    # Draw bounding box
                    bbox = self.detector.get_bounding_box(frame)
                    if bbox:
                        x, y, bw, bh = bbox
                        color = (0, 255, 0) if self.confidence > 60 else (0, 165, 255)
                        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                        if self.detected_letter:
                            cv2.putText(frame, self.detected_letter,
                                        (x + bw // 2 - 20, y - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                else:
                    self.stable_count = 0
                    self.last_letter = ""
                    self.last_spoken_letter = ""
                    self.confidence = 0

            # Convert frame for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            label_w = self.camera_label.winfo_width()
            label_h = self.camera_label.winfo_height()
            if label_w > 10 and label_h > 10:
                frame_h, frame_w = frame.shape[:2]
                scale = min(label_w / frame_w, label_h / frame_h)
                new_w = int(frame_w * scale)
                new_h = int(frame_h * scale)
                img = img.resize((new_w, new_h), Image.NEAREST)

            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)

            self.update_ui()

            elapsed = time.time() - start_time
            fps = int(1 / elapsed) if elapsed > 0 else 0
            self.fps_label.config(text=f"{fps} FPS")

        self.root.after(40, self.update_frame)

    def update_ui(self):
        """Update sidebar elements."""
        if self.detected_letter:
            self.letter_display.config(text=self.detected_letter, fg=ACCENT)
            self.letter_card.config(highlightbackground=ACCENT)
        elif self.hand_detected:
            self.letter_display.config(text="?", fg=YELLOW)
            self.letter_card.config(highlightbackground=YELLOW)
        else:
            self.letter_display.config(text="—", fg=TEXT_SECONDARY)
            self.letter_card.config(highlightbackground=BORDER)

        self.conf_label.config(text=f"{self.confidence:.0f}%")
        self.conf_canvas.delete("all")
        canvas_w = self.conf_canvas.winfo_width()
        if canvas_w > 1:
            bar_w = int((self.confidence / 100) * canvas_w)
            color = GREEN if self.confidence > 60 else ORANGE
            self.conf_canvas.create_rectangle(0, 0, bar_w, 6, fill=color, outline="")

        if self.current_word:
            self.word_display.config(text=self.current_word, fg=TEXT_PRIMARY)
        else:
            self.word_display.config(text="...", fg=TEXT_SECONDARY)

        if self.hand_detected:
            self.status_label.config(text="● HAND DETECTED", fg=GREEN)
        else:
            self.status_label.config(text="● WAITING", fg=YELLOW)

    def speak_word(self):
        """Speak the current text."""
        if self.current_word:
            word = self.current_word.strip()
            self.history.append(word)
            self.history_text.config(state=tk.NORMAL)
            self.history_text.insert(tk.END, f"🔊 {word}\n")
            self.history_text.see(tk.END)
            self.history_text.config(state=tk.DISABLED)
            speak_text(word)

    def add_space(self):
        """Add a space for building sentences."""
        if self.current_word and not self.current_word.endswith(" "):
            self.current_word += " "
            self.last_spoken_letter = ""

    def clear_word(self):
        """Clear entire text."""
        self.current_word = ""
        self.last_spoken_letter = ""

    def delete_last(self):
        """Delete last character."""
        if self.current_word:
            self.current_word = self.current_word[:-1]
            self.last_spoken_letter = ""

    def quit_app(self):
        """Clean up and quit."""
        self.is_running = False
        self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = SignVoxGUI()
    app.run()