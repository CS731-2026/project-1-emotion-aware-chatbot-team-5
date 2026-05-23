import sys

import cv2
from PyQt5.QtWidgets import QApplication

import gui as base_gui
from chatbot_free import EmotionAwareChatbot as FreeEmotionAwareChatbot
from inference_new import Inferencer


MODEL_PATH = "checkpoints/8.pt"

EMOTION_INDEX_TO_KEY = [
    "anger",
    "happy",
    "surprise",
    "sad",
    "contempt",
    "fear",
    "disgust",
    "neutral",
]

EMOTION_INDEX_TO_DISPLAY = [
    "Angry",
    "Happy",
    "Surprise",
    "Sad",
    "Contempt",
    "Fear",
    "Disgust",
    "Neutral",
]

EMOTION_INDEX_TO_EMOJI = [
    "😠",
    "😊",
    "😲",
    "😢",
    "😒",
    "😨",
    "🤢",
    "😐",
]


base_gui.MODEL_PATH = MODEL_PATH
base_gui.Inferencer = Inferencer
base_gui.EmotionAwareChatbot = FreeEmotionAwareChatbot


class EmotionChatbotGUINew(base_gui.EmotionChatbotGUI):
    def _detect_emotion(self, frame):
        """Detect faces with YOLOv8 and classify each face with the new inference helper."""
        try:
            results = self.face_model(frame, verbose=False, conf=0.4)
            best_emotion = None
            best_score = None
            best_box = None

            frame_h, frame_w = frame.shape[:2]
            frame_cx = frame_w / 2.0
            frame_cy = frame_h / 2.0

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])

                    pad = 15
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(frame.shape[1], x2 + pad)
                    y2 = min(frame.shape[0], y2 + pad)

                    box_w = max(1, x2 - x1)
                    box_h = max(1, y2 - y1)
                    area = box_w * box_h
                    box_cx = x1 + box_w / 2.0
                    box_cy = y1 + box_h / 2.0
                    distance = ((box_cx - frame_cx) ** 2 + (box_cy - frame_cy) ** 2) ** 0.5
                    center_penalty = distance / ((frame_w ** 2 + frame_h ** 2) ** 0.5)
                    score = area * (1.0 - center_penalty)

                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    predicted_class, confidence = self.inferencer.predict(face_crop)
                    if predicted_class < 0 or predicted_class >= len(EMOTION_INDEX_TO_KEY):
                        continue

                    emotion_key = EMOTION_INDEX_TO_KEY[predicted_class]
                    emotion_display = EMOTION_INDEX_TO_DISPLAY[predicted_class]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{emotion_display}: {confidence:.0%}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                    )

                    if best_score is None or score > best_score:
                        best_score = score
                        best_emotion = (emotion_key, emotion_display, confidence, EMOTION_INDEX_TO_EMOJI[predicted_class])
                        best_box = (x1, y1, x2, y2)

            if best_emotion:
                emotion_key, emotion_display, confidence, emotion_emoji = best_emotion
                print(f"Detected emotion: {emotion_display} ({confidence:.1%})")

                self.emotion_history.append(emotion_key)
                if len(self.emotion_history) > base_gui.SMOOTH_WINDOW:
                    self.emotion_history.pop(0)
                smoothed = max(set(self.emotion_history), key=self.emotion_history.count)

                color = base_gui.EMOTION_COLORS.get(smoothed, '#ECF0F1')
                emoji_map = {
                    'happy': '😊', 'sad': '😢', 'angry': '😠',
                    'fear': '😨', 'surprise': '😲', 'disgust': '🤢',
                    'contempt': '😒', 'neutral': '😐'
                }
                self.emotion_label.setText(f"{emoji_map.get(smoothed, '😐')}  {smoothed.capitalize()}")
                self.emotion_label.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: bold;")
                self.confidence_label.setText(f"Confidence: {confidence:.1%}")

                if smoothed != self.current_emotion:
                    self.current_emotion = smoothed
                    self.chatbot.set_emotion(smoothed)
                self._auto_emotion_message(smoothed)

                if best_box:
                    x1, y1, x2, y2 = best_box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{emotion_display}: {confidence:.0%}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                    )

        except Exception as e:
            print(f"Emotion detection error: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName("CS731 Emotion Chatbot - New Inference")
    window = EmotionChatbotGUINew()
    window.show()
    sys.exit(app.exec_())
        self.status_label.setText("🎤  Listening... Speak now!")
        self._set_input_enabled(False)

        self.voice_thread = QThread()
        self.voice_worker = VoiceWorker()
        self.voice_worker.moveToThread(self.voice_thread)
        self.voice_thread.started.connect(self.voice_worker.run)
        self.voice_worker.text_ready.connect(self._on_voice_text_ready)
        self.voice_worker.error.connect(self._on_voice_error)
        self.voice_worker.text_ready.connect(self.voice_thread.quit)
        self.voice_worker.error.connect(self.voice_thread.quit)
        self.voice_thread.start()

    def _on_voice_text_ready(self, text):
        self.is_listening = False
        self.voice_btn.setText("🎤")
        self.voice_btn.setStyleSheet("")
        self._set_input_enabled(False)
        self._add_chat_message("🎤 You (voice)", text)
        self.status_label.setText("🤔  AI is thinking...")

        self.chat_thread = QThread()
        self.chat_worker = ChatWorker(self.chatbot, text)
        self.chat_worker.moveToThread(self.chat_thread)
        self.chat_thread.started.connect(self.chat_worker.run)
        self.chat_worker.response_ready.connect(self._on_response_ready)
        self.chat_worker.response_ready.connect(self.chat_thread.quit)
        self.chat_thread.start()

    def _on_voice_error(self, error_msg):
        self.is_listening = False
        self.voice_btn.setText("🎤")
        self.voice_btn.setStyleSheet("")
        self.status_label.setText(f"⚠️  {error_msg}")
        self._set_input_enabled(True)

    # -----------------------------------------------------------
    # CHAT HELPERS
    # -----------------------------------------------------------
    def _add_chat_message(self, sender, message):
        if sender.startswith("You"):
            color = "#64B5F6"
        elif sender.startswith("AI"):
            color = "#81C784"
        else:
            color = "#FFD54F"

        html = (
            f'<p style="margin: 6px 0;">'
            f'<span style="color: {color}; font-weight: bold;">{sender}:</span> '
            f'<span style="color: #ECEFF1;">{message}</span>'
            f'</p>'
        )
        self.chat_display.append(html)
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _clear_chat(self):
        self.chat_display.clear()
        self.chatbot.clear_history()
        self._add_chat_message("System", "Chat cleared. Starting fresh!")

    def _set_input_enabled(self, enabled):
        self.text_input.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)
        self.voice_btn.setEnabled(enabled)

    # -----------------------------------------------------------
    # STYLESHEET
    # -----------------------------------------------------------
    def _get_stylesheet(self):
        return """
            QMainWindow, QWidget {
                background-color: #1A1A2E;
                color: #ECEFF1;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QFrame#leftPanel, QFrame#rightPanel {
                background-color: #16213E;
                border-radius: 12px;
                padding: 8px;
            }
            QLabel#panelTitle {
                font-size: 15px;
                font-weight: bold;
                color: #90CAF9;
                padding: 6px;
                border-bottom: 1px solid #0F3460;
                margin-bottom: 4px;
            }
            QLabel#webcamLabel {
                background-color: #0D0D1A;
                border-radius: 8px;
                border: 2px solid #0F3460;
                color: #546E7A;
                font-size: 14px;
            }
            QFrame#emotionFrame {
                background-color: #0F3460;
                border-radius: 10px;
                padding: 10px;
                margin-top: 4px;
            }
            QLabel#emotionHeader {
                color: #90CAF9;
                font-size: 12px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            QLabel#emotionLabel {
                font-size: 28px;
                font-weight: bold;
                color: #ECF0F1;
                padding: 8px;
            }
            QLabel#confidenceLabel {
                color: #B0BEC5;
                font-size: 12px;
            }
            QTextEdit#chatDisplay {
                background-color: #0D0D1A;
                border: 1px solid #0F3460;
                border-radius: 8px;
                padding: 10px;
                color: #ECEFF1;
                font-size: 13px;
            }
            QLabel#statusLabel {
                color: #FFD54F;
                font-size: 12px;
                font-style: italic;
                min-height: 20px;
            }
            QLineEdit#textInput {
                background-color: #0F3460;
                border: 1px solid #1565C0;
                border-radius: 8px;
                padding: 10px 14px;
                color: #ECEFF1;
                font-size: 13px;
                min-height: 28px;
            }
            QLineEdit#textInput:focus { border: 1px solid #42A5F5; }
            QPushButton#sendBtn {
                background-color: #1565C0;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 13px;
                min-width: 90px;
            }
            QPushButton#sendBtn:hover    { background-color: #1976D2; }
            QPushButton#sendBtn:pressed  { background-color: #0D47A1; }
            QPushButton#sendBtn:disabled { background-color: #37474F; color: #607D8B; }
            QPushButton#voiceBtn {
                background-color: #0F3460;
                color: white;
                border: 1px solid #1565C0;
                border-radius: 8px;
                font-size: 20px;
            }
            QPushButton#voiceBtn:hover    { background-color: #1565C0; }
            QPushButton#voiceBtn:disabled { background-color: #37474F; color: #607D8B; }
            QPushButton#clearBtn {
                background-color: transparent;
                color: #546E7A;
                border: 1px solid #37474F;
                border-radius: 6px;
                padding: 6px;
                font-size: 12px;
            }
            QPushButton#clearBtn:hover { background-color: #263238; color: #90A4AE; }
            QScrollBar:vertical {
                background: #0D0D1A;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #1565C0;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical { height: 0px; }
        """

    # -----------------------------------------------------------
    # CLEANUP
    # -----------------------------------------------------------
    def closeEvent(self, event):
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()


# -----------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName("CS731 Emotion Chatbot")
    window = EmotionChatbotGUI()
    window.show()
    sys.exit(app.exec_())
