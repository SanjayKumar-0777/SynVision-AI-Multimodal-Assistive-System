import cv2
import threading
import os
import numpy as np
import pickle
import time
sign_buffer = []
buffer_lock = threading.Lock()

model = None
labels = None
model_loaded = False

def load_model():
    global model, labels, model_loaded

    if model_loaded:
        return

    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, 'model', 'model.p')
        labels_path = os.path.join(BASE_DIR, 'model', 'labels.txt')

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                model = model_data['model']

            with open(labels_path, 'r') as f:
                labels = [l.strip() for l in f.readlines()]

            print("Model loaded successfully")
        else:
            print("Model not found")

        model_loaded = True

    except Exception as e:
        print("MODEL LOAD ERROR:", e)
        model_loaded = True


def detect_hand():
    from utils.hand_utils import detect_and_process_hand
    return detect_and_process_hand


class VideoCamera:

    def __init__(self):
        cv2.destroyAllWindows()
        load_model()
        self.capture_lock = threading.Lock()
        self.pred_buffer = []
        self.last_sign = None
        self.hand_present = False
        self.detect = detect_hand()
        self.mode = "collect"
        self.prev_mode = None
        self.video = None

        for i in range(3):
            self.video = cv2.VideoCapture(0, cv2.CAP_MSMF)
            time.sleep(1)

            if not self.video.isOpened():
                print("MSMF failed → trying DSHOW")
                self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                time.sleep(1)

            if not self.video.isOpened():
                print("DSHOW failed → trying default")
                self.video = cv2.VideoCapture(0)
                time.sleep(1)

            if not self.video.isOpened():
                print("Camera not opened")
            else:
                print("Camera opened successfully")

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_frame_bytes(self):

        if self.prev_mode != self.mode:
            print("MODE CHANGED →", self.mode)
            self.prev_mode = self.mode
        with self.capture_lock:
            if self.video is None or not self.video.isOpened():
                return None
            ret, frame = self.video.read()

        if not ret:
            print("Frame failed → retrying")
            time.sleep(0.05)
            return None

        frame = cv2.flip(frame, 1)

        result = self.detect(frame)

        annotated = frame.copy()
        landmarks = None

        if result is not None:
            landmarks, annotated = result

        # ⭐ NO HAND → RESET STATE
        if landmarks is None:
            self.pred_buffer.clear()
            self.last_sign = None

        # ⭐ PREDICTION BLOCK
        
        if self.mode == "recognize" and landmarks is not None and model is not None:

            try:
                landmarks = np.array(landmarks, dtype=np.float32)

                # ⭐ FIX 1 → remove NaN / Inf
                if not np.isfinite(landmarks).all():
                    raise ValueError("Invalid landmark values")

                # ⭐ FIX 2 → reshape
                landmarks = landmarks.reshape(1, -1)

                # ⭐ FIX 3 → feature mismatch safety
                if landmarks.shape[1] != model.n_features_in_:
                    raise ValueError(f"Feature mismatch: model expects {model.n_features_in_}, but got {landmarks.shape[1]}")

                probs = model.predict_proba(landmarks)[0]
                confidence = np.max(probs)
                pred = np.argmax(probs)

                if confidence < 0.65:
                    label = "Unknown"
                else:
                    label = labels[pred]

                # ⭐ TEMPORAL SMOOTHING
                if label != "Unknown":
                    self.pred_buffer.append(label)

                if len(self.pred_buffer) > 10:
                    self.pred_buffer.pop(0)

                # ⭐ STABLE SIGN
                if self.pred_buffer.count(label) >= 7:
                    if self.last_sign != label:
                        self.last_sign = label
                        with buffer_lock:
                            sign_buffer.append(label)

            except Exception as e:
                print("Prediction skipped:", e)
        # ⭐ SHOW TEXT ONLY IN RECOGNIZE MODE
        if self.mode == "recognize":
            display_text = self.last_sign if self.last_sign else "Waiting..."
            cv2.putText(
                annotated,
                f"Sign: {display_text}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        ret, jpeg = cv2.imencode('.jpg', annotated)
        return jpeg.tobytes()
    
    def release(self):
        if self.video is not None:
            if self.video.isOpened():
                self.video.release()
                print("Camera fully released")

        self.video = None

    def save_current_frame(self, label):
        print("TRY SAVE CALLED")
        with self.capture_lock:
            success, frame = self.video.read()
        if not success:
            return False

        # Detect features
        landmarks, _ = self.detect(frame)

        # IMPORTANT: allow face OR hand OR pose
        if landmarks is None:
            return False

        save_dir = os.path.join(os.getcwd(), 'dataset', label)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filename = f"{int(time.time()*1000)}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), frame)

        print(f"Saved sample for {label}")
        return True