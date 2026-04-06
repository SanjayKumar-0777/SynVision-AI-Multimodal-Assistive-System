import os
import time
import json
import google.generativeai
genai = google.generativeai
from PIL import Image
import cv2
import threading

class GeminiDetector:
    def __init__(self, api_key=None):
        if not api_key:
            api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key or "YOUR_API_KEY" in api_key:
            print("WARNING: Gemini API Key not set/invalid.")
            self.model = None
            return

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.lock = threading.Lock()
        self.last_result = None
        self.is_running = False

    def detect_thread(self, frame, target_name):
        """
        Runs detection in a separate thread. behavior:
        - If a thread is already running, returns immediately (skips frame).
        - If not, starts a new thread.
        - Returns the *last known* result (or None).
        """
        if self.is_running:
            return self.last_result

        # Start new thread
        t = threading.Thread(target=self._run_detection, args=(frame.copy(), target_name))
        t.start()
        return self.last_result

    def _run_detection(self, frame, target_name):
        self.is_running = True
        try:
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            prompt = (
                f"Locate the '{target_name}' in this image. "
                "Return a JSON object with a single key 'box_2d' containing [ymin, xmin, ymax, xmax] "
                "where values are normalized 0 to 1000. "
                "If the object is not clearly visible, return null or empty object."
                "Example: {\"box_2d\": [100, 200, 500, 600]}"
            )

            response = self.model.generate_content([prompt, pil_img], generation_config={"response_mime_type": "application/json"})
            text = response.text.strip()
            
            # Parse JSON
            # Clean potential markdown code blocks
            if text.startswith("```json"):
                text = text[7:-3]
            
            data = json.loads(text)
            
            box = data.get("box_2d")
            if box and len(box) == 4:
                # Store result
                with self.lock:
                    self.last_result = box # [ymin, xmin, ymax, xmax] 0-1000
        except Exception as e:
            print(f"Gemini API Error: {e}")
        finally:
            self.is_running = False

    def get_parsed_box(self, frame_shape):
        """
        Returns (x, y, w, h) based on last result and frame shape
        """
        with self.lock:
            if not self.last_result:
                return None
            
            ymin, xmin, ymax, xmax = self.last_result
            h_img, w_img = frame_shape[:2]
            
            # Convert 0-1000 to pixels
            x = int((xmin / 1000) * w_img)
            y = int((ymin / 1000) * h_img)
            w = int(((xmax - xmin) / 1000) * w_img)
            h = int(((ymax - ymin) / 1000) * h_img)
            
            return (x, y, w, h)
