
import cv2
import numpy as np
from collections import deque
import time
import os
import threading
import pyttsx3
import winsound
import pythoncom
from ultralytics import YOLO
from gemini_loader import GeminiDetector

# Constants
# Widths in CM (Approximations)
KNOWN_WIDTHS = {
    'face': 14.5,
    'person': 45.0, # Shoulder width approx
    'cell phone': 7.5,
    'laptop': 35.0,
    'bottle': 7.0,
    'cup': 8.0,
    'chair': 50.0,
    'tv': 100.0,
    'book': 15.0,
    'keyboard': 45.0,
    'mouse': 6.0,
    'remote': 5.0,
}

DEFAULT_FOCAL_LENGTH = 600 # Initial guess, user should calibrate
BUFFER_SIZE = 5

class ObjectFinder:
    def __init__(self, face_cascade_path):
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        # Load YOLO model - will download 'yolov8n.pt' on first run if not present
        self.yolo = YOLO('yolov8n.pt') 
        
        self.focal_length = DEFAULT_FOCAL_LENGTH
        self.distance_buffer = deque(maxlen=BUFFER_SIZE)
        
        # Audio
        self.engine = pyttsx3.init()
        self.last_speech_time = 0
        self.speech_cooldown = 2.0 # seconds (User requested 2s)
        self.last_spoken_distance = 0
        self.speech_lock = threading.Lock()
        
        # Calibration
        self.pending_calibration_distance = None # If set, calibrates on next valid frame
        
        # Mode
        self.target_mode = "face" # "face", "phone", "all"
        
        # Gemini
        self.gemini = GeminiDetector()
        
        # Cache YOLO names check
        self.yolo_classes = self.yolo.names

    def calibrate(self, known_distance_cm, real_width_cm, image_gray, box_width_px):
        """
        F = (P * D) / W
        """
        if box_width_px == 0:
            return
        self.focal_length = (box_width_px * known_distance_cm) / real_width_cm
        print(f"Calibrated Focal Length: {self.focal_length}")

    def get_distance(self, width_px, real_width_cm):
        """
        D = (W * F) / P
        """
        if width_px == 0:
            return 0
        return (real_width_cm * self.focal_length) / width_px

    def speak(self, text):
        def _speak():
            with self.speech_lock:
                try:
                    pythoncom.CoInitialize()
                    # Beep: 1000Hz, 200ms
                    winsound.Beep(1000, 200) 
                    local_engine = pyttsx3.init()
                    local_engine.say(text)
                    local_engine.runAndWait()
                except Exception as e:
                    print(f"TTS Error: {e}")
                finally:
                    pythoncom.CoUninitialize()
        
        threading.Thread(target=_speak).start()

    def process_frame(self, frame):
        """
        Returns processed frame with overlays and current distance info
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width, _ = frame.shape
        
        detected_width_px = 0
        label = ""
        real_width = 0
        
        # Detection
        if self.target_mode == 'face':
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                (x, y, w, h) = largest_face
                detected_width_px = w
                label = "Face"
                real_width = KNOWN_WIDTHS['face']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        else:
            # YOLO detection (Phone or All)
            results = self.yolo(frame, verbose=False)
            
            # Filter logic
            best_box = None
            max_area = 0
            
            for r in results:
                names = r.names
                boxes = r.boxes
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = names[cls_id]
                    b = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                    w = b[2] - b[0]
                    h = b[3] - b[1]
                    area = w * h
                    
                    # Target Mode Filtering
                    is_target = False
                    
                    if self.target_mode == 'all':
                         is_target = True
                    elif self.target_mode == 'phone' and cls_name == 'cell phone':
                        is_target = True # Backward compatibility
                    elif cls_name.lower() == self.target_mode.lower():
                         is_target = True
                        
                    if is_target:
                        # Draw detected objects
                        x1, y1, x2, y2 = map(int, b)
                        color = (0, 255, 0)
                        if self.target_mode != 'all':
                             color = (0, 0, 255) # Red for specific target
                             
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                        cv2.putText(frame, cls_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # Find Largest for Main Feedback
                        if area > max_area:
                            max_area = area
                            best_box = b
                            detected_width_px = w
                            label = cls_name
            
            # GEMINI FALLBACK
            # If no box found via YOLO, OR checking for something YOLO definitely doesn't know?
            # Strategy: If target_mode is NOT 'all' and NOT found in YOLO classes (or just failed to find it?), try Gemini.
            # Best approach: If target_mode is specific and NOT a known YOLO class, run Gemini.
            
            check_gemini = False
            # Check if target_mode is a known YOLO class (fuzzy match)
            is_yolo_supported = False
            if self.target_mode in ['all', 'face', 'phone']:
                is_yolo_supported = True
            else:
                 for id, name in self.yolo_classes.items():
                     if name.lower() == self.target_mode.lower():
                         is_yolo_supported = True
                         break
            
            if not is_yolo_supported:
                check_gemini = True
                
            if check_gemini:
                # Run threaded detection
                self.gemini.detect_thread(frame, self.target_mode)
                
                # Get last result
                g_box = self.gemini.get_parsed_box(frame.shape)
                if g_box:
                    gx, gy, gw, gh = g_box
                    detected_width_px = gw
                    label = self.target_mode
                    
                    # Highlight Gemini Result
                    cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (255, 0, 255), 3)
                    cv2.putText(frame, f"GEMINI: {label}", (gx, gy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                    
                    # Set generic width for unknown objects found by Gemini
                    if label in KNOWN_WIDTHS:
                         real_width = KNOWN_WIDTHS[label]
                    else:
                         real_width = 15.0
                            
            if best_box is not None:
                # Highlight Primary Object
                x1, y1, x2, y2 = map(int, best_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"TARGET: {label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # Check for Known Width
                if label in KNOWN_WIDTHS:
                    real_width = KNOWN_WIDTHS[label]
                else:
                    # Generic width fallback
                    real_width = 15.0 # Average small object

        # Distance Calculation and Feedback
        dist_text = ""
        final_distance = 0
        
        if detected_width_px > 0 and real_width > 0:
            # Check for pending calibration
            if self.pending_calibration_distance is not None:
                # perform calibration: F = (P * D) / W
                self.focal_length = (detected_width_px * self.pending_calibration_distance) / real_width
                print(f"Calibration Successful! New Focal Length: {self.focal_length}")
                self.pending_calibration_distance = None # Reset flag

            dist = self.get_distance(detected_width_px, real_width)
            self.distance_buffer.append(dist)
            
            # Median Filter for stability
            if len(self.distance_buffer) >= 3:
                final_distance = np.median(self.distance_buffer)
            else:
                final_distance = sum(self.distance_buffer) / len(self.distance_buffer)
            
            dist_text = f"Distance: {final_distance:.1f} cm"
            cv2.putText(frame, dist_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif detected_width_px > 0:
            dist_text = "Distance: Unknown Object Width"
            cv2.putText(frame, dist_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Audio Feedback
        current_time = time.time()
        # Audio Feedback
        current_time = time.time()
        if (current_time - self.last_speech_time > 2.0) and detected_width_px > 0: # Check every 2.0s
            should_speak = False
            spoken_text = ""
            
            if final_distance > 0:
                 if final_distance < 50:
                     # Target reached
                     if self.last_spoken_distance > 50 or (current_time - self.last_speech_time > 5.0): # Don't spam "Reached" too fast
                         spoken_text = f"Target {label} reached. {int(final_distance)} centimeters."
                         should_speak = True
                 else:
                    if final_distance >= 100:
                        spoken_text = f"{label} is {final_distance/100:.1f} meters ahead"
                    else:
                        spoken_text = f"{label} is {int(final_distance)} centimeters ahead"
                    should_speak = True
                    self.last_spoken_distance = final_distance
            else:
                 # Just speak label if no distance
                 spoken_text = f"Found {label}"
                 should_speak = True
            
            if should_speak:
                self.speak(spoken_text)
                self.last_speech_time = current_time

        return frame

# Singleton instance to be used by app.py (initialized in app.py)
