import webbrowser
import threading
import cv2
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import shutil
# Check for face module (requires opencv-contrib-python)
if hasattr(cv2, 'face'):
    LBPHFaceRecognizer_create = cv2.face.LBPHFaceRecognizer_create
else:
    print("WARNING: cv2.face module not found! Face recognition will fail. Ensure opencv-contrib-python is installed.")
    # Dummy function to prevent immediate crash on load, will crash on use if not handled
    def LBPHFaceRecognizer_create():
        raise ImportError("cv2.face module is missing")
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configuration
DATASET_DIR = 'dataset'
MODEL_FILE = 'model.yml'
CASCADE_FILE = 'haarcascade_frontalface_default.xml'

# Create dataset directory if not exists
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Load Face Detector
face_cascade = cv2.CascadeClassifier(CASCADE_FILE)

# Global variables for capture state
capture_name = ""
capturing = False
capture_count = 0
MAX_CAPTURE = 50

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/train_page')
def train_page():
    return render_template('train.html')

@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

# --- Logic Modules ---

# 1. Capture Module

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capture_name, capturing, capture_count
    data = request.json
    capture_name = data.get('name')
    if not capture_name:
        return jsonify({'status': 'error', 'message': 'Name is required'})
    
    # Create directory for the person
    person_dir = os.path.join(DATASET_DIR, capture_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    else:
        # Optional: Clear existing images if re-capturing same person? 
        # For now, let's append or overwrite implies we might want to clean up.
        # But instructions say "Capture at least 30-50", let's just use the dir.
        pass

    capture_count = 0
    capturing = True
    return jsonify({'status': 'success', 'message': f'Started capturing for {capture_name}'})

def gen_capture():
    global capturing, capture_count, capture_name
    camera = cv2.VideoCapture(0)
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                if capturing and capture_count < MAX_CAPTURE:
                    capture_count += 1
                    person_dir = os.path.join(DATASET_DIR, capture_name)
                    # Save image
                    img_path = os.path.join(person_dir, f"{capture_count}.jpg")
                    cv2.imwrite(img_path, gray[y:y+h, x:x+w])
            
            if capturing:
                cv2.putText(frame, f"Capturing: {capture_count}/{MAX_CAPTURE}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if capture_count >= MAX_CAPTURE:
                    capturing = False # Stop capturing
                    cv2.putText(frame, "Capture Complete!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()

@app.route('/video_feed_capture')
def video_feed_capture():
    return Response(gen_capture(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_status')
def capture_status():
    global capture_count, capturing
    return jsonify({'count': capture_count, 'uploading': capturing, 'complete': capture_count >= MAX_CAPTURE})


# 2. Training Module

@app.route('/train_model', methods=['POST'])
def train_model():
    recognizer = LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(CASCADE_FILE)
    
    faces = []
    ids = []
    names = {}
    
    # Map names to integer IDs
    current_id = 0
    
    # Check dataset
    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        return jsonify({'status': 'error', 'message': 'Dataset is empty.'})

    try:
        # Sort directories to ensure consistent ID generation
        sorted_dirs = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
        
        if not sorted_dirs:
             return jsonify({'status': 'error', 'message': 'No person directories found in dataset.'})

        for person_name in sorted_dirs:
            person_dir = os.path.join(DATASET_DIR, person_name)
                
            names[current_id] = person_name
            
            # Filter for image files only
            image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_name in image_files:
                image_path = os.path.join(person_dir, image_name)
                try:
                    # Read image as grayscale
                    img_numpy = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img_numpy is None:
                        continue
                    faces.append(img_numpy)
                    ids.append(current_id)
                except Exception as e:
                    print(f"Error reading image {image_path}: {e}")
            
            current_id += 1
            
        if not faces:
             return jsonify({'status': 'error', 'message': 'No valid faces found in dataset.'})

        recognizer.train(faces, np.array(ids))
        recognizer.save(MODEL_FILE)
        
        return jsonify({'status': 'success', 'message': 'Training completed successfully!'})
    except Exception as e:
         print(f"Training Error: {e}")
         return jsonify({'status': 'error', 'message': str(e)})


# 3. Recognition Module

# ... Imports at the top ...
import threading
import time
import pyttsx3

# Initialize TTS Engine (Global)
engine = pyttsx3.init()
last_spoken_time = {}
speech_lock = threading.Lock()

def speak_name(name):
    """Threaded function to speak name without blocking video feed"""
    with speech_lock:
        try:
            # Re-init engine inside thread if needed for some OS, but global usually works for simple cases
            # or better: assume engine is thread-unsafe and use a queue if it crashes.
            # Simple approach: Create a new engine instance per thread or use lock.
            # pyttsx3 runAndWait blocks.
            local_engine = pyttsx3.init() 
            local_engine.say(f"Identified {name}")
            local_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")

@app.route('/video_feed_recognize')
def video_feed_recognize():
    return Response(gen_recognize(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_recognize():
    print("Initializing Recognition Generator...") # DEBUG
    camera = cv2.VideoCapture(0)
    recognizer = LBPHFaceRecognizer_create()
    
    model_loaded = False
    if os.path.exists(MODEL_FILE):
        try:
            recognizer.read(MODEL_FILE)
            model_loaded = True
            print(f"Model loaded successfully from {MODEL_FILE}") # DEBUG
        except Exception as e:
             print(f"Error loading model: {e}")
    else:
        print(f"Model file {MODEL_FILE} not found!") # DEBUG
    
    names = {}
    if os.path.exists(DATASET_DIR):
        sorted_dirs = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
        idx = 0
        for person_name in sorted_dirs:
             names[idx] = person_name
             idx += 1
        print(f"Loaded names mapping: {names}") # DEBUG
    else:
        print("Dataset directory not found!") # DEBUG
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read frame from camera") # DEBUG
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Check if cascade is loaded
            if face_cascade.empty():
                print("Error: Face cascade not loaded!") # DEBUG
            
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if model_loaded:
                    try:
                        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                        
                        # Tuned Threshold: 85 (Lower is stricter, 0 is perfect match)
                        # Standard LBPH is around 50-80 for good match. 100 is very loose.
                        if confidence < 85:
                            name = names.get(id, "Unknown")
                            conf_str = "  {0}%".format(round(100 - confidence))
                            
                            # TTS Logic
                            current_time = time.time()
                            if name != "Unknown":
                                if current_time - last_spoken_time.get(name, 0) > 30: # 30 seconds cooldown
                                    last_spoken_time[name] = current_time
                                    # Start speech thread
                                    t = threading.Thread(target=speak_name, args=(name,))
                                    t.start()
                        else:
                            name = "Unknown"
                            conf_str = "  {0}%".format(round(100 - confidence))
                    except Exception as e:
                        name = "Error"
                        conf_str = ""
                        print(f"Prediction error: {e}")
                else:
                    name = "No Model"
                    conf_str = ""
                
                cv2.putText(frame, str(name), (x+5, y-5), font, 1, (255, 255, 255), 2)
                cv2.putText(frame, str(conf_str), (x+5, y+h-5), font, 1, (255, 255, 0), 1)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()
        print("Camera released in recognition loop") # DEBUG


# ... (Existing imports)
from object_finder import ObjectFinder

# Initialize Object Finder
obj_finder = ObjectFinder(CASCADE_FILE)

# ... (Existing routes) ...

# --- Find My Object Routes ---

@app.route('/find_object')
def find_object():
    return render_template('find_object.html')

@app.route('/video_feed_find_object')
def video_feed_find_object():
    return Response(gen_find_object(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_find_object():
    camera = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            # Process frame using ObjectFinder module
            frame = obj_finder.process_frame(frame)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()

@app.route('/api/set_target', methods=['POST'])
def set_target():
    data = request.json
    mode = data.get('mode', 'face')
    # Allow any mode, validation happens in object_finder logic or ignored
    obj_finder.target_mode = mode
    return jsonify({'status': 'success', 'mode': mode})

@app.route('/api/calibrate', methods=['POST'])
def calibrate_endpoint():
    data = request.json
    distance = float(data.get('distance', 50))
    # Set pending flag, next frame will calibrate
    obj_finder.pending_calibration_distance = distance
    return jsonify({'status': 'success', 'message': f'Calibration pending for {distance}cm...'})

if __name__ == '__main__':
    def open_browser():
        webbrowser.open_new("http://localhost:5000")

    # Open browser only once
    threading.Timer(1.0, open_browser).start()

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,        # 🔥 IMPORTANT
        use_reloader=False # 🔥 IMPORTANT
    )

