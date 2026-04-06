from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera, sign_buffer, buffer_lock
import threading
import webbrowser
import os
import sys
import subprocess
import time

app = Flask(__name__)
camera_instance = None

camera_instance = None

def get_camera():
    global camera_instance

    if camera_instance is None:
        camera_instance = VideoCamera()
        print("Camera created ONCE")

    return camera_instance

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/collect')
def collect():

    global camera_instance

    if camera_instance:
        camera_instance.release()
        camera_instance = None

    cam = get_camera()
    cam.mode = "collect"

    return render_template('collect.html')

@app.route('/save_image', methods=['POST'])
def save_image():
    label = request.form.get('label')

    if not label:
        return jsonify({'error': 'No label provided'}), 400

    camera = get_camera()
    success = camera.save_current_frame(label)

    if success:
        dataset_path = os.path.join('dataset', label)

        if os.path.exists(dataset_path):
            count = len([
                f for f in os.listdir(dataset_path)
                if f.endswith('.jpg')
            ])
        else:
            count = 0

        return jsonify({'status': 'check', 'count': count})
    else:
        return jsonify({'error': 'No face/hand detected'}), 400

@app.route('/get_count')
def get_count():
    label = request.args.get('label')
    if not label:
        return jsonify({'count': 0})
    
    dataset_path = os.path.join('dataset', label)
    if os.path.exists(dataset_path):
        count = len([f for f in os.listdir(dataset_path) if f.endswith('.jpg')])
        return jsonify({'count': count})
    return jsonify({'count': 0})

@app.route('/train_page')
def train_page():
    return render_template('train.html')

@app.route('/start_training', methods=['POST'])
def start_training():

    global camera_instance

    # ⭐ STOP CAMERA BEFORE TRAINING
    if camera_instance:
        camera_instance.release()
        camera_instance = None
        print("Camera stopped before training")

    try:
        result = subprocess.run(
            [sys.executable, 'train.py'],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        print(result.stdout)
        print(result.stderr)

        # ⭐ FORCE MODEL RELOAD
        import camera
        camera.model_loaded = False

        return jsonify({
            "status": "Model Trained Successfully",
            "log": result.stdout
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/recognize')
def recognize():

    global camera_instance

    if camera_instance:
        camera_instance.release()
        camera_instance = None

    cam = get_camera()
    cam.mode = "recognize"

    return render_template('recognize.html')

def gen(camera):

    global camera_instance

    try:
        while True:

            if camera_instance is None:
                break

            frame = camera.get_frame_bytes()

            if frame is None:
                time.sleep(0.05)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except GeneratorExit:
        print("Browser closed → releasing camera")

    finally:
        if camera_instance:
            camera_instance.release()
            camera_instance = None
            print("Camera generator stopped")

@app.route('/video_feed')
def video_feed():
    cam = get_camera()
    print("VIDEO FEED MODE:", cam.mode)
    return Response(gen(cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/frame')
def frame():
    camera = get_camera()
    frame = camera.get_frame_bytes()
    return Response(frame, mimetype='image/jpeg')

@app.route('/translate', methods=['POST'])
def translate():
    global sign_buffer
    with buffer_lock:
        if not sign_buffer:
            return jsonify({'sentence': ''})

        final_sentence = []
        final_sentence.append(sign_buffer[0])

        for i in range(1, len(sign_buffer)):
            if sign_buffer[i] != sign_buffer[i-1]:
                final_sentence.append(sign_buffer[i])

        return jsonify({'sentence': ' '.join(final_sentence)})
    

@app.route('/stop_camera')
def stop_camera():
    global camera_instance

    if camera_instance is not None:
        camera_instance.release()
        camera_instance = None

    return "camera stopped"

@app.route('/start_camera')
def start_camera():
    global camera_instance

    previous_mode = "collect"

    if camera_instance is not None:
        previous_mode = camera_instance.mode
        camera_instance.release()

    camera_instance = VideoCamera()

    # ⭐ restore previous mode
    camera_instance.mode = previous_mode

    print(f"Camera restarted safely with mode: {previous_mode}")

    return "started"

@app.route('/reset', methods=['POST'])
def reset_buffer():
    global sign_buffer
    cam = get_camera()

    with buffer_lock:
        sign_buffer.clear()

    cam.pred_buffer.clear()
    cam.last_sign = None

    print("Buffer Reset")

last_ping_time = time.time() + 15 # 15s grace period on startup

@app.route('/ping')
def ping():
    global last_ping_time
    last_ping_time = time.time()
    return "ok"

def check_heartbeat():
    while True:
        time.sleep(3)
        if time.time() - last_ping_time > 6:
            print("No active browser tabs detected. Shutting down the terminal...")
            os._exit(0)

if __name__ == '__main__':
    threading.Thread(target=check_heartbeat, daemon=True).start()
    threading.Timer(4, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True, use_reloader=False)