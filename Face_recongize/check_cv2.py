
import cv2
import sys

print(f"OpenCV Version: {cv2.__version__}")
try:
    import cv2.face
    print("cv2.face module imported successfully via 'import cv2.face'")
    print(f"Directory of cv2.face: {dir(cv2.face)}")
except ImportError as e:
    print(f"Failed to 'import cv2.face': {e}")

try:
    from cv2 import face
    print("face module imported successfully from cv2")
except ImportError as e:
    print(f"Failed to 'from cv2 import face': {e}")
except AttributeError as e:
    print(f"Failed access face from cv2: {e}")

try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("LBPHFaceRecognizer_create() found and instantiated.")
except Exception as e:
    print(f"Failed to create LBPHFaceRecognizer: {e}")
