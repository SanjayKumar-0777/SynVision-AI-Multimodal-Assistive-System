import pyttsx3
import threading

def speak_func(text):
    try:
        engine = pyttsx3.init()
        # Set property before doing anything
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
        # engine.stop() # Good practice?
    except Exception as e:
        print(f"TTS Error: {e}")

def speak(text):
    """
    Speaks the text in a separate thread to avoid blocking the main loop.
    """
    if not text: return
    thread = threading.Thread(target=speak_func, args=(text,), daemon=True)
    thread.start()
