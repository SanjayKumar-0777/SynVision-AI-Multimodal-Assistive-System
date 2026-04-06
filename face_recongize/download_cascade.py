import urllib.request
import os

url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
filename = "c:/Face_recongize/haarcascade_frontalface_default.xml"

print(f"Downloading {filename}...")
try:
    urllib.request.urlretrieve(url, filename)
    print("Download complete.")
except Exception as e:
    print(f"Error downloading: {e}")
