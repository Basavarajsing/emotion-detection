"""
Top-level runner that forwards to the package demo (copied from the package folder).
Run this from the project root: python .\run_demo.py --mode webcam
"""
from facial_emotion_recognition import EmotionRecognition

# Delegate to the packaged demo to avoid duplication. Import and call.
from facial_emotion_recognition import __file__ as pkg_file
# Execute the package's demo file for convenience
import runpy
runpy.run_path(r"c:\Users\basav\Downloads\facial_emotion_recognition-0.3.4\facial_emotion_recognition-0.3.4\run_demo.py", run_name='__main__')
