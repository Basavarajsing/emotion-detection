"""
Simple demo runner for facial_emotion_recognition package.

Usage:
    python run_demo.py --device cpu
    python run_demo.py --device gpu --gpu_id 0

This script opens the default webcam, runs emotion recognition on each frame and shows the result.
"""
import argparse
import sys
import os

try:
    import cv2 as cv
except Exception as e:
    print("Missing dependency: opencv-python (cv2). Install it with: pip install opencv-python")
    raise

try:
    from facial_emotion_recognition import EmotionRecognition
except Exception as e:
    print("Could not import package 'facial_emotion_recognition'. If you're running from the source tree, run this from the project root and/or install in editable mode: pip install -e .")
    raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu', help='Device to run on')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id (if device=gpu)')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index')
    args = parser.parse_args()

    # quick check for model file
    model_path = os.path.join(os.path.dirname(__file__), 'facial_emotion_recognition', 'model', 'model.pkl')
    if not os.path.exists(model_path):
        print(f"Warning: model file not found at {model_path}\nMake sure the repository includes 'facial_emotion_recognition/model/model.pkl' or install the package from PyPI.")

    try:
        er = EmotionRecognition(device=args.device, gpu_id=args.gpu_id)
    except Exception as e:
        print('Failed to initialize EmotionRecognition:')
        raise

    cam = cv.VideoCapture(args.camera)
    if not cam.isOpened():
        print(f'Unable to open camera index {args.camera}.')
        sys.exit(1)

    print('Press ESC to quit.')
    while True:
        success, frame = cam.read()
        if not success:
            print('Failed to read frame from camera, exiting.')
            break

        try:
            out_frame = er.recognise_emotion(frame, return_type='BGR')
        except Exception as e:
            print('Error during recognition:')
            raise

        cv.imshow('facial_emotion_recognition demo', out_frame)
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

    cam.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
