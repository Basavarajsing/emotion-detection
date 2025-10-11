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
    parser.add_argument('--mode', choices=['webcam', 'image'], default=None, help="Run mode: 'webcam' to open camera or 'image' to run on a single image")
    parser.add_argument('--image_path', type=str, default=None, help='Path to image file when --mode image')
    parser.add_argument('--process_every', type=int, default=3, help='Process (run detection) every N frames for webcam; higher -> slower, more stable')
    parser.add_argument('--delay_ms', type=int, default=1, help='cv2.waitKey delay in milliseconds (higher -> slower display)')
    parser.add_argument('--downscale', type=int, default=1, help='Downscale factor for detection (run MTCNN on a smaller image). Integer >=1. Higher -> faster but less precise')
    parser.add_argument('--hold_time', type=int, default=20, help='Seconds to hold a detected emotion before updating to a new value')
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

    # Interactive fallback: ask user if mode not provided
    mode = args.mode
    if mode is None:
        print("Choose mode:\n  1) webcam\n  2) image file")
        choice = input('Enter 1 or 2: ').strip()
        mode = 'webcam' if choice == '1' else 'image'

    process_every = max(1, int(args.process_every))
    delay_ms = max(1, int(args.delay_ms))
    downscale = max(1, int(args.downscale))

    if mode == 'image':
        img_path = args.image_path
        if not img_path:
            img_path = input('Enter image file path: ').strip()
        if not os.path.exists(img_path):
            print(f'Image not found: {img_path}')
            sys.exit(1)

        img = cv.imread(img_path)
        if img is None:
            print('Failed to read image file')
            sys.exit(1)

        out = er.recognise_emotion(img, return_type='BGR', downscale=downscale, hold_time=args.hold_time)
        cv.imshow('facial_emotion_recognition - image', out)
        print('Press any key to exit the image view.')
        cv.waitKey(0)
        cv.destroyAllWindows()
        return

    # Webcam mode
    cam = cv.VideoCapture(args.camera)
    if not cam.isOpened():
        print(f'Unable to open camera index {args.camera}.')
        sys.exit(1)

    print('Press ESC to quit. Processing every', process_every, 'frames. Display delay (ms):', delay_ms)
    last_frame = None
    frame_idx = 0
    while True:
        success, frame = cam.read()
        if not success:
            print('Failed to read frame from camera, exiting.')
            break

        if frame_idx % process_every == 0 or last_frame is None:
            # run detection and annotate (with optional downscaling for speed)
            try:
                last_frame = er.recognise_emotion(frame, return_type='BGR', downscale=downscale, hold_time=args.hold_time)
            except Exception:
                print('Error during recognition on frame', frame_idx)
                raise

        cv.imshow('facial_emotion_recognition demo', last_frame)
        key = cv.waitKey(delay_ms)
        if key == 27:  # ESC
            break

        frame_idx += 1

    cam.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
