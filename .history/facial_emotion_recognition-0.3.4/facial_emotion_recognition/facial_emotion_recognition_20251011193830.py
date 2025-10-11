from .networks import NetworkV2
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv
from facenet_pytorch import MTCNN
import os
import time


class EmotionRecognition(object):
    def __init__(self, device, gpu_id=0):
        assert device == 'cpu' or device == 'gpu'
        if torch.cuda.is_available():
            if device == 'cpu':
                print('[*]Warning: Your device have GPU, for better performance do EmotionRecognition(device=gpu)')
                self.device = torch.device('cpu')
            if device == 'gpu':
                self.device = torch.device(f'cuda:{str(gpu_id)}')
        else:
            if device == 'gpu':
                print('[*]Warning: No GPU is detected, so cpu is selected as device')
                self.device = torch.device('cpu')
            if device == 'cpu':
                self.device = torch.device('cpu')
        self.network = NetworkV2(in_c=1, nl=32, out_f=7).to(self.device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

        model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
        # Load with map_location to support CPU-only environments when model was saved on CUDA
        try:
            model_dict = torch.load(model_path, map_location=self.device)
        except Exception:
            # Fallback: map to CPU
            model_dict = torch.load(model_path, map_location=torch.device('cpu'))
        print(f'[*] Accuracy: {model_dict["accuracy"]}')
        self.emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        self.network.load_state_dict(model_dict['network'])
        self.network.eval()
        # trackers: list of dicts with keys: id, bbox, emotion, last_update, last_seen
        self.trackers = []
        self.next_tracker_id = 0

    def _predict(self, image):
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.network(tensor)

        out_np = output.detach().cpu().numpy()[0]
        # If outputs look like probabilities (0..1), use them directly; otherwise use softmax
        if out_np.max() <= 1.0 and out_np.min() >= 0.0:
            probs = out_np
        else:
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

        index = int(np.argmax(probs))
        return self.emotions[index]

    def recognise_emotion(self, frame, return_type='BGR', downscale=1, hold_time=20, match_threshold=None):
        f_h, f_w, c = frame.shape
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Optionally run MTCNN on a downscaled copy to speed up CPU processing
        if downscale is not None and downscale > 1:
            small = cv.resize(frame, (f_w // downscale, f_h // downscale))
            boxes, _ = self.mtcnn.detect(small)
            # map boxes back to original coordinates
            if boxes is not None:
                boxes = boxes * float(downscale)
        else:
            boxes, _ = self.mtcnn.detect(frame)
        if boxes is not None:
            now = time.time()
            # match_threshold in pixels
            if match_threshold is None:
                match_threshold = int(max(f_w, f_h) * 0.12)

            # process each detected face, match to trackers by center distance
            for i in range(len(boxes)):
                bx = boxes[i]
                x1, y1, x2, y2 = int(round(bx[0])), int(round(bx[1])), int(round(bx[2])), int(round(bx[3]))
                # guard against invalid crop
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(f_w, x2), min(f_h, y2)
                if x2c <= x1c or y2c <= y1c:
                    continue

                detected_emotion = self._predict(gray[y1c:y2c, x1c:x2c])

                # compute center
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # find best matching tracker
                best = None
                best_dist = None
                for t in self.trackers:
                    tbx = t['bbox']
                    tcx = (tbx[0] + tbx[2]) / 2.0
                    tcy = (tbx[1] + tbx[3]) / 2.0
                    dist = (tcx - cx) ** 2 + (tcy - cy) ** 2
                    if best is None or dist < best_dist:
                        best = t
                        best_dist = dist

                matched = None
                if best is not None and best_dist is not None and best_dist <= (match_threshold ** 2):
                    matched = best

                if matched is not None:
                    # matched existing tracker
                    # update last_seen and bbox
                    matched['bbox'] = [x1, y1, x2, y2]
                    matched['last_seen'] = now
                    # only update displayed emotion if hold_time has passed
                    if now - matched.get('last_update', 0) >= hold_time:
                        matched['emotion'] = detected_emotion
                        matched['last_update'] = now
                    display_emotion = matched['emotion']
                else:
                    # create new tracker
                    tid = self.next_tracker_id
                    self.next_tracker_id += 1
                    tracker = {
                        'id': tid,
                        'bbox': [x1, y1, x2, y2],
                        'emotion': detected_emotion,
                        'last_update': now,
                        'last_seen': now
                    }
                    self.trackers.append(tracker)
                    display_emotion = detected_emotion

                # draw box and label
                frame = cv.rectangle(frame, (x1, y1), (x2, y2), color=[0, 255, 0], thickness=1)
                frame = cv.rectangle(frame, (x1, y1 - int(f_h * 0.03125)), (x1 + int(f_w * 0.125), y1), color=[0, 255, 0], thickness=-1)
                frame = cv.putText(frame, text=display_emotion, org=(x1 + 5, y1 - 3), fontFace=cv.FONT_HERSHEY_PLAIN,
                                   color=[0, 0, 0], fontScale=1, thickness=1)

            # remove stale trackers not seen for >5 seconds
            self.trackers = [t for t in self.trackers if now - t.get('last_seen', now) <= 5.0]

            if return_type == 'BGR':
                return frame
            if return_type == 'RGB':
                return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        else:
            print('No face detected')
            if return_type == 'BGR':
                return frame
            if return_type == 'RGB':
                return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
