"""
Face Detection and Recognition - OpenCV
TechGuide AI Internship | Project 5

Install once :  pip install opencv-contrib-python

Commands
  python face_app.py webcam                    live face detection (webcam)
  python face_app.py image  photo.jpg          detect faces in an image
  python face_app.py video  clip.mp4           detect faces in a video file
  python face_app.py register --name Aman      capture your face for recognition
  python face_app.py recognize                 live face RECOGNITION (names)

How it works
- DETECTION : a pre-trained Haar Cascade (Viola-Jones) scans the image at many
  scales and finds regions that look like a frontal face.
- RECOGNITION: the faces you register are used to train an LBPH
  (Local Binary Patterns Histograms) model. For each new face, the model
  predicts a name. A LOWER confidence value means a BETTER match.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CASCADE_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("output")
FACE_SIZE = (200, 200)
GREEN, RED, WHITE, BLACK = (0, 220, 0), (0, 0, 230), (255, 255, 255), (0, 0, 0)


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------
class FaceDetector:
    """Wraps the Haar Cascade face detector."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5,
                 min_size: int = 60):
        self.cascade = cv2.CascadeClassifier(CASCADE_FILE)
        if self.cascade.empty():
            sys.exit(f"Could not load cascade file: {CASCADE_FILE}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = (min_size, min_size)

    def to_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.equalizeHist(gray)  # improves detection in uneven lighting

    def detect(self, frame: np.ndarray):
        """Return a list of (x, y, w, h) boxes and the grayscale image."""
        gray = self.to_gray(frame)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors, minSize=self.min_size)
        return [tuple(int(v) for v in f) for f in faces], gray


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def draw_label(frame, text, x, y, colour=GREEN):
    """Text with a filled background so it is readable on any image."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = max(y - th - base - 4, 0)
    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + base + 4), colour, -1)
    cv2.putText(frame, text, (x + 4, top + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2, cv2.LINE_AA)


def draw_faces(frame, faces, labels=None, colours=None):
    for i, (x, y, w, h) in enumerate(faces):
        colour = colours[i] if colours else GREEN
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        draw_label(frame, labels[i] if labels else "Face", x, y, colour)


def draw_status(frame, text):
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                WHITE, 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                BLACK, 1, cv2.LINE_AA)


def open_camera(index: int):
    cap = (cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt"
           else cv2.VideoCapture(index))
    if not cap.isOpened():
        sys.exit("Could not open the webcam. Close other apps using it, "
                 "or try --camera 1.")
    return cap


def crop_face(gray, box):
    x, y, w, h = box
    return cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)


# --------------------------------------------------------------------------
# 1. Live webcam detection
# --------------------------------------------------------------------------
def run_webcam(args):
    detector = FaceDetector()
    cap = open_camera(args.camera)
    print("Webcam started.  Press 'q' to quit, 's' to save a screenshot.")
    OUTPUT_DIR.mkdir(exist_ok=True)
    prev = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view feels natural
        faces, _ = detector.detect(frame)
        draw_faces(frame, faces)

        now = time.perf_counter()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        draw_status(frame, f"Faces: {len(faces)}   FPS: {fps:.0f}   (q = quit, s = save)")
        cv2.imshow("Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = OUTPUT_DIR / f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"Saved {path}")

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# 2. Image detection
# --------------------------------------------------------------------------
def run_image(args):
    frame = cv2.imread(args.path)
    if frame is None:
        sys.exit(f"Could not read image: {args.path}")

    detector = FaceDetector()
    faces, _ = detector.detect(frame)
    draw_faces(frame, faces, [f"Face {i + 1}" for i in range(len(faces))])
    draw_status(frame, f"Faces found: {len(faces)}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = Path(args.output) if args.output else OUTPUT_DIR / f"{Path(args.path).stem}_detected.jpg"
    cv2.imwrite(str(out), frame)
    print(f"Found {len(faces)} face(s).  Saved result to {out}")

    if not args.no_show:
        cv2.imshow("Face Detection - press any key to close", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# 3. Video file detection
# --------------------------------------------------------------------------
def run_video(args):
    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.path}")

    detector = FaceDetector()
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = Path(args.output) if args.output else OUTPUT_DIR / f"{Path(args.path).stem}_detected.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    total_faces, frames = 0, 0
    print("Processing video...  press 'q' in the window to stop early.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces, _ = detector.detect(frame)
        draw_faces(frame, faces)
        draw_status(frame, f"Faces: {len(faces)}")
        writer.write(frame)
        total_faces += len(faces)
        frames += 1

        if not args.no_show:
            cv2.imshow("Video Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Processed {frames} frames, {total_faces} face detections.  Saved to {out}")


# --------------------------------------------------------------------------
# 4. Register a new person (collect training images)
# --------------------------------------------------------------------------
def run_register(args):
    name = args.name.strip()
    if not name:
        sys.exit("Please provide a name:  --name Aman")

    person_dir = DATASET_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    detector = FaceDetector()
    cap = open_camera(args.camera)

    saved, last_save = 0, 0.0
    print(f"Registering '{name}'. Look at the camera and slowly turn your head "
          "a little. Press 'q' to stop.")

    while saved < args.samples:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        faces, gray = detector.detect(frame)

        if len(faces) == 1 and time.perf_counter() - last_save > 0.2:
            cv2.imwrite(str(person_dir / f"{int(time.time() * 1000)}.png"),
                        crop_face(gray, faces[0]))
            saved += 1
            last_save = time.perf_counter()

        draw_faces(frame, faces, [f"Capturing {saved}/{args.samples}"] * len(faces))
        msg = ("Only ONE face should be visible" if len(faces) > 1
               else "No face found" if not faces else f"Saved {saved}/{args.samples}")
        draw_status(frame, msg)
        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} images to {person_dir}")


# --------------------------------------------------------------------------
# 5. Live recognition
# --------------------------------------------------------------------------
def train_recognizer():
    """Train an LBPH model from every folder inside dataset/."""
    if not hasattr(cv2, "face"):
        sys.exit("Face recognition needs the contrib package:\n"
                 "  pip uninstall opencv-python\n"
                 "  pip install opencv-contrib-python")

    images, ids, names = [], [], []
    people = sorted(p for p in DATASET_DIR.glob("*") if p.is_dir()) if DATASET_DIR.exists() else []
    for person in people:
        for file in person.glob("*.png"):
            img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img, FACE_SIZE))
                ids.append(len(names))
        if any(i == len(names) for i in ids):
            names.append(person.name)

    if not images:
        sys.exit("No training images found. Register someone first:\n"
                 "  python face_app.py register --name YourName")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(ids))
    print(f"Trained on {len(images)} images of {len(names)} person(s): {', '.join(names)}")
    return recognizer, names


def run_recognize(args):
    recognizer, names = train_recognizer()
    detector = FaceDetector()
    cap = open_camera(args.camera)
    print(f"Recognition started (threshold {args.threshold}). Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        faces, gray = detector.detect(frame)

        labels, colours = [], []
        for box in faces:
            idx, distance = recognizer.predict(crop_face(gray, box))
            if distance < args.threshold:
                labels.append(f"{names[idx]} ({distance:.0f})")
                colours.append(GREEN)
            else:
                labels.append(f"Unknown ({distance:.0f})")
                colours.append(RED)
        draw_faces(frame, faces, labels, colours)
        draw_status(frame, f"Faces: {len(faces)}   (q = quit)")
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="Face detection and recognition")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("webcam", help="live face detection")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.set_defaults(func=run_webcam)

    p = sub.add_parser("image", help="detect faces in an image")
    p.add_argument("path")
    p.add_argument("--output")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    p.set_defaults(func=run_image)

    p = sub.add_parser("video", help="detect faces in a video file")
    p.add_argument("path")
    p.add_argument("--output")
    p.add_argument("--no-show", action="store_true")
    p.set_defaults(func=run_video)

    p = sub.add_parser("register", help="capture face images of a person")
    p.add_argument("--name", required=True)
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--camera", type=int, default=0)
    p.set_defaults(func=run_register)

    p = sub.add_parser("recognize", help="live face recognition")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threshold", type=float, default=70,
                   help="lower = stricter matching (default 70)")
    p.set_defaults(func=run_recognize)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)