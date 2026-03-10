"""
Single-script web app: Live posture/body detection + 3D model.
Rewritten for MediaPipe 0.10+ Tasks API (mp.solutions removed).
On first run, model files (~30 MB total) are auto-downloaded to ./models/
"""

import cv2
import math
import time
import json
import threading
import urllib.request
import os

from flask import Flask, Response, send_file
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ==================== MODEL AUTO-DOWNLOAD ====================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = {
    "pose": (
        os.path.join(MODEL_DIR, "pose_landmarker_lite.task"),
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    ),
    "hand": (
        os.path.join(MODEL_DIR, "hand_landmarker.task"),
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    ),
    "face": (
        os.path.join(MODEL_DIR, "face_landmarker.task"),
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    ),
}


def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, (path, url) in MODELS.items():
        if not os.path.exists(path):
            print(f"  Downloading {name} model (this may take a moment)...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"  ✓ {name} model saved to {path}")
            except Exception as exc:
                print(f"  ✗ Failed to download {name} model: {exc}")
                raise SystemExit(1)
        else:
            print(f"  ✓ {name} model already present")


# ==================== CONFIGURATION ====================
CONFIG = {
    "camera_id": 0,
    "frame_width": 640,
    "frame_height": 480,
    "good_posture_min_angle": 140,
    "good_posture_max_angle": 180,
    "tracking_confidence": 0.5,
    "enable_hand_detection": True,
    "max_hands": 2,
    "enable_face_detection": True,
    "max_faces": 1,
    "enable_eye_tracking": True,
    "blink_threshold": 0.21,
    "enable_distance_estimation": True,
    # Set these after calibration if desired:
    "calibrated_distance": None,   # cm
    "calibrated_face_width": None, # pixels at that distance
    "min_safe_distance": 20,
    "max_safe_distance": 40,
    "bad_posture_alert_delay": 10,
    "enable_visual_alert": True,
    "show_skeleton": True,
    "show_angles": True,
    "show_statistics": True,
    "mirror_mode": False,
    "show_landmark_labels": True,   # draw names next to pose/face/hand dots
    "detection_confidence": 0.6,    # higher = more stable, fewer false detections (pose/hand/face)
    "pose_smoothing_alpha": 0.55,   # blend: higher = livelier 3D, same-to-same with detection
}

# ==================== LANDMARK TOPOLOGY ====================
# BlazePose 33-point connections (same indices as legacy API)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Key joints for 3D model driving (BlazePose indices)
# Arms: 11,12=shoulders 13,14=elbows 15,16=wrists
# Legs: 23,24=hips 25,26=knees 27,28=ankles
DRIVER_JOINTS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

# BlazePose 33 landmark names (for on-screen labels)
POSE_LANDMARK_NAMES = {
    0: "nose", 1: "L_eye_in", 2: "L_eye", 3: "L_eye_out", 4: "R_eye_in", 5: "R_eye", 6: "R_eye_out",
    7: "L_ear", 8: "R_ear", 9: "mouth_L", 10: "mouth_R",
    11: "L_shoulder", 12: "R_shoulder", 13: "L_elbow", 14: "R_elbow",
    15: "L_wrist", 16: "R_wrist", 17: "L_pinky", 18: "R_pinky", 19: "L_index", 20: "R_index",
    21: "L_thumb", 22: "R_thumb",
    23: "L_hip", 24: "R_hip", 25: "L_knee", 26: "R_knee",
    27: "L_ankle", 28: "R_ankle", 29: "L_heel", 30: "R_heel", 31: "L_foot", 32: "R_foot",
}
# Driver joint index -> label (for 3D driver dots)
DRIVER_JOINT_LABELS = {
    11: "L_shoulder", 12: "R_shoulder", 13: "L_elbow", 14: "R_elbow",
    15: "L_wrist", 16: "R_wrist",
    23: "L_hip", 24: "R_hip", 25: "L_knee", 26: "R_knee", 27: "L_ankle", 28: "R_ankle",
}

# Face landmarks for head driving (MediaPipe Face Mesh 468 indices)
# nose_tip, left_eye, right_eye, chin, left_cheek, right_cheek
FACE_DRIVER_INDICES = {
    "nose": 4,
    "left_eye": 33,
    "right_eye": 263,
    "chin": 152,
    "left_cheek": 234,
    "right_cheek": 454,
}

# ==================== SHARED STATE ====================
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_jpeg = None
        self.posture_data = {
            "angle": 0,
            "status": "unknown",
            "good_posture": False,
            "good_time": 0,
            "bad_time": 0,
            "posture_type": "",
            "blinks": 0,
            "blinks_per_minute": 0,
            "distance_cm": None,
            "distance_status": "Unknown",
            "timestamp": 0,
            "pose_landmarks": None,
            "driver_joints": None,   # arms + legs for 3D model
            "head_joints": None,     # face points for head driving
            "hand_landmarks": None,  # left/right hand 21 points for 3D hands
        }

    def set_frame(self, jpeg_bytes):
        with self.lock:
            self.frame_jpeg = jpeg_bytes

    def get_frame(self):
        with self.lock:
            return self.frame_jpeg

    def set_posture(self, data):
        with self.lock:
            self.posture_data = {**self.posture_data, **data, "timestamp": time.time()}

    def get_posture(self):
        with self.lock:
            return self.posture_data.copy()


shared = SharedState()

# ==================== DRAWING HELPERS ====================
def draw_text_with_background(frame, text, position, font_scale=0.6,
                               text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + 5), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)


def draw_landmark_label(frame, text, px, py, font_scale=0.35, above=True):
    """Draw a small label above (or below) a dot so landmarks are named by position."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
    x = int(px - tw / 2)
    y = int(py - 8) if above else int(py + th + 8)
    y = max(th + 2, min(frame.shape[0] - 2, y))
    x = max(2, min(frame.shape[1] - tw - 2, x))
    cv2.rectangle(frame, (x - 2, y - th - 2), (x + tw + 2, y + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, (200, 255, 200), 1)


def draw_connections(frame, norm_landmarks, connections, w, h,
                     dot_color=(0, 255, 0), line_color=(0, 0, 255),
                     dot_r=3, line_t=2):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in norm_landmarks]
    for a, b in connections:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], line_color, line_t)
    for pt in pts:
        cv2.circle(frame, pt, dot_r, dot_color, -1)
    return pts


# ==================== MATH HELPERS ====================
def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return abs(angle)


# ==================== DETECTION THREAD ====================
good_posture_time   = 0.0
bad_posture_time    = 0.0
last_time           = time.time()
bad_posture_cont    = 0.0
_smooth_prev_joints = None   # for EMA smoothing
_smooth_prev_head   = None
_prev_left_wrist    = None  # (x, y) normalized for hand movement direction
_prev_right_wrist   = None
_HAND_MOVE_THRESH   = 0.008  # min normalized delta to show direction
_HAND_MOVE_SMOOTH   = 0.25   # blend for smoothing movement


def _hand_move_direction(dx, dy, thresh):
    """Return direction string from normalized deltas. y down = positive dy."""
    if abs(dx) < thresh and abs(dy) < thresh:
        return "Still"
    parts = []
    if dy < -thresh:
        parts.append("Up")
    elif dy > thresh:
        parts.append("Down")
    if dx < -thresh:
        parts.append("Left")
    elif dx > thresh:
        parts.append("Right")
    return "-".join(parts) if parts else "Still"


def _blend_joints(prev, new, alpha):
    """Blend two joint dicts (each key = [x,y,z]). new * alpha + prev * (1-alpha)."""
    if prev is None or new is None or alpha >= 1.0:
        return new
    out = {}
    for k in new:
        if k in prev and len(prev[k]) == 3 and len(new[k]) == 3:
            out[k] = [
                round(alpha * new[k][0] + (1 - alpha) * prev[k][0], 4),
                round(alpha * new[k][1] + (1 - alpha) * prev[k][1], 4),
                round(alpha * new[k][2] + (1 - alpha) * prev[k][2], 4),
            ]
        else:
            out[k] = new[k]
    return out


def detection_loop():
    global good_posture_time, bad_posture_time, last_time, bad_posture_cont
    global _smooth_prev_joints, _smooth_prev_head
    global _prev_left_wrist, _prev_right_wrist

    print("  Building pose landmarker...")
    pose_det = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODELS["pose"][0]),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=CONFIG["detection_confidence"],
            min_pose_presence_confidence=CONFIG["detection_confidence"],
            min_tracking_confidence=CONFIG["tracking_confidence"],
        )
    )

    print("  Building hand landmarker...")
    hand_det = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODELS["hand"][0]),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=CONFIG["max_hands"],
            min_hand_detection_confidence=CONFIG["detection_confidence"],
            min_hand_presence_confidence=CONFIG["detection_confidence"],
            min_tracking_confidence=CONFIG["tracking_confidence"],
        )
    )

    print("  Building face landmarker...")
    face_det = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODELS["face"][0]),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=CONFIG["max_faces"],
            min_face_detection_confidence=CONFIG["detection_confidence"],
            min_face_presence_confidence=CONFIG["detection_confidence"],
            min_tracking_confidence=CONFIG["tracking_confidence"],
        )
    )

    print("  Opening camera...")
    cap = cv2.VideoCapture(CONFIG["camera_id"])
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
    print("  Camera ready. Detection running.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        if CONFIG["mirror_mode"]:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        now = time.time()
        elapsed = now - last_time
        last_time = now

        angle          = 0
        status         = "No person detected"
        good_posture   = False
        posture_type   = ""

        # ---- Pose ----
        pose_res = pose_det.detect(mp_img)
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks[0]
            shoulder = [lm[11].x * w, lm[11].y * h]
            ear      = [lm[7].x  * w, lm[7].y  * h]
            hip      = [lm[23].x * w, lm[23].y * h]
            angle    = calculate_angle(ear, shoulder, hip)

            if CONFIG["show_skeleton"]:
                # Only draw dots/lines for skeleton; skip text labels for performance.
                pts = draw_connections(frame, lm, POSE_CONNECTIONS, w, h,
                                       dot_color=(0, 255, 0), line_color=(0, 255, 0))
            for pt in [ear, shoulder, hip]:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 8, (255, 0, 0), -1)  # blue for posture points

            # Driver dots for 3D: arms (shoulders, elbows, wrists) + legs (hips, knees, ankles)
            for idx in DRIVER_JOINTS:
                if idx < len(lm):
                    px = int(lm[idx].x * w)
                    py = int(lm[idx].y * h)
                    cv2.circle(frame, (px, py), 10, (0, 255, 255), -1)   # fill: cyan
                    cv2.circle(frame, (px, py), 10, (255, 255, 0), 2)   # outline: yellow

            if CONFIG["good_posture_min_angle"] <= angle <= CONFIG["good_posture_max_angle"]:
                good_posture_time += elapsed
                bad_posture_cont   = 0
                status             = "Good Posture"
                good_posture       = True
            else:
                bad_posture_time += elapsed
                bad_posture_cont += elapsed
                posture_type = "Slouching Forward" if angle < CONFIG["good_posture_min_angle"] else "Leaning Back"
                status       = f"Bad Posture – {posture_type}"
                good_posture = False
                if bad_posture_cont >= CONFIG["bad_posture_alert_delay"] and CONFIG["enable_visual_alert"]:
                    pass  # visual alert disabled: no red border on webcam

            # Hand movement direction (Up, Down, Left, Right, etc.) from wrist motion
            if len(lm) >= 17:
                left_wrist_cur = (lm[15].x, lm[15].y)
                right_wrist_cur = (lm[16].x, lm[16].y)
                left_dir = "Still"
                right_dir = "Still"
                if _prev_left_wrist is not None:
                    dx = left_wrist_cur[0] - _prev_left_wrist[0]
                    dy = left_wrist_cur[1] - _prev_left_wrist[1]
                    left_dir = _hand_move_direction(dx, dy, _HAND_MOVE_THRESH)
                if _prev_right_wrist is not None:
                    dx = right_wrist_cur[0] - _prev_right_wrist[0]
                    dy = right_wrist_cur[1] - _prev_right_wrist[1]
                    right_dir = _hand_move_direction(dx, dy, _HAND_MOVE_THRESH)
                _prev_left_wrist = left_wrist_cur
                _prev_right_wrist = right_wrist_cur
                # Skip drawing text labels on the frame to keep streaming smooth.
        else:
            _prev_left_wrist = None
            _prev_right_wrist = None

        # ---- Face: driver dots for head (3D model)
        head_joints = None
        if CONFIG["enable_face_detection"]:
            face_res = face_det.detect(mp_img)
            if face_res.face_landmarks:
                face_lm = face_res.face_landmarks[0]
                head_joints = {}
                for name, idx in FACE_DRIVER_INDICES.items():
                    if idx < len(face_lm):
                        p = face_lm[idx]
                        head_joints[name] = [round(p.x, 4), round(p.y, 4), round(p.z, 4)]
                        px, py = int(p.x * w), int(p.y * h)
                        cv2.circle(frame, (px, py), 3, (255, 180, 100), -1)   # fill: orange (small)
                        cv2.circle(frame, (px, py), 3, (255, 200, 0), 1)     # outline

        # ---- Hands ----
        hand_landmarks_payload = None
        if CONFIG["enable_hand_detection"]:
            hand_res = hand_det.detect(mp_img)
            if hand_res.hand_landmarks:
                for hand_lm in hand_res.hand_landmarks:
                    draw_connections(frame, hand_lm, HAND_CONNECTIONS, w, h,
                                     dot_color=(0, 255, 255), line_color=(255, 0, 255),
                                     dot_r=3, line_t=2)
                hand_landmarks_payload = []
                for j in range(len(hand_res.hand_landmarks)):
                    hand_lm = hand_res.hand_landmarks[j]
                    handedness = "Left"
                    if hand_res.handedness and j < len(hand_res.handedness):
                        hd = hand_res.handedness[j]
                        if getattr(hd, "classification", None) and len(hd.classification):
                            handedness = hd.classification[0].display_name
                        elif isinstance(hd, (list, tuple)) and len(hd):
                            handedness = getattr(hd[0], "display_name", handedness)
                    hand_landmarks_payload.append({
                        "handedness": handedness,
                        "landmarks": [[round(p.x, 4), round(p.y, 4), round(p.z, 4)] for p in hand_lm],
                    })
        # Serialize pose (33 points) and driver joints (arms + legs) and head_joints (face)
        pose_landmarks = None
        driver_joints_raw = None
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks[0]
            pose_landmarks = [[round(lm[i].x, 4), round(lm[i].y, 4), round(lm[i].z, 4)] for i in range(33)]
            driver_joints_raw = {
                "left_shoulder":  [round(lm[11].x, 4), round(lm[11].y, 4), round(lm[11].z, 4)],
                "right_shoulder": [round(lm[12].x, 4), round(lm[12].y, 4), round(lm[12].z, 4)],
                "left_elbow":     [round(lm[13].x, 4), round(lm[13].y, 4), round(lm[13].z, 4)],
                "right_elbow":    [round(lm[14].x, 4), round(lm[14].y, 4), round(lm[14].z, 4)],
                "left_wrist":     [round(lm[15].x, 4), round(lm[15].y, 4), round(lm[15].z, 4)],
                "right_wrist":    [round(lm[16].x, 4), round(lm[16].y, 4), round(lm[16].z, 4)],
                "left_hip":       [round(lm[23].x, 4), round(lm[23].y, 4), round(lm[23].z, 4)],
                "right_hip":      [round(lm[24].x, 4), round(lm[24].y, 4), round(lm[24].z, 4)],
                "left_knee":      [round(lm[25].x, 4), round(lm[25].y, 4), round(lm[25].z, 4)],
                "right_knee":     [round(lm[26].x, 4), round(lm[26].y, 4), round(lm[26].z, 4)],
                "left_ankle":     [round(lm[27].x, 4), round(lm[27].y, 4), round(lm[27].z, 4)],
                "right_ankle":    [round(lm[28].x, 4), round(lm[28].y, 4), round(lm[28].z, 4)],
            }
        alpha = CONFIG.get("pose_smoothing_alpha", 0.35)
        driver_joints = _blend_joints(_smooth_prev_joints, driver_joints_raw, alpha) if driver_joints_raw else _smooth_prev_joints
        head_smoothed = _blend_joints(_smooth_prev_head, head_joints, alpha) if head_joints else _smooth_prev_head
        if driver_joints_raw is not None:
            _smooth_prev_joints = driver_joints
        if head_joints is not None:
            _smooth_prev_head = head_smoothed
        head_to_send = head_smoothed if head_smoothed is not None else _smooth_prev_head

        shared.set_posture({
            "angle":           round(angle, 1),
            "status":          status,
            "good_posture":    good_posture,
            "good_time":       int(good_posture_time),
            "bad_time":        int(bad_posture_time),
            "posture_type":    posture_type,
            "pose_landmarks":  pose_landmarks,
            "driver_joints":   driver_joints,
            "head_joints":     head_to_send,
            "hand_landmarks":  hand_landmarks_payload,
        })

        # Encode a slightly compressed JPEG to keep streaming smooth and lightweight.
        _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        shared.set_frame(jpeg.tobytes())
        # Small sleep to avoid maxing out CPU; keep it short for low latency.
        time.sleep(0.005)

    cap.release()
    pose_det.close()
    hand_det.close()
    face_det.close()


# ==================== EMBEDDED HTML ====================
# Design: reference-style (centered main text, small label above, footer legend)
PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Live Posture Detection + 3D Model</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0d0d0d;
      color: #e0e0e0;
      min-height: 100vh;
      overflow: hidden;
    }
    .container { display: flex; width: 100vw; height: 100vh; }
    .panel {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 12px;
      border-right: 1px solid #333;
    }
    .panel:last-child { border-right: none; }
    .panel.canvas-panel {
      position: relative;
      padding: 0;
      align-items: stretch;
      justify-content: stretch;
    }
    .panel.canvas-panel #canvas-3d {
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      width: 100%; height: 100%;
      display: block; background: #000;
    }
    .panel.canvas-panel .drag-overlay {
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      z-index: 1;
      cursor: grab;
      touch-action: none;
      -webkit-user-select: none;
      user-select: none;
    }
    .panel.canvas-panel .drag-overlay:active { cursor: grabbing; }
    .panel-label {
      font-size: 0.7rem; margin-bottom: 6px; color: #00ffff;
      letter-spacing: 0.05em; text-transform: uppercase;
    }
    .video-wrap {
      width: 100%; max-width: 990px;
      background: #111; overflow: hidden;
    }
    .video-wrap img { width: 100%; height: auto; display: block; }
    .status-footer {
      margin-top: 8px; font-size: 0.5rem; color: #c8c8c8;
      width: 100%; max-width: 640px; text-align: center;
    }
    .status-footer.main { max-width: 480px; }
    .status-footer.good   { color: #22c55e; }
    .status-footer.bad    { color: #ef4444; }
  </style>
  <script type="importmap">
  {"imports":{"three":"https://unpkg.com/three@0.128.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.128.0/examples/jsm/"}}
  </script>
</head>
<body>
  <div class="container">
    <div class="panel">
      <div class="panel-label">Live Detection</div>
      <div class="video-wrap">
        <img id="video" src="/video_feed" alt="Camera feed" />
      </div>
      <div id="statusBar" class="status-footer unknown">Waiting for posture data…</div>
    </div>
    <div class="panel canvas-panel">
      <canvas id="canvas-3d"></canvas>
      <div class="drag-overlay" id="drag-overlay" tabindex="0" title="Drag to rotate, scroll to zoom"></div>
    </div>
  </div>
  <script type="module">
  import * as THREE from 'three';
  import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
  import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

  const statusBar = document.getElementById('statusBar');
  const canvas = document.getElementById('canvas-3d');
  const dragOverlay = document.getElementById('drag-overlay');

  const scene  = new THREE.Scene();
  scene.background = new THREE.Color(0x0d0d0d);  // dark background
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
  camera.position.set(0, 2, 4);
  camera.lookAt(0, 0, 0);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x9bb5ce, 1);
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1;
  renderer.physicallyCorrectLights = true;

  function setSize() {
    const w = Math.max(1, dragOverlay.clientWidth || canvas.clientWidth || 1);
    const h = Math.max(1, dragOverlay.clientHeight || canvas.clientHeight || 1);
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  setSize();
  window.addEventListener('resize', setSize);
  requestAnimationFrame(() => setSize());

  const controls = new OrbitControls(camera, dragOverlay);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(0, 0, 0);
  controls.enableRotate = true;
  controls.enableZoom = true;
  controls.enablePan = true;
  dragOverlay.addEventListener('mousedown', function () { dragOverlay.focus({ preventScroll: true }); });
  dragOverlay.addEventListener('touchstart', function (e) { e.preventDefault(); }, { passive: false });
  dragOverlay.addEventListener('touchmove', function (e) { e.preventDefault(); }, { passive: false });
  dragOverlay.addEventListener('touchend', function (e) { e.preventDefault(); }, { passive: false });

  scene.add(new THREE.AmbientLight(0xffffff, 0.85));
  const dLight = new THREE.DirectionalLight(0xffffff, 1.2);
  dLight.position.set(2, 4, 3);
  scene.add(dLight);

  const floorGeo = new THREE.PlaneGeometry(5, 5);
  const floorMat = new THREE.MeshLambertMaterial({ color: 0x000000 });  // black floor
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -1;
  scene.add(floor);

  let avatarModel = null;
  let avatarSkeleton = null;
  const boneByName = {};
  const restBoneQuats = {};
  const _v1 = new THREE.Vector3();
  const _v2 = new THREE.Vector3();
  const _v3 = new THREE.Vector3();
  const _v4 = new THREE.Vector3();
  const _quat = new THREE.Quaternion();
  const _quat2 = new THREE.Quaternion();
  const _restAxisY = new THREE.Vector3(0, 1, 0);
  const _restAxisLeft = new THREE.Vector3(-1, 0, 0);   // left shoulder "out" from torso
  const _restAxisRight = new THREE.Vector3(1, 0, 0);    // right shoulder "out" from torso

  function vecFromLandmark(p) {
    return new THREE.Vector3((p[0] - 0.5) * 2, (0.5 - p[1]) * 2, -p[2]);
  }
  // Arm joints: use z = p[2] so "toward camera" (negative p[2]) = front in 3D; fixes front/back inversion.
  function vecFromLandmarkArm(p) {
    return new THREE.Vector3((p[0] - 0.5) * 2, (0.5 - p[1]) * 2, (p[2] != null ? p[2] : 0));
  }

  function webcamTo3D(p) {
    return new THREE.Vector3((p[0] - 0.5) * 2, (0.5 - p[1]) * 2, (p[2] != null ? -p[2] : 0));
  }

  function alignDirectionTo3DScreen(v) {
    v.z = -v.z;
  }

  function findBoneByNames(names) {
    const keys = Object.keys(boneByName);
    for (const n of names) {
      let best = null, bestLen = 1e9;
      for (const key of keys) {
        if ((key === n || key.indexOf(n) !== -1) && key.length < bestLen) {
          best = boneByName[key]; bestLen = key.length;
        }
      }
      if (best) return best;
    }
    return null;
  }

  // Drive one bone so it points along targetDir (world). Uses rig rest pose in parent space.
  function pointBoneAlong(bone, targetDirWorld, restLocalAxis, slerpAmount) {
    if (!bone || !bone.parent || !restBoneQuats[bone.name]) return;
    const blend = slerpAmount != null ? slerpAmount : 0.3;
    const restQ = restBoneQuats[bone.name];
    _v3.copy(restLocalAxis).applyQuaternion(restQ);
    bone.parent.getWorldQuaternion(_quat2).invert();
    _v4.copy(targetDirWorld).applyQuaternion(_quat2);
    if (_v4.lengthSq() < 1e-8) return;
    _v4.normalize();
    if (_v3.lengthSq() < 1e-8) return;
    _v3.normalize();
    if (_v3.dot(_v4) < 0) _v3.negate();
    _quat.setFromUnitVectors(_v3, _v4);
    bone.quaternion.slerp(_quat, blend);
  }

  function applyPose(data) {
    if (!avatarModel) return;
    const joints = data.driver_joints;
    const hands = data.hand_landmarks || [];
    const headJoints = data.head_joints || {};
    if (!joints) return;
    if (!avatarSkeleton && Object.keys(boneByName).length === 0) return;

    const leftShoulderBone = findBoneByNames(['LeftShoulder', 'mixamorigLeftShoulder', 'mixamorig:LeftShoulder']);
    const rightShoulderBone = findBoneByNames(['RightShoulder', 'mixamorigRightShoulder', 'mixamorig:RightShoulder']);
    const leftUpper = findBoneByNames(['LeftUpperArm', 'LeftArm', 'mixamorigLeftArm', 'mixamorig:LeftArm', 'Arm_L', 'upper_arm_l', 'Left arm', 'left_upper_arm', 'L_UpperArm', 'Shoulder_L', 'upper_arm.L', 'CC_Base_L_Upperarm']);
    const leftFore = findBoneByNames(['LeftLowerArm', 'LeftForeArm', 'mixamorigLeftForeArm', 'mixamorig:LeftForeArm', 'ForeArm_L', 'forearm_l', 'Left forearm', 'left_lower_arm', 'L_ForeArm', 'Forearm_L', 'lower_arm.L', 'CC_Base_L_Forearm']);
    const leftHand = findBoneByNames(['LeftHand', 'mixamorigLeftHand', 'mixamorig:LeftHand', 'Hand_L', 'hand_l', 'left_hand', 'Left hand', 'Wrist_L', 'wrist_l', 'CC_Base_L_Hand']);
    const rightUpper = findBoneByNames(['RightUpperArm', 'RightArm', 'mixamorigRightArm', 'mixamorig:RightArm', 'Arm_R', 'upper_arm_r', 'Right arm', 'right_upper_arm', 'R_UpperArm', 'Shoulder_R', 'upper_arm.R', 'CC_Base_R_Upperarm']);
    const rightFore = findBoneByNames(['RightLowerArm', 'RightForeArm', 'mixamorigRightForeArm', 'mixamorig:RightForeArm', 'ForeArm_R', 'forearm_r', 'Right forearm', 'right_lower_arm', 'R_ForeArm', 'Forearm_R', 'lower_arm.R', 'CC_Base_R_Forearm']);
    const rightHand = findBoneByNames(['RightHand', 'mixamorigRightHand', 'mixamorig:RightHand', 'Hand_R', 'hand_r', 'right_hand', 'Right hand', 'Wrist_R', 'wrist_r', 'CC_Base_R_Hand']);
    const spine = findBoneByNames(['Spine', 'spine', 'mixamorigSpine', 'mixamorig:Spine', 'spine_01', 'CC_Base_Spine', 'Torso', 'torso']);
    const spine1 = findBoneByNames(['Spine1', 'spine1', 'mixamorigSpine1', 'mixamorig:Spine1', 'spine_02', 'CC_Base_Spine1']);
    const chest = findBoneByNames(['Chest', 'chest', 'mixamorigChest', 'mixamorig:Chest', 'UpperChest', 'spine_03', 'CC_Base_Chest']);
    const neck = findBoneByNames(['Neck', 'neck', 'mixamorigNeck', 'mixamorig:Neck', 'CC_Base_Neck', 'neck_01']);
    const head = findBoneByNames(['Head', 'head', 'mixamorigHead', 'mixamorig:Head', 'CC_Base_Head', 'head_01']);
    const leftThigh = findBoneByNames(['LeftUpLeg', 'LeftThigh', 'mixamorigLeftUpLeg', 'mixamorig:LeftUpLeg', 'Thigh_L', 'thigh_l', 'upper_leg_l', 'CC_Base_L_Thigh', 'leg_l']);
    const leftCalf = findBoneByNames(['LeftLeg', 'LeftLowerLeg', 'mixamorigLeftLeg', 'mixamorig:LeftLeg', 'Calf_L', 'calf_l', 'lower_leg_l', 'CC_Base_L_Calf', 'shin_l']);
    const rightThigh = findBoneByNames(['RightUpLeg', 'RightThigh', 'mixamorigRightUpLeg', 'mixamorig:RightUpLeg', 'Thigh_R', 'thigh_r', 'upper_leg_r', 'CC_Base_R_Thigh', 'leg_r']);
    const rightCalf = findBoneByNames(['RightLeg', 'RightLowerLeg', 'mixamorigRightLeg', 'mixamorig:RightLeg', 'Calf_R', 'calf_r', 'lower_leg_r', 'CC_Base_R_Calf', 'shin_r']);
    const leftFoot = findBoneByNames(['LeftFoot', 'LeftToeBase', 'mixamorigLeftFoot', 'mixamorig:LeftFoot', 'Foot_L', 'foot_l', 'CC_Base_L_Foot']);
    const rightFoot = findBoneByNames(['RightFoot', 'RightToeBase', 'mixamorigRightFoot', 'mixamorig:RightFoot', 'Foot_R', 'foot_r', 'CC_Base_R_Foot']);

    const ls = joints.left_shoulder && vecFromLandmarkArm(joints.left_shoulder);
    const le = joints.left_elbow && vecFromLandmarkArm(joints.left_elbow);
    const lw = joints.left_wrist && vecFromLandmarkArm(joints.left_wrist);
    const rs = joints.right_shoulder && vecFromLandmarkArm(joints.right_shoulder);
    const re = joints.right_elbow && vecFromLandmarkArm(joints.right_elbow);
    const rw = joints.right_wrist && vecFromLandmarkArm(joints.right_wrist);
    const lh = joints.left_hip && vecFromLandmark(joints.left_hip);
    const rh = joints.right_hip && vecFromLandmark(joints.right_hip);
    const lk = joints.left_knee && vecFromLandmark(joints.left_knee);
    const rk = joints.right_knee && vecFromLandmark(joints.right_knee);
    const la = joints.left_ankle && vecFromLandmark(joints.left_ankle);
    const ra = joints.right_ankle && vecFromLandmark(joints.right_ankle);

    // For now we only drive the RIGHT side; keep the rest of the avatar static in its bind pose.
    const leftInFront = false;
    const rightInFront = true;
    const armBlend = 0.58;
    function clampToFrontHalfSpace(v) {
      if (v.z > 0) v.z = 0;
    }
    function clampDirToFront(dir) {
      if (dir.z > 0) { dir.z = 0; dir.normalize(); }
    }
    const lsClamp = ls ? ls.clone() : null;
    const leClamp = le ? le.clone() : null;
    const lwClamp = lw ? lw.clone() : null;
    const rsClamp = rs ? rs.clone() : null;
    const reClamp = re ? re.clone() : null;
    const rwClamp = rw ? rw.clone() : null;
    if (lsClamp) clampToFrontHalfSpace(lsClamp);
    if (leClamp) clampToFrontHalfSpace(leClamp);
    if (lwClamp) clampToFrontHalfSpace(lwClamp);
    if (rsClamp) clampToFrontHalfSpace(rsClamp);
    if (reClamp) clampToFrontHalfSpace(reClamp);
    if (rwClamp) clampToFrontHalfSpace(rwClamp);
    // Only drive RIGHT shoulder + arm from pose; keep left side at rest.
    // Shoulder: orient from mid-shoulder toward RIGHT shoulder only.
    const shoulderBlend = 0.52;
    if (rs && rightShoulderBone) {
      const midShoulder = ls ? new THREE.Vector3().addVectors(lsClamp || ls, rsClamp || rs).multiplyScalar(0.5)
                             : (rsClamp || rs);
      const dirRightOut = _v2.subVectors(rsClamp || rs, midShoulder).normalize();
      if (dirRightOut.lengthSq() > 1e-6) {
        clampDirToFront(dirRightOut);
        pointBoneAlong(rightShoulderBone, dirRightOut, _restAxisRight, shoulderBlend);
      } else if (restBoneQuats[rightShoulderBone.name]) {
        rightShoulderBone.quaternion.slerp(restBoneQuats[rightShoulderBone.name], 0.2);
      }
    } else if (rightShoulderBone && restBoneQuats[rightShoulderBone.name]) {
      rightShoulderBone.quaternion.slerp(restBoneQuats[rightShoulderBone.name], 0.2);
    }
    // Left shoulder/arm: always relax back to bind pose.
    if (leftShoulderBone && restBoneQuats[leftShoulderBone.name]) {
      leftShoulderBone.quaternion.slerp(restBoneQuats[leftShoulderBone.name], 0.25);
    }
    if (leftUpper && restBoneQuats[leftUpper.name]) {
      leftUpper.quaternion.slerp(restBoneQuats[leftUpper.name], 0.25);
    }
    if (leftFore && restBoneQuats[leftFore.name]) {
      leftFore.quaternion.slerp(restBoneQuats[leftFore.name], 0.25);
    }

    // Right arm: drive only shoulder → upper arm (elbow). Forearm stays in bind pose.
    if (rightInFront && rsClamp && reClamp && rightUpper) {
      _v2.subVectors(reClamp, rsClamp);
      if (_v2.lengthSq() > 1e-6) { _v2.normalize(); clampDirToFront(_v2); pointBoneAlong(rightUpper, _v2, _restAxisY, armBlend); }
    } else if (rightUpper && restBoneQuats[rightUpper.name]) {
      rightUpper.quaternion.slerp(restBoneQuats[rightUpper.name], 0.22);
    }
    // Always relax right forearm to its bind pose for now.
    if (rightFore && restBoneQuats[rightFore.name]) {
      rightFore.quaternion.slerp(restBoneQuats[rightFore.name], 0.25);
    }

    // Torso and legs: keep static (no spine/hip driving).

    // Head/neck: drive from face landmarks (nose/eyes/chin), allowing left/right
    // rotation but clamping so the head never flips backwards.
    const nose = headJoints.nose && webcamTo3D(headJoints.nose);
    const leftEye = headJoints.left_eye && webcamTo3D(headJoints.left_eye);
    const rightEye = headJoints.right_eye && webcamTo3D(headJoints.right_eye);
    const chin = headJoints.chin && webcamTo3D(headJoints.chin);
    if ((neck || head) && (nose || chin) && (leftEye || rightEye)) {
      // Use nose→chin as \"up\" direction for the head.
      const headUp = (nose && chin) ? _v1.subVectors(nose, chin).normalize() : new THREE.Vector3(0, 1, 0);
      const eyeMid = (leftEye && rightEye)
        ? new THREE.Vector3().addVectors(leftEye, rightEye).multiplyScalar(0.5)
        : (leftEye || rightEye || nose || chin);
      let headForward = (nose && eyeMid) ? _v2.subVectors(nose, eyeMid).normalize() : new THREE.Vector3(0, 0, -1);
      if (headForward.lengthSq() < 1e-6) headForward.set(0, 0, -1);
      // Map from webcam to 3D space.
      alignDirectionTo3DScreen(headUp);
      alignDirectionTo3DScreen(headForward);
      // Clamp so head never points backwards (z >= 0) and only allows moderate up/down.
      const MAX_UP_TILT = 0.6;
      const MAX_DOWN_TILT = 0.6;
      if (headForward.y > MAX_UP_TILT) headForward.y = MAX_UP_TILT;
      if (headForward.y < -MAX_DOWN_TILT) headForward.y = -MAX_DOWN_TILT;
      if (headForward.z > -0.15) headForward.z = -0.15;  // always at least slightly facing camera
      headForward.normalize();
      // Build an orthonormal basis from up + forward (with yaw still preserved).
      const headRight = new THREE.Vector3().crossVectors(headUp, headForward).normalize();
      headForward.crossVectors(headRight, headUp).normalize();
      const headMat = new THREE.Matrix4();
      headMat.set(
        headRight.x, headRight.y, headRight.z, 0,
        headUp.x, headUp.y, headUp.z, 0,
        headForward.x, headForward.y, headForward.z, 0,
        0, 0, 0, 1
      );
      _quat.setFromRotationMatrix(headMat);
      // Smaller slerp factors = smoother, less sensitive head motion.
      if (neck && restBoneQuats[neck.name]) neck.quaternion.slerp(_quat, 0.1);
      if (head && restBoneQuats[head.name]) head.quaternion.slerp(_quat, 0.5);
    } else {
      if (neck && restBoneQuats[neck.name]) neck.quaternion.slerp(restBoneQuats[neck.name], 0.22);
      if (head && restBoneQuats[head.name]) head.quaternion.slerp(restBoneQuats[head.name], 0.22);
    }

    // Keep right hand in bind pose for now (no wrist orientation).
    if (rightHand && restBoneQuats[rightHand.name]) {
      rightHand.quaternion.slerp(restBoneQuats[rightHand.name], 0.25);
    }

    if (avatarSkeleton && typeof avatarSkeleton.update === 'function') {
      avatarSkeleton.update();
    } else {
      avatarModel.traverse((o) => { if (o.isBone) o.updateMatrixWorld(true); });
    }
  }

  // --- High-quality rigged character (GLB) ---
  // Using the official three.js example Soldier model (Mixamo-style skeleton),
  // so the avatar looks good and bones are stable for driving.
  const loader = new GLTFLoader();
  const MODEL_URL = 'https://threejs.org/examples/models/gltf/Soldier.glb';
  loader.load(MODEL_URL, (gltf) => {
    const model = gltf.scene;
    avatarModel = model;

    // Reset caches
    for (const k of Object.keys(boneByName)) delete boneByName[k];
    for (const k of Object.keys(restBoneQuats)) delete restBoneQuats[k];
    avatarSkeleton = null;

    model.traverse((o) => {
      if (o.isSkinnedMesh && o.skeleton && !avatarSkeleton) {
        avatarSkeleton = o.skeleton;
        o.skeleton.bones.forEach((b) => {
          boneByName[b.name] = b;
          restBoneQuats[b.name] = b.quaternion.clone();
        });
      }
      if (o.isBone) boneByName[o.name] = o;
    });
    if (!avatarSkeleton && Object.keys(boneByName).length > 0) {
      avatarSkeleton = { bones: Object.values(boneByName), update: function() {} };
    }

    // Face the camera
    model.rotation.y = Math.PI;
    model.position.set(0, 0, 0);
    model.scale.setScalar(1);
    model.updateMatrixWorld(true);

    // Auto-scale to ~1.8m height and place on floor
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const modelH = Math.max(Math.abs(size.y), 0.01);
    const scale = Math.min(10, Math.max(0.5, 1.8 / modelH));
    model.scale.setScalar(scale);
    model.updateMatrixWorld(true);
    const box2 = new THREE.Box3().setFromObject(model);
    const center = box2.getCenter(new THREE.Vector3());
    model.position.x = -center.x;
    model.position.z = -center.z;
    model.position.y = -1 - box2.min.y;
    scene.add(model);

    console.log('GLB avatar loaded. Bones:', Object.keys(boneByName).sort().join(', '));
  }, undefined, (err) => {
    console.error('Failed to load GLB avatar:', err);
  });

  function updatePosture(angle, good, status) {
    const cls = good ? 'good' : (angle > 0 ? 'bad' : 'unknown');
    statusBar.className   = 'status-footer ' + cls;
    statusBar.textContent = status || 'No data';
  }

  let lastAngle = null;
  let smoothJoints = null;
  let smoothHead = null;
  const SMOOTH_ALPHA = 0.45;
  function blendJoints(prev, next, alpha) {
    if (!next) return prev;
    if (!prev) return next;
    const out = {};
    for (const k of Object.keys(next)) {
      if (prev[k] && prev[k].length === 3 && next[k].length === 3)
        out[k] = [prev[k][0] * (1 - alpha) + next[k][0] * alpha, prev[k][1] * (1 - alpha) + next[k][1] * alpha, prev[k][2] * (1 - alpha) + next[k][2] * alpha];
      else out[k] = next[k];
    }
    return out;
  }
  const evtSource = new EventSource('/posture_events');
  evtSource.onmessage = e => {
    try {
      let raw = (e.data || '').trim();
      if (raw.startsWith('data:')) raw = raw.slice(5).trim();
      const d = JSON.parse(raw);
      lastAngle = d.angle;
      updatePosture(d.angle, d.good_posture, d.status);
      smoothJoints = blendJoints(smoothJoints, d.driver_joints, SMOOTH_ALPHA);
      smoothHead = blendJoints(smoothHead, d.head_joints, SMOOTH_ALPHA);
      // Connect dots: drive 3D model same-to-same from detection (joints + head).
      applyPose({ ...d, driver_joints: smoothJoints, head_joints: smoothHead });
    } catch (_) {}
  };
  evtSource.onerror = () => updatePosture(lastAngle, false, 'Reconnecting…');

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
  </script>
</body>
</html>
"""

# ==================== FLASK ====================
app = Flask(__name__)


@app.route("/")
def index():
    return Response(PAGE_HTML, mimetype="text/html")


@app.route("/video_feed")
def video_feed():
    def generate():
        boundary = b"frame"
        while True:
            jpeg = shared.get_frame()
            if jpeg:
                yield (b"--" + boundary + b"\r\n"
                       b"Content-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")
            else:
                yield (b"--" + boundary + b"\r\n"
                       b"Content-Type: image/jpeg\r\nContent-Length: 0\r\n\r\n\r\n")
            # Keep this small so the stream can update as soon as new frames are available.
            time.sleep(0.01)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/posture_events")
def posture_events():
    def generate():
        while True:
            yield "data: " + json.dumps(shared.get_posture()) + "\n\n"
            time.sleep(0.05)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


# ==================== MAIN ====================
if __name__ == "__main__":
    print("=== Posture Detection (MediaPipe 0.10+ Tasks API) ===")
    print("Checking / downloading models...")
    download_models()
    print("Starting detection thread...")
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    time.sleep(2.0)  # let models load
    print("\nOpen in browser: http://127.0.0.1:5000\n")
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)