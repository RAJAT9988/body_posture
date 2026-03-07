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

from flask import Flask, Response
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
    "detection_confidence": 0.5,
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
    "mirror_mode": True,
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


def detection_loop():
    global good_posture_time, bad_posture_time, last_time, bad_posture_cont

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
                draw_connections(frame, lm, POSE_CONNECTIONS, w, h,
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
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)

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
        if CONFIG["enable_hand_detection"]:
            hand_res = hand_det.detect(mp_img)
            if hand_res.hand_landmarks:
                for hand_lm in hand_res.hand_landmarks:
                    draw_connections(frame, hand_lm, HAND_CONNECTIONS, w, h,
                                     dot_color=(0, 255, 255), line_color=(255, 0, 255),
                                     dot_r=3, line_t=2)
        # Serialize pose (33 points) and driver joints (arms + legs) and head_joints (face)
        pose_landmarks = None
        driver_joints = None
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks[0]
            pose_landmarks = [[round(lm[i].x, 4), round(lm[i].y, 4), round(lm[i].z, 4)] for i in range(33)]
            driver_joints = {
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

        shared.set_posture({
            "angle":           round(angle, 1),
            "status":          status,
            "good_posture":    good_posture,
            "good_time":       int(good_posture_time),
            "bad_time":        int(bad_posture_time),
            "posture_type":    posture_type,
            "pose_landmarks":  pose_landmarks,
            "driver_joints":   driver_joints,
            "head_joints":     head_joints,
        })

        _, jpeg = cv2.imencode(".jpg", frame)
        shared.set_frame(jpeg.tobytes())
        time.sleep(0.02)

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
    .panel.canvas-panel .panel-label {
      position: absolute; top: 8px; left: 50%; transform: translateX(-50%);
      z-index: 2; margin: 0;
    }
    .panel.canvas-panel #canvas-3d {
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      width: 100%; height: 100%;
      display: block; background: #000;
    }
    .panel.canvas-panel #statusBar3d {
      position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%);
      z-index: 2; margin: 0;
    }
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
      <div class="panel-label">3D Posture</div>
      <canvas id="canvas-3d"></canvas>
      <div id="statusBar3d" class="status-footer main unknown">Spine angle: -- °</div>
    </div>
  </div>
  <script type="module">
  import * as THREE from 'three';
  import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

  const statusBar   = document.getElementById('statusBar');
  const statusBar3d = document.getElementById('statusBar3d');
  const canvas      = document.getElementById('canvas-3d');

  const scene  = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
  camera.position.set(0, 2, 4);
  camera.lookAt(0, 0, 0);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  function setSize() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w;
    canvas.height = h;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  setSize();
  window.addEventListener('resize', setSize);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(0, 0, 0);

  scene.add(new THREE.AmbientLight(0x404040, 0.6));
  const dLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dLight.position.set(2, 4, 3);
  scene.add(dLight);

  const grid = new THREE.GridHelper(8, 16, 0xb0b0b0, 0x909090);
  grid.material.transparent = true;
  grid.material.opacity = 0.55;
  grid.position.y = -1;
  scene.add(grid);

  function updatePosture(angle, good, status) {
    const cls = good ? 'good' : (angle > 0 ? 'bad' : 'unknown');
    statusBar.className   = 'status-footer ' + cls;
    statusBar.textContent = status || 'No data';
    statusBar3d.className = 'status-footer main ' + cls;
    statusBar3d.textContent = 'Spine angle: ' + (angle != null ? angle + ' °' : '--');
  }

  let lastAngle = null;
  const evtSource = new EventSource('/posture_events');
  evtSource.onmessage = e => {
    try {
      const d = JSON.parse(e.data);
      lastAngle = d.angle;
      updatePosture(d.angle, d.good_posture, d.status);
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
            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.route("/posture_events")
def posture_events():
    def generate():
        while True:
            yield "data: " + json.dumps(shared.get_posture()) + "\n\n"
            time.sleep(0.1)

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