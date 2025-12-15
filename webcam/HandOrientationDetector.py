import argparse
import cv2
import json
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import socket
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


SMOOTHING_ALPHA = 0.3
FINGER_ORDER = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
ORIENTATION_AXES = ["Pitch", "Yaw", "Roll"]
FINGER_TIP_INDEX = {
    "Thumb": 4,
    "Index": 8,
    "Middle": 12,
    "Ring": 16,
    "Pinky": 20,
}


# ---------------------------------------------------------
# UTIL : Dessiner les landmarks (version simplifiée)
# ---------------------------------------------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = rgb_image.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            landmark_list = landmark_pb2.NormalizedLandmarkList(
                landmark=[
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in hand_landmarks
                ]
            )

            mp_drawing.draw_landmarks(
                annotated_image,
                landmark_list,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
    return annotated_image


def compute_palm_normal(hand_landmarks):
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z], dtype=np.float32)
    index_mcp = np.array([hand_landmarks[5].x, hand_landmarks[5].y, hand_landmarks[5].z], dtype=np.float32)
    pinky_mcp = np.array([hand_landmarks[17].x, hand_landmarks[17].y, hand_landmarks[17].z], dtype=np.float32)

    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)

    if norm == 0:
        return None

    return normal / norm


def smooth_vector(previous, current, alpha=SMOOTHING_ALPHA):
    if current is None:
        return None

    if previous is None:
        previous = current

    smoothed = previous * (1.0 - alpha) + current * alpha
    norm = np.linalg.norm(smoothed)
    if norm == 0:
        return current

    return smoothed / norm


def smooth_scalar(previous, current, alpha=SMOOTHING_ALPHA):
    if current is None:
        return previous

    if previous is None:
        return current

    return previous * (1.0 - alpha) + current * alpha


def compute_hand_axes(hand_landmarks):
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z], dtype=np.float32)
    middle_mcp = np.array([hand_landmarks[9].x, hand_landmarks[9].y, hand_landmarks[9].z], dtype=np.float32)

    forward = compute_palm_normal(hand_landmarks)
    if forward is None:
        return None

    up = middle_mcp - wrist
    up_norm = np.linalg.norm(up)
    if up_norm == 0:
        return None
    up = up / up_norm

    right = np.cross(up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm == 0:
        return None
    right = right / right_norm

    # Re-orthogonalize up using the computed right and forward to avoid drift.
    up = np.cross(forward, right)
    up_norm = np.linalg.norm(up)
    if up_norm == 0:
        return None
    up = up / up_norm

    return {
        "forward": forward,
        "up": up,
        "right": right,
    }


def draw_palm_normal(image, hand_landmarks, normal, label):
    h, w, _ = image.shape
    wrist = hand_landmarks[0]

    start = (int(wrist.x * w), int(wrist.y * h))
    scale = 120
    end = (
        int(start[0] + normal[0] * scale),
        int(start[1] + normal[1] * scale),
    )

    cv2.arrowedLine(image, start, end, (0, 255, 255), 2, tipLength=0.25)
    text = f"{label} n=({normal[0]:+.2f},{normal[1]:+.2f},{normal[2]:+.2f})"
    text_origin = (start[0] + 10, start[1] - 10)
    cv2.putText(image, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def draw_hand_axes(image, hand_landmarks, axes):
    colors = {
        "forward": (0, 0, 255),
        "up": (0, 255, 0),
        "right": (255, 0, 0),
    }

    h, w, _ = image.shape
    wrist = hand_landmarks[0]
    origin = (int(wrist.x * w), int(wrist.y * h))
    scale = 100

    for name, vec in axes.items():
        end = (
            int(origin[0] + vec[0] * scale),
            int(origin[1] + vec[1] * scale),
        )
        cv2.arrowedLine(image, origin, end, colors[name], 2, tipLength=0.25)
        cv2.putText(
            image,
            name[0].upper(),
            (end[0] + 5, end[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colors[name],
            1,
            cv2.LINE_AA,
        )


def build_udp_payload(curls, horizontal_values, orientation_tracker):
    payload = {
        "handedness": "Right",
        "fingers": {},
        "wrist": {
            "calibrated": orientation_tracker.is_calibrated,
            "pitch": None,
            "yaw": None,
            "roll": None,
            "quaternion": None,
        },
    }

    curls = curls or {}
    horizontal_values = horizontal_values or {}

    for finger in FINGER_ORDER:
        payload["fingers"][finger] = {
            "curl": float(np.clip(curls.get(finger, 0.0), 0.0, 1.0)) if finger in curls else None,
            "horizontal": float(horizontal_values.get(finger, 0.0)) if finger in horizontal_values else None,
        }

    angles = orientation_tracker.get_latest_angles() if orientation_tracker else None
    if angles:
        payload["wrist"].update(
            {
                "pitch": float(angles.get("Pitch", 0.0)),
                "yaw": float(angles.get("Yaw", 0.0)),
                "roll": float(-angles.get("Roll", 0.0)),
            }
        )
    
    # Ajouter quaternion (pas de gimbal lock)
    quat = orientation_tracker.get_latest_quaternion() if orientation_tracker else None
    if quat:
        payload["wrist"]["quaternion"] = quat

    return payload


def send_udp_metrics(curls, horizontal_values, orientation_tracker):
    if udp_socket is None:
        return

    payload = build_udp_payload(curls, horizontal_values, orientation_tracker)
    try:
        message = json.dumps(payload).encode("utf-8")
        udp_socket.sendto(message, (UDP_IP, UDP_PORT))
    except OSError as exc:
        print(f"⚠️ Erreur d'envoi UDP: {exc}")


class WristGizmoWindow:
    def __init__(self):
        self.enabled = True
        self.fig = None
        self.ax = None

        try:
            plt.ion()
            self.fig = plt.figure("Wrist Gizmo")
            self.ax = self.fig.add_subplot(111, projection="3d")
            self._configure_axes()
            plt.show(block=False)
        except Exception as exc:
            self.enabled = False
            self.fig = None
            self.ax = None
            print(f"⚠️ Impossible d'ouvrir la fenêtre gizmo: {exc}")

    def _configure_axes(self):
        if self.ax is None:
            return

        limit = 1.2
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-limit, limit)
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.view_init(elev=25, azim=-45)
        self.ax.grid(True)

    def update(self, rotation_matrix, angles=None):
        if not self.enabled or self.fig is None or self.ax is None:
            return

        if not plt.fignum_exists(self.fig.number):
            self.enabled = False
            return

        self.ax.cla()
        self._configure_axes()

        if rotation_matrix is None:
            rotation_matrix = np.eye(3, dtype=np.float32)

        axis_vectors = [
            ("X", "r", rotation_matrix @ np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            ("Y", "g", rotation_matrix @ np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            ("Z", "b", rotation_matrix @ np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ]

        origin = np.zeros(3, dtype=np.float32)
        for label, color, vec in axis_vectors:
            self.ax.plot([origin[0], vec[0]], [origin[1], vec[1]], [origin[2], vec[2]], color=color, linewidth=2)
            self.ax.text(vec[0], vec[1], vec[2], label, color=color)

        if angles:
            display = " | ".join(
                f"{axis}: {np.rad2deg(val):+.1f}°" for axis, val in angles.items()
            )
            self.ax.text2D(0.05, 0.92, display, transform=self.ax.transAxes)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)


class HorizontalFingerTracker:
    def __init__(self, min_span=0.02):
        self.min_span = min_span
        self.baseline = {}
        self.span_left = {}
        self.span_right = {}
        self.values = {}
        self.is_calibrated = False

    def reset(self):
        self.baseline.clear()
        self.span_left.clear()
        self.span_right.clear()
        self.values.clear()
        self.is_calibrated = False

    def calibrate(self, hand_landmarks, axes):
        if axes is None or "right" not in axes:
            return False

        wrist = landmark_to_np(hand_landmarks[0])
        right_axis = axes["right"]

        for finger, tip_idx in FINGER_TIP_INDEX.items():
            tip = landmark_to_np(hand_landmarks[tip_idx])
            projection = float(np.dot(tip - wrist, right_axis))
            self.baseline[finger] = projection
            self.span_left[finger] = self.min_span
            self.span_right[finger] = self.min_span
            self.values[finger] = 0.0

        self.is_calibrated = True
        return True

    def update(self, hand_landmarks, axes):
        if not self.is_calibrated or axes is None or "right" not in axes:
            return None

        wrist = landmark_to_np(hand_landmarks[0])
        right_axis = axes["right"]
        updated = {}

        for finger, tip_idx in FINGER_TIP_INDEX.items():
            baseline = self.baseline.get(finger)
            if baseline is None:
                continue

            tip = landmark_to_np(hand_landmarks[tip_idx])
            projection = float(np.dot(tip - wrist, right_axis))
            delta = projection - baseline

            if delta < 0:
                self.span_left[finger] = max(self.span_left.get(finger, self.min_span), abs(delta))
                denom = self.span_left[finger]
            else:
                self.span_right[finger] = max(self.span_right.get(finger, self.min_span), abs(delta))
                denom = self.span_right[finger]

            denom = max(denom, self.min_span)
            normalized = float(np.clip(delta / denom, -1.0, 1.0))
            smoothed = smooth_scalar(self.values.get(finger), normalized)
            self.values[finger] = smoothed
            updated[finger] = smoothed

        return updated if updated else None


def axes_to_matrix(axes):
    if axes is None:
        return None

    required = ["right", "up", "forward"]
    if not all(axis in axes and axes[axis] is not None for axis in required):
        return None

    return np.column_stack((axes["right"], axes["up"], axes["forward"]))


def rotation_matrix_to_euler_zyx(matrix):
    # Returns pitch (around Y), yaw (around Z), roll (around X)
    if matrix is None:
        return None

    m20 = matrix[2, 0]
    if abs(m20) < 1.0:
        pitch = np.arcsin(-m20)
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        pitch = np.pi / 2 if m20 <= -1.0 else -np.pi / 2
        roll = 0.0
        yaw = np.arctan2(-matrix[0, 1], matrix[1, 1])

    return {
        "Pitch": pitch,
        "Yaw": yaw,
        "Roll": roll,
    }


def rotation_matrix_to_quaternion(matrix):
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w) for Unity."""
    if matrix is None:
        return None
    
    # Shepperd's method - numerically stable
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    
    # Normaliser
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm > 0:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # Conversion vers système Unity (main gauche): inverser x et z
    return {"x": float(-x), "y": float(y), "z": float(-z), "w": float(w)}


class WristOrientationTracker:
    def __init__(self, min_span_deg=15.0):
        self.min_span = np.deg2rad(min_span_deg)
        self.baseline_matrix = None
        self.span_neg = {axis: self.min_span for axis in ORIENTATION_AXES}
        self.span_pos = {axis: self.min_span for axis in ORIENTATION_AXES}
        self.values = {axis: 0.0 for axis in ORIENTATION_AXES}
        self.is_calibrated = False
        self.latest_matrix = np.eye(3, dtype=np.float32)
        self.latest_angles = {axis: 0.0 for axis in ORIENTATION_AXES}

    def reset(self):
        self.baseline_matrix = None
        self.span_neg = {axis: self.min_span for axis in ORIENTATION_AXES}
        self.span_pos = {axis: self.min_span for axis in ORIENTATION_AXES}
        self.values = {axis: 0.0 for axis in ORIENTATION_AXES}
        self.is_calibrated = False
        self.latest_matrix = np.eye(3, dtype=np.float32)
        self.latest_angles = {axis: 0.0 for axis in ORIENTATION_AXES}

    def calibrate(self, axes):
        matrix = axes_to_matrix(axes)
        if matrix is None:
            return False

        self.baseline_matrix = matrix
        self.span_neg = {axis: self.min_span for axis in ORIENTATION_AXES}
        self.span_pos = {axis: self.min_span for axis in ORIENTATION_AXES}
        self.values = {axis: 0.0 for axis in ORIENTATION_AXES}
        self.is_calibrated = True
        self.latest_matrix = np.eye(3, dtype=np.float32)
        self.latest_angles = {axis: 0.0 for axis in ORIENTATION_AXES}
        return True

    def update(self, axes):
        if not self.is_calibrated or self.baseline_matrix is None:
            return None

        current_matrix = axes_to_matrix(axes)
        if current_matrix is None:
            return None

        delta = current_matrix @ self.baseline_matrix.T
        angles = rotation_matrix_to_euler_zyx(delta)
        if angles is None:
            return None

        self.latest_matrix = delta
        self.latest_angles = angles

        updated = {}
        for axis, angle in angles.items():
            span = self.span_pos if angle >= 0 else self.span_neg
            key = axis
            span[key] = max(span.get(key, self.min_span), abs(angle))
            denom = span[key]
            denom = max(denom, self.min_span)
            normalized = float(np.clip(angle / denom, -1.0, 1.0))
            smoothed = smooth_scalar(self.values.get(axis), normalized)
            self.values[axis] = smoothed
            updated[axis] = smoothed

        return updated if updated else None

    def get_latest_matrix(self):
        return self.latest_matrix

    def get_latest_angles(self):
        return self.latest_angles

    def get_latest_quaternion(self):
        return rotation_matrix_to_quaternion(self.latest_matrix)


def landmark_to_np(landmark):
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)


def angle_between(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return None

    dot = np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), -1.0, 1.0)
    return np.arccos(dot)


def compute_finger_curls(hand_landmarks):
    configs = {
        "Thumb": (1, 2, 4),
        "Index": (5, 6, 8),
        "Middle": (9, 10, 12),
        "Ring": (13, 14, 16),
        "Pinky": (17, 18, 20),
    }

    curls = {}
    for finger, (a_idx, b_idx, c_idx) in configs.items():
        a = landmark_to_np(hand_landmarks[a_idx])
        b = landmark_to_np(hand_landmarks[b_idx])
        c = landmark_to_np(hand_landmarks[c_idx])

        vec_ab = a - b
        vec_cb = c - b
        angle = angle_between(vec_ab, vec_cb)
        if angle is None:
            continue

        curl = 1.0 - min(angle / np.pi, 1.0)
        curls[finger] = curl

    return curls


def draw_finger_curl_ui(image, curls):
    bar_width = 200
    bar_height = 18
    margin_left = 30
    margin_top = 30
    spacing = 12

    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (margin_left - 20, margin_top - 20),
        (margin_left + bar_width + 40, margin_top + (bar_height + spacing) * len(FINGER_ORDER)),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)

    for idx, finger in enumerate(FINGER_ORDER):
        value = np.clip(curls.get(finger, 0.0), 0.0, 1.0)
        top = margin_top + idx * (bar_height + spacing)
        start = (margin_left, top)
        end = (margin_left + bar_width, top + bar_height)
        fill_end = (margin_left + int(bar_width * value), top + bar_height)

        cv2.rectangle(image, start, end, (80, 80, 80), 1)
        cv2.rectangle(image, start, fill_end, (0, 165, 255), -1)
        label = f"{finger}: {value:.2f}"
        cv2.putText(
            image,
            label,
            (end[0] + 15, end[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_horizontal_bars(image, horizontal_values, is_calibrated):
    bar_width = 220
    bar_height = 16
    spacing = 10
    margin_top = 30
    margin_right = 30

    h, w, _ = image.shape
    start_x = w - margin_right - bar_width

    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (start_x - 20, margin_top - 20),
        (w - margin_right + 20, margin_top + (bar_height + spacing) * len(FINGER_ORDER)),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)

    for idx, finger in enumerate(FINGER_ORDER):
        value = float(horizontal_values.get(finger, 0.0)) if horizontal_values else 0.0
        top = margin_top + idx * (bar_height + spacing)
        bottom = top + bar_height
        left = start_x
        right = start_x + bar_width
        center = left + bar_width // 2

        cv2.rectangle(image, (left, top), (right, bottom), (80, 80, 80), 1)
        cv2.line(image, (center, top), (center, bottom), (120, 120, 120), 1)

        half_width = bar_width // 2
        pixels = int(value * half_width)
        if pixels > 0:
            cv2.rectangle(image, (center, top), (center + pixels, bottom), (0, 200, 100), -1)
        elif pixels < 0:
            cv2.rectangle(image, (center + pixels, top), (center, bottom), (0, 140, 255), -1)

        cv2.putText(
            image,
            f"{finger}: {value:+.2f}",
            (left - 115, bottom - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    prompt = "Press 'C' to calibrate" if not is_calibrated else "Horizontal slide (±1)"
    cv2.putText(
        image,
        prompt,
        (start_x - 10, margin_top - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_orientation_bars(image, orientation_values, is_calibrated):
    bar_width = 220
    bar_height = 18
    spacing = 12
    margin_bottom = 30
    margin_right = 30

    h, w, _ = image.shape
    start_x = w - margin_right - bar_width
    base_y = h - margin_bottom - (bar_height + spacing) * len(ORIENTATION_AXES)

    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (start_x - 20, base_y - 20),
        (w - margin_right + 20, h - margin_bottom + 10),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)

    half_width = bar_width // 2
    for idx, axis in enumerate(ORIENTATION_AXES):
        value = float(orientation_values.get(axis, 0.0)) if orientation_values else 0.0
        top = base_y + idx * (bar_height + spacing)
        bottom = top + bar_height
        left = start_x
        right = start_x + bar_width
        center = left + half_width

        cv2.rectangle(image, (left, top), (right, bottom), (80, 80, 80), 1)
        cv2.line(image, (center, top), (center, bottom), (120, 120, 120), 1)

        pixels = int(value * half_width)
        if pixels > 0:
            cv2.rectangle(image, (center, top), (center + pixels, bottom), (120, 220, 120), -1)
        elif pixels < 0:
            cv2.rectangle(image, (center + pixels, top), (center, bottom), (120, 150, 255), -1)

        cv2.putText(
            image,
            f"{axis}: {value:+.2f}",
            (left - 110, bottom - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    prompt = "Press 'C' to calibrate" if not is_calibrated else "Pitch/Yaw/Roll (±1)"
    cv2.putText(
        image,
        prompt,
        (start_x - 10, base_y - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
# ---------------------------------------------------------
# MAIN : Webcam + MediaPipe Tasks HandLandmarker
# ---------------------------------------------------------
def main(show_3d_view=False):
    # STEP 1: Create HandLandmarker
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    detector = vision.HandLandmarker.create_from_options(options)

    smoothed_normals = {}
    smoothed_axes = {}
    smoothed_curls = {}
    horizontal_tracker = HorizontalFingerTracker()
    orientation_tracker = WristOrientationTracker()
    calibration_requested = False
    gizmo_window = WristGizmoWindow() if show_3d_view else None

    # STEP 2: OpenCV webcam loop
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Impossible d’ouvrir la webcam.")
        return

    print("🎉 HandOrientationDetector démarré. Appuie sur Q pour quitter.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame non lue.")
                break

            # Convertir frame en mp.Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # STEP 3: Détection
            detection_result = detector.detect(mp_image)

            # STEP 4: Dessin
            annotated = draw_landmarks_on_image(rgb_frame, detection_result)

            right_index = None
            handedness_list = detection_result.handedness or []
            for idx, handedness in enumerate(handedness_list):
                if handedness and handedness[0].category_name.lower() == "right":
                    right_index = idx
                    break

            if (
                detection_result.hand_landmarks
                and right_index is not None
                and right_index < len(detection_result.hand_landmarks)
            ):
                hand_landmarks = detection_result.hand_landmarks[right_index]
                label = "Right"
                if handedness_list and handedness_list[right_index]:
                    label = handedness_list[right_index][0].category_name

                normal = compute_palm_normal(hand_landmarks)
                if normal is not None:
                    smoothed_normal = smooth_vector(smoothed_normals.get(label), normal)
                    smoothed_normals[label] = smoothed_normal
                    draw_palm_normal(annotated, hand_landmarks, smoothed_normal, label)

                axes = compute_hand_axes(hand_landmarks)
                axes_for_tracker = None
                if axes:
                    prev_axes = smoothed_axes.get(label, {})
                    blended_axes = {}
                    smoothed_axes[label] = {}

                    for axis_name, axis_vec in axes.items():
                        blended = smooth_vector(prev_axes.get(axis_name), axis_vec)
                        if blended is None:
                            continue
                        smoothed_axes[label][axis_name] = blended
                        blended_axes[axis_name] = blended

                    if blended_axes:
                        draw_hand_axes(annotated, hand_landmarks, blended_axes)
                        axes_for_tracker = blended_axes
                    else:
                        axes_for_tracker = axes
                else:
                    axes_for_tracker = None

                if calibration_requested and axes_for_tracker:
                    horiz_ok = horizontal_tracker.calibrate(hand_landmarks, axes_for_tracker)
                    orient_ok = orientation_tracker.calibrate(axes_for_tracker)
                    if horiz_ok or orient_ok:
                        print("✅ Calibration enregistrée.")
                        calibration_requested = False
                        if gizmo_window is not None:
                            gizmo_window.close()
                            gizmo_window = None

                curls = compute_finger_curls(hand_landmarks)
                if curls:
                    for finger, value in curls.items():
                        smoothed_curls[finger] = smooth_scalar(smoothed_curls.get(finger), value)

                    draw_finger_curl_ui(annotated, smoothed_curls)

                horizontal_tracker.update(hand_landmarks, axes_for_tracker)
                orientation_tracker.update(axes_for_tracker)

                send_udp_metrics(smoothed_curls, horizontal_tracker.values, orientation_tracker)

            draw_horizontal_bars(annotated, horizontal_tracker.values, horizontal_tracker.is_calibrated)
            draw_orientation_bars(annotated, orientation_tracker.values, orientation_tracker.is_calibrated)

            if gizmo_window is not None:
                gizmo_window.update(
                    orientation_tracker.get_latest_matrix(),
                    orientation_tracker.get_latest_angles() if orientation_tracker.is_calibrated else None,
                )

            # STEP 5: Affichage
            cv2.imshow("Hand Orientation Detector", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                calibration_requested = True
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if gizmo_window is not None:
            gizmo_window.close()
        udp_socket.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Orientation Detector")
    parser.add_argument(
        "--show-3d-view",
        action="store_true",
        help="Affiche la fenêtre de visu 3D (désactivée après la calibration)",
    )
    args = parser.parse_args()
    main(show_3d_view=args.show_3d_view)
