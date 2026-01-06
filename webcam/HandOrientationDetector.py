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


def build_udp_payload(curls, splay_values, orientation_tracker):
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
    splay_values = splay_values or {}

    for finger in FINGER_ORDER:
        payload["fingers"][finger] = {
            "curl": float(np.clip(curls.get(finger, 0.0), 0.0, 1.0)) if finger in curls else None,
            "splay": float(np.clip(splay_values.get(finger, 0.0), 0.0, 1.0)) if finger in splay_values else None,
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


def send_udp_metrics(curls, splay_values, orientation_tracker):
    if udp_socket is None:
        return

    payload = build_udp_payload(curls, splay_values, orientation_tracker)
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


def compute_finger_splay(hand_landmarks, axes=None):
    """
    Calcule l'écartement (splay) entre doigts adjacents en utilisant des ANGLES.
    
    Méthode: On mesure l'angle entre les vecteurs MCP→PIP de doigts adjacents,
    projeté dans le plan de la paume. Les angles sont invariants à la distance caméra.
    
    Retourne des valeurs de 0 (doigts collés) à 1 (doigts très écartés).
    """
    # MCP et PIP indices pour les 4 doigts
    mcp_indices = {"Index": 5, "Middle": 9, "Ring": 13, "Pinky": 17}
    pip_indices = {"Index": 6, "Middle": 10, "Ring": 14, "Pinky": 18}
    
    # Indices du pouce: CMC=1, MCP=2, IP=3, TIP=4
    THUMB_CMC = 1
    THUMB_MCP = 2
    
    pairs = [
        ("Index", "Middle"),
        ("Middle", "Ring"),
        ("Ring", "Pinky"),
    ]
    
    # Angles de référence (en radians) - déterminés empiriquement
    # Doigts collés ~ 5-8°, doigts écartés ~ 20-35°
    MIN_ANGLE_DEG = 5.0
    MAX_ANGLE_DEG = 30.0
    min_angle = np.deg2rad(MIN_ANGLE_DEG)
    max_angle = np.deg2rad(MAX_ANGLE_DEG)
    
    # Pour le pouce, la plage est plus grande (il peut s'écarter beaucoup plus)
    THUMB_MIN_ANGLE_DEG = 20.0   # Pouce collé à l'index
    THUMB_MAX_ANGLE_DEG = 80.0   # Pouce très écarté
    thumb_min_angle = np.deg2rad(THUMB_MIN_ANGLE_DEG)
    thumb_max_angle = np.deg2rad(THUMB_MAX_ANGLE_DEG)
    
    splay = {}
    
    # --- Splay du pouce par rapport à l'index ---
    # Vecteur du pouce: CMC → MCP (direction de la base du pouce)
    thumb_cmc = landmark_to_np(hand_landmarks[THUMB_CMC])
    thumb_mcp = landmark_to_np(hand_landmarks[THUMB_MCP])
    vec_thumb = thumb_mcp - thumb_cmc
    
    # Vecteur de l'index: MCP → PIP
    index_mcp = landmark_to_np(hand_landmarks[mcp_indices["Index"]])
    index_pip = landmark_to_np(hand_landmarks[pip_indices["Index"]])
    vec_index = index_pip - index_mcp
    
    # Projeter dans le plan de la paume si disponible
    if axes is not None and "forward" in axes:
        forward = axes["forward"]
        vec_thumb_proj = vec_thumb - np.dot(vec_thumb, forward) * forward
        vec_index_proj = vec_index - np.dot(vec_index, forward) * forward
    else:
        vec_thumb_proj = vec_thumb
        vec_index_proj = vec_index
    
    thumb_angle = angle_between(vec_thumb_proj, vec_index_proj)
    if thumb_angle is not None:
        thumb_normalized = float(np.clip(
            (thumb_angle - thumb_min_angle) / (thumb_max_angle - thumb_min_angle), 
            0.0, 1.0
        ))
        splay["Thumb"] = thumb_normalized
    
    # --- Splay des autres doigts ---
    for finger1, finger2 in pairs:
        # Vecteur du doigt 1: MCP → PIP
        mcp1 = landmark_to_np(hand_landmarks[mcp_indices[finger1]])
        pip1 = landmark_to_np(hand_landmarks[pip_indices[finger1]])
        vec1 = pip1 - mcp1
        
        # Vecteur du doigt 2: MCP → PIP
        mcp2 = landmark_to_np(hand_landmarks[mcp_indices[finger2]])
        pip2 = landmark_to_np(hand_landmarks[pip_indices[finger2]])
        vec2 = pip2 - mcp2
        
        # Si on a les axes de la paume, projeter les vecteurs dans le plan de la paume
        # Cela rend la mesure plus stable car on ignore la composante de curl
        if axes is not None and "forward" in axes:
            forward = axes["forward"]  # Normale à la paume
            # Projeter vec1 et vec2 dans le plan perpendiculaire à forward
            vec1 = vec1 - np.dot(vec1, forward) * forward
            vec2 = vec2 - np.dot(vec2, forward) * forward
        
        # Calculer l'angle entre les deux vecteurs
        angle = angle_between(vec1, vec2)
        if angle is None:
            continue
        
        # Normaliser l'angle entre 0 et 1
        normalized = float(np.clip((angle - min_angle) / (max_angle - min_angle), 0.0, 1.0))
        splay[finger1] = normalized
    
    # Le Pinky n'a pas de voisin à droite, on copie la valeur Ring-Pinky
    splay["Pinky"] = splay.get("Ring", 0.0)
    
    return splay


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


def draw_splay_bars(image, splay_values):
    """Affiche les barres de splay (écartement entre doigts) en bas à gauche."""
    bar_width = 200
    bar_height = 16
    spacing = 10
    margin_left = 30
    margin_bottom = 30

    h, w, _ = image.shape
    # 4 paires de doigts: Thumb-Index, Index-Middle, Middle-Ring, Ring-Pinky
    splay_pairs = ["Thumb", "Index", "Middle", "Ring"]
    pair_labels = ["T-I", "I-M", "M-R", "R-P"]
    
    base_y = h - margin_bottom - (bar_height + spacing) * len(splay_pairs)

    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (margin_left - 20, base_y - 25),
        (margin_left + bar_width + 60, h - margin_bottom + 10),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)

    # Titre
    cv2.putText(
        image,
        "Splay (0-1)",
        (margin_left, base_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    for idx, (finger, label) in enumerate(zip(splay_pairs, pair_labels)):
        value = float(np.clip(splay_values.get(finger, 0.0), 0.0, 1.0)) if splay_values else 0.0
        top = base_y + idx * (bar_height + spacing)
        bottom = top + bar_height
        left = margin_left
        right = margin_left + bar_width

        cv2.rectangle(image, (left, top), (right, bottom), (80, 80, 80), 1)
        fill_width = int(bar_width * value)
        if fill_width > 0:
            cv2.rectangle(image, (left, top), (left + fill_width, bottom), (255, 180, 0), -1)

        cv2.putText(
            image,
            f"{label}: {value:.2f}",
            (right + 10, bottom - 3),
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
# MAIN : Webcam + MediaPipe Tasks HandLandmarker (THREADED)
# ---------------------------------------------------------
def main(show_3d_view=False, use_zed=False, headless=False, benchmark=False):
    import time
    import pyzed.sl as sl
    from threading import Thread, Lock
    from queue import Queue
    
    # Benchmark timers
    bench_capture = 0.0
    bench_detect = 0.0
    bench_process = 0.0
    bench_draw = 0.0
    bench_display = 0.0
    bench_count = 0
    
    # STEP 1: Create HandLandmarker
    try:
        base_options = python.BaseOptions(
            model_asset_path="hand_landmarker.task",
            delegate=python.BaseOptions.Delegate.GPU
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        detector = vision.HandLandmarker.create_from_options(options)
        print("✅ MediaPipe GPU delegate activé")
    except Exception as e:
        print(f"⚠️ GPU non disponible, fallback CPU: {e}")
        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        detector = vision.HandLandmarker.create_from_options(options)

    smoothed_normals = {}
    smoothed_axes = {}
    smoothed_curls = {}
    smoothed_splay = {}
    orientation_tracker = WristOrientationTracker()
    calibration_requested = False
    gizmo_window = WristGizmoWindow() if show_3d_view else None

    # FPS tracking
    stats_timer = time.perf_counter()
    loop_counter = 0
    udp_counter = 0
    loop_fps = 0.0
    udp_fps = 0.0

    # STEP 2: Camera setup
    cap = None
    zed = None
    zed_image = None
    zed_runtime = None
    
    if use_zed:
        # ZED Camera
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 60
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"❌ Impossible d'ouvrir la ZED: {err}")
            return
        
        actual_fps = zed.get_camera_information().camera_configuration.fps
        print(f"✅ ZED Camera ouverte - FPS réel: {actual_fps}")
        
        zed_image = sl.Mat()
        zed_runtime = sl.RuntimeParameters()
    else:
        # Webcam classique
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la webcam.")
            return

    # Threading: Queue pour les frames et résultats
    frame_queue = Queue(maxsize=2)
    result_queue = Queue(maxsize=2)
    stop_flag = [False]
    
    def capture_thread():
        """Thread dédié à la capture caméra."""
        nonlocal zed_image
        while not stop_flag[0]:
            if use_zed:
                err = zed.grab(zed_runtime)
                if err != sl.ERROR_CODE.SUCCESS:
                    continue
                zed.retrieve_image(zed_image, sl.VIEW.LEFT)
                frame = zed_image.get_data().copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    continue
            
            # Drop old frame if queue full
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except:
                    pass
            frame_queue.put((frame, time.perf_counter()))
    
    def detect_thread():
        """Thread dédié à la détection MediaPipe."""
        while not stop_flag[0]:
            try:
                frame, t_capture = frame_queue.get(timeout=0.1)
            except:
                continue
            
            t0 = time.perf_counter()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)
            t1 = time.perf_counter()
            
            # Drop old result if queue full
            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except:
                    pass
            result_queue.put((rgb_frame, detection_result, t_capture, t1 - t0))
    
    # Démarrer les threads
    cap_thread = Thread(target=capture_thread, daemon=True)
    det_thread = Thread(target=detect_thread, daemon=True)
    cap_thread.start()
    det_thread.start()
    
    print("🎉 HandOrientationDetector démarré (THREADED). Appuie sur Q pour quitter.")

    try:
        while True:
            t0 = time.perf_counter()
            
            # Récupérer le résultat de détection
            try:
                rgb_frame, detection_result, t_capture, detect_time = result_queue.get(timeout=0.1)
            except:
                continue

            t1 = time.perf_counter()
            
            loop_counter += 1
            sent_udp = False

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
                    orient_ok = orientation_tracker.calibrate(axes_for_tracker)
                    if orient_ok:
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

                # Calcul du splay (écartement entre doigts) - pas besoin de calibration
                # On passe les axes pour projeter dans le plan de la paume
                splay = compute_finger_splay(hand_landmarks, axes_for_tracker)
                if splay:
                    for finger, value in splay.items():
                        smoothed_splay[finger] = smooth_scalar(smoothed_splay.get(finger), value)

                orientation_tracker.update(axes_for_tracker)

                send_udp_metrics(smoothed_curls, smoothed_splay, orientation_tracker)
                sent_udp = True

            t2 = time.perf_counter()

            if sent_udp:
                udp_counter += 1

            draw_orientation_bars(annotated, orientation_tracker.values, orientation_tracker.is_calibrated)
            draw_splay_bars(annotated, smoothed_splay)

            if gizmo_window is not None:
                gizmo_window.update(
                    orientation_tracker.get_latest_matrix(),
                    orientation_tracker.get_latest_angles() if orientation_tracker.is_calibrated else None,
                )

            t3 = time.perf_counter()

            # STEP 5: Affichage (skip si headless)
            if not headless:
                cv2.imshow("Hand Orientation Detector", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.pollKey() & 0xFF
            
            t4 = time.perf_counter()
            
            # Accumuler les temps pour benchmark
            if benchmark:
                bench_capture += (t1 - t0)  # Temps d'attente résultat
                bench_detect += detect_time  # Temps MediaPipe (du thread)
                bench_process += (t2 - t1)
                bench_draw += (t3 - t2)
                bench_display += (t4 - t3)
                bench_count += 1

            # FPS stats toutes les secondes
            now = time.perf_counter()
            if now - stats_timer >= 1.0:
                elapsed = now - stats_timer
                loop_fps = loop_counter / elapsed
                udp_fps = udp_counter / elapsed
                
                if benchmark and bench_count > 0:
                    total = bench_capture + bench_process + bench_draw + bench_display
                    print(f"\n{'='*60}")
                    print(f"BENCHMARK THREADED ({bench_count} frames)")
                    print(f"  Wait result: {1000*bench_capture/bench_count:6.2f} ms  ({100*bench_capture/total:5.1f}%)")
                    print(f"  MediaPipe:   {1000*bench_detect/bench_count:6.2f} ms  (in parallel thread)")
                    print(f"  Process:     {1000*bench_process/bench_count:6.2f} ms  ({100*bench_process/total:5.1f}%)")
                    print(f"  Draw:        {1000*bench_draw/bench_count:6.2f} ms  ({100*bench_draw/total:5.1f}%)")
                    print(f"  Display:     {1000*bench_display/bench_count:6.2f} ms  ({100*bench_display/total:5.1f}%)")
                    print(f"  Loop time:   {1000*total/bench_count:6.2f} ms/frame")
                    print(f"  Loop FPS:    {loop_fps:4.1f} | UDP: {udp_fps:4.1f} pkt/s")
                    print(f"{'='*60}")
                    bench_capture = bench_detect = bench_process = bench_draw = bench_display = 0.0
                    bench_count = 0
                else:
                    print(f"[Threaded] Loop FPS: {loop_fps:4.1f} | UDP: {udp_fps:4.1f} pkt/s")
                
                loop_counter = 0
                udp_counter = 0
                stats_timer = now
                
            if key == ord('q'):
                break
            if key == ord('c'):
                calibration_requested = True
    finally:
        stop_flag[0] = True
        cap_thread.join(timeout=1.0)
        det_thread.join(timeout=1.0)
        if cap is not None:
            cap.release()
        if zed is not None:
            zed.close()
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
    parser.add_argument(
        "--use-zed",
        action="store_true",
        help="Utilise la caméra ZED au lieu de la webcam",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Désactive l'affichage pour optimiser les performances",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Affiche les temps détaillés de chaque étape du pipeline",
    )
    args = parser.parse_args()
    main(show_3d_view=args.show_3d_view, use_zed=args.use_zed, headless=args.no_display, benchmark=args.benchmark)
