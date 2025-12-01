import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------
# Config
# -------------------------

MODEL_PATH = "hand_landmarker.task"
CAM_INDEX = 0  # 0 = webcam par défaut

# -------------------------
# MediaPipe Tasks init
# -------------------------

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

base_options = BaseOptions(model_asset_path=MODEL_PATH)
options = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,  # on laisse 2, mais on ne gardera que la main droite
    running_mode=RunningMode.VIDEO,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = HandLandmarker.create_from_options(options)

# -------------------------
# Utilitaires
# -------------------------

def mp_image_from_bgr(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

def get_right_hand_world_landmarks(result):
    """
    Retourne un np.array (21, 3) pour la main droite seulement,
    ou None si aucune main droite détectée.
    """
    if not result.hand_world_landmarks:
        return None

    right_index = None
    if result.handedness:
        for i, hand_handedness in enumerate(result.handedness):
            if hand_handedness and hand_handedness[0].category_name == "Right":
                right_index = i
                break

    if right_index is None:
        return None

    hand = result.hand_world_landmarks[right_index]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
    return pts

def normalize_hand(points):
    pts = points.copy()
    origin = pts[0]  # poignet
    pts -= origin
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 0:
        pts /= scale
    return pts

# Connexions entre landmarks (squelette)
HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],      # pouce
    [0, 5], [5, 6], [6, 7], [7, 8],      # index
    [0, 9], [9, 10], [10, 11], [11, 12], # majeur
    [0, 13], [13, 14], [14, 15], [15, 16], # annulaire
    [0, 17], [17, 18], [18, 19], [19, 20], # auriculaire
    [5, 9], [9, 13], [13, 17]            # paume
]

# -------------------------
# Caméra
# -------------------------

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 30.0

# -------------------------
# Open3D
# -------------------------

vis = o3d.visualization.Visualizer()
vis.create_window("MediaPipe 3D Right Hand", width=800, height=600)

pcd = o3d.geometry.PointCloud()
line_set = o3d.geometry.LineSet()

hand_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # rouge

render_opt = vis.get_render_option()
render_opt.point_size = 8.0

geoms_added = False
first_valid_frame = True

# -------------------------
# Boucle principale
# -------------------------

frame_idx = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame non valide")
            break

        timestamp_ms = int((frame_idx / fps) * 1000)
        frame_idx += 1

        mp_image = mp_image_from_bgr(frame)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        # Récupère seulement la main droite
        right_hand_pts = get_right_hand_world_landmarks(result)

        if right_hand_pts is not None and right_hand_pts.shape == (21, 3):
            pts = normalize_hand(right_hand_pts)

            # Met à jour le PointCloud
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(hand_color, (pts.shape[0], 1))
            )

            # Met à jour le LineSet
            line_set.points = o3d.utility.Vector3dVector(pts)
            line_set.lines = o3d.utility.Vector2iVector(
                np.array(HAND_CONNECTIONS, dtype=np.int32)
            )
            line_set.colors = o3d.utility.Vector3dVector(
                np.tile(hand_color, (len(HAND_CONNECTIONS), 1))
            )

            # Ajoute les géométries à la scène la première fois
            if not geoms_added:
                vis.add_geometry(pcd, reset_bounding_box=True)
                vis.add_geometry(line_set, reset_bounding_box=False)
                geoms_added = True
            else:
                vis.update_geometry(pcd)
                vis.update_geometry(line_set)

            if first_valid_frame:
                vis.reset_view_point(True)
                first_valid_frame = False

        # update Open3D
        if not vis.poll_events():
            break
        vis.update_renderer()

finally:
    vis.destroy_window()
    cap.release()
    cv2.destroyAllWindows()
