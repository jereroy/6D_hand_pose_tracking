import sys
import time
import numpy as np
import open3d as o3d
import pyzed.sl as sl

# -------------------------
# CONFIG
# -------------------------

# Indices des keypoints des mains (BODY_38)
HAND_IDXS = np.array([30, 31, 32, 33, 34, 35, 36, 37], dtype=np.int32)

# Métadonnées (gauche/droite + ordre doigts)
HAND_META = np.array([
    ("L", 0), ("R", 0),
    ("L", 1), ("R", 1),
    ("L", 2), ("R", 2),
    ("L", 3), ("R", 3),
], dtype=object)

LEFT_COLOR = np.array([0.0, 0.3, 1.0], dtype=np.float32)
RIGHT_COLOR = np.array([0.0, 1.0, 0.3], dtype=np.float32)
OTHER_COLOR = np.array([0.6, 0.6, 0.6], dtype=np.float32)  # corps gris


# -------------------------
# UTILITAIRES
# -------------------------

def body_to_numpy_keypoints(body):
    """Transforme sl.Body.keypoint -> np.array (N,3)."""
    return np.array([[p[0], p[1], p[2]] for p in body.keypoint], dtype=np.float32)


def extract_hand_points_and_meta(body):
    """Retourne (pts, sides, fingers) pour les mains uniquement."""
    kps = body_to_numpy_keypoints(body)
    if kps.shape[0] <= HAND_IDXS.max():
        return None, None, None

    hand_pts = kps[HAND_IDXS]
    valid_mask = ~(np.all(hand_pts == 0, axis=1))

    if not np.any(valid_mask):
        return None, None, None

    hand_pts = hand_pts[valid_mask]
    meta_valid = HAND_META[valid_mask]
    sides = np.array([m[0] for m in meta_valid])
    fingers = np.array([m[1] for m in meta_valid], dtype=np.int32)

    return hand_pts, sides, fingers


def extract_all_body_points(body):
    """Retourne (pts_valid, mask) pour tous les points du BODY_38."""
    kps = body_to_numpy_keypoints(body)
    valid_mask = ~(np.all(kps == 0.0, axis=1))
    return kps[valid_mask], valid_mask


def normalize_points(pts):
    """Normalise pour affichage dans Open3D."""
    p = pts.copy()
    center = p.mean(axis=0, keepdims=True)
    p -= center
    scale = np.linalg.norm(p, axis=1).max()
    if scale > 0:
        p /= scale
    return p


def build_hand_lines(sides, fingers):
    """Construit les segments reliant les doigts (L et R indépendants)."""
    lines = []
    for side in ["L", "R"]:
        idxs = np.where(sides == side)[0]
        if len(idxs) < 2:
            continue
        order = np.argsort(fingers[idxs])
        pts_sorted = idxs[order]
        for i in range(len(pts_sorted) - 1):
            lines.append([int(pts_sorted[i]), int(pts_sorted[i+1])])
    return np.array(lines, dtype=np.int32) if lines else None


def build_full_body_colors(valid_mask, sides_main, fingers_main):
    """Construit les couleurs pour les 38 points (gris + mains L/R)."""
    colors = []
    body_indices = np.where(valid_mask)[0]

    for global_idx in body_indices:
        if global_idx in HAND_IDXS:
            m = np.where(HAND_IDXS == global_idx)[0][0]
            side = sides_main[m]
            colors.append(LEFT_COLOR if side == "L" else RIGHT_COLOR)
        else:
            colors.append(OTHER_COLOR)
    return np.array(colors, dtype=np.float32)


# -------------------------
# ZED INITIALISATION
# -------------------------

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_units = sl.UNIT.METER
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Erreur ouverture ZED :", err)
    sys.exit(1)

# Tracking
pos_params = sl.PositionalTrackingParameters()
pos_params.set_as_static = True
zed.enable_positional_tracking(pos_params)

# Body tracking
body_params = sl.BodyTrackingParameters()
body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
body_params.enable_tracking = True
body_params.enable_body_fitting = True
body_params.body_format = sl.BODY_FORMAT.BODY_38

zed.enable_body_tracking(body_params)
body_runtime = sl.BodyTrackingRuntimeParameters()

bodies = sl.Bodies()


# -------------------------
# OPEN3D INITIALISATION
# -------------------------

vis = o3d.visualization.Visualizer()
vis.create_window("ZED BODY_38 Stable View", 900, 720)

pcd = o3d.geometry.PointCloud()
line_set = o3d.geometry.LineSet()

render_opt = vis.get_render_option()
render_opt.point_size = 8.0

geoms_added = False
bbox_initialized = False


# -------------------------
# BOUCLE
# -------------------------

print("Boucle tracking…")

try:
    while True:
        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_bodies(bodies, body_runtime)
        if len(bodies.body_list) == 0:
            vis.poll_events()
            vis.update_renderer()
            continue

        body = bodies.body_list[0]

        # ---- Mains ----
        hand_pts, sides, fingers = extract_hand_points_and_meta(body)
        if hand_pts is None:
            vis.poll_events()
            vis.update_renderer()
            continue

        print(f"Detected hands: {list(sides)}")

        # ---- Corps complet ----
        all_pts, valid_mask = extract_all_body_points(body)

        # Normalisation pour affichage
        pts_norm = normalize_points(all_pts)

        # Couleurs complètes
        full_colors = build_full_body_colors(valid_mask, sides, fingers)

        # Point cloud
        pcd.points = o3d.utility.Vector3dVector(pts_norm.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(full_colors.astype(np.float64))

        print(f"Total keypoints displayed: {len(pts_norm)}")

        # LineSet mains
        lines = build_hand_lines(sides, fingers)
        if lines is not None:
            line_set.points = o3d.utility.Vector3dVector(hand_pts.astype(np.float64))
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(
                np.array([LEFT_COLOR if s == "L" else RIGHT_COLOR for s in sides])[:len(lines)]
            )
        
        print(f"Total hand lines displayed: {len(lines) if lines is not None else 0}")

        # ----- Open3D update -----

        if not geoms_added:
            vis.add_geometry(pcd, reset_bounding_box=False)
            vis.add_geometry(line_set, reset_bounding_box=False)
            geoms_added = True

        else:
            vis.update_geometry(pcd)
            vis.update_geometry(line_set)

        print("Updated Open3D geometries.")

        # ---- Fix bounding box une fois ----
        if not bbox_initialized:
            print("[INFO] Première détection -> initialisation caméra")
            bbox_initialized = True

        vis.reset_view_point(True)

        if not vis.poll_events():
            break

        vis.update_renderer()

finally:
    print("Fermeture ZED / Open3D…")
    vis.destroy_window()
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
