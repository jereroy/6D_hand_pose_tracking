import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import mediapipe as mp
import cv2
import pyzed.sl as sl
import numpy as np
import matplotlib.pyplot as plt
from HandTrackingModule.HandTracking import HandTracking
from HandTrackingModule.Zed import Zed


def extract_landmarks_2d(detector, img):
    """
    Extrait les landmarks 2D (en pixels) pour chaque main à partir de MediaPipe.
    Retourne (lm_left, lm_right) comme listes de 21 tuples (x, y).
    """
    lm_left, lm_right = [], []
    h, w = img.shape[:2]

    if hasattr(detector, "results") and detector.results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(detector.results.multi_hand_landmarks):
            handedness = detector.results.multi_handedness[i].classification[0].index
            # 0 = Right, 1 = Left dans MediaPipe
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            if handedness == 1:
                lm_left = coords
            elif handedness == 0:
                lm_right = coords

    return lm_left, lm_right

def get_landmarks_3d(landmarks_2d, depth_map):
    """
    landmarks_2d : liste [(x, y), ...] en pixels
    depth_map : numpy (H, W) profondeur métrique (en mètres)
    Retour : array (N, 3) [x_pixel, y_pixel, z_metre]
    """
    import numpy as np
    landmarks_3d = []
    for pt in landmarks_2d:
        if pt is None or len(pt) != 2:
            landmarks_3d.append((np.nan, np.nan, np.nan))
            continue
        x, y = pt
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < depth_map.shape[0] and 0 <= xi < depth_map.shape[1]:
            z_val = depth_map[yi, xi]
            # z_val est un float32 (mètres)
            z = float(z_val)
            # Si z <= 0 ou NaN, on met NaN
            if not np.isfinite(z) or z <= 0:
                z = np.nan
        else:
            z = np.nan
        landmarks_3d.append((float(x), float(y), z))
    return np.array(landmarks_3d, dtype=np.float32)


def pixel_to_camera(landmarks_3d_px, fx, fy, cx, cy):
    """
    Convertit (u, v, z) en coordonnées caméra (X, Y, Z) en mètres.
    landmarks_3d_px : array (N, 3) avec (u_pixel, v_pixel, z_metre)
    fx, fy, cx, cy : paramètres intrinsèques de la caméra
    Retourne array (N, 3) [X, Y, Z] en mètres
    """
    if landmarks_3d_px.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    cam_points = np.full_like(landmarks_3d_px, np.nan, dtype=np.float32)

    for i, (u, v, z) in enumerate(landmarks_3d_px):
        if not np.isfinite(z) or z <= 0:
            continue
        X = (u - cx) / fx * z
        Y = (v - cy) / fy * z
        Z = z
        cam_points[i] = (X, Y, Z)

    return cam_points


def main():
    # Choix SVO vs Live
    if len(sys.argv) == 2:
        print("Camera Mode: SVO")
        filename = sys.argv[1]
    else:
        print("Camera Mode: Live Streaming")
        filename = None

    # Hand detector MediaPipe
    detector = HandTracking()

    # ZED
    cam = Zed(filename)
    cam.print_information()
    camera_params = cam.camera_params  # pour fx, fy, cx, cy

    fx = camera_params.fx
    fy = camera_params.fy
    cx = camera_params.cx
    cy = camera_params.cy

    # 3D plot
    plot = True
    if plot:
        fig = plt.figure()
        plt.ion()
        ax = fig.add_subplot(111, projection="3d")

    while True:
        err = cam.zed.grab(cam.runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS:
            # Récupère image + depth
            cam.get_image()
            img = cam.img
            depth_vis = cam.depth_img       # pour affichage
            depth_map = cam.depth_map_metric  # profondeur en mètres pour calcul

            # ZED renvoie souvent BGRA → passe en BGR pour OpenCV / MediaPipe
            if img.shape[2] == 4:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                img_bgr = img.copy()

            # Hand detection
            img_bgr = detector.findHands(img_bgr)

            # Landmarks 2D
            lm_left, lm_right = extract_landmarks_2d(detector, img_bgr)

            # Landmarks (u, v, z) avec z en mètres
            data_left_3d_px = (
                get_landmarks_3d(lm_left, depth_map)
                if len(lm_left) > 0
                else np.empty((0, 3), dtype=np.float32)
            )
            data_right_3d_px = (
                get_landmarks_3d(lm_right, depth_map)
                if len(lm_right) > 0
                else np.empty((0, 3), dtype=np.float32)
            )

            # Conversion en coords caméra (X, Y, Z) en mètres
            data_left_cam = pixel_to_camera(data_left_3d_px, fx, fy, cx, cy)
            data_right_cam = pixel_to_camera(data_right_3d_px, fx, fy, cx, cy)

            # Pour affichage OpenCV (resize)
            img_disp = cv2.resize(img_bgr, (0, 0), None, 0.5, 0.5)
            if depth_vis is not None:
                depth_disp = cv2.resize(depth_vis, (0, 0), None, 0.5, 0.5)
            else:
                depth_disp = np.zeros_like(img_disp)

            # Plot 3D
            if plot:
                # Fusionner les deux mains en coordonnées caméra
                if data_left_cam.shape[0] == 21 and data_right_cam.shape[0] == 21:
                    data_plot = np.vstack((data_right_cam, data_left_cam))
                elif data_right_cam.shape[0] == 21:
                    data_plot = data_right_cam
                elif data_left_cam.shape[0] == 21:
                    data_plot = data_left_cam
                else:
                    data_plot = np.empty((0, 3), dtype=np.float32)

                # Filtrer les points valides (pas NaN, Z > 0.1 m)
                if data_plot.size > 0:
                    valid = (~np.isnan(data_plot).any(axis=1)) & (data_plot[:, 2] > 0.1)
                    data_plot_valid = data_plot[valid]
                else:
                    data_plot_valid = np.empty((0, 3), dtype=np.float32)

                if data_plot_valid.size > 0:
                    X = data_plot_valid[:, 0]
                    Y = data_plot_valid[:, 1]
                    Z = data_plot_valid[:, 2]

                    x_min, x_max = np.min(X), np.max(X)
                    y_min, y_max = np.min(Y), np.max(Y)
                    z_min, z_max = np.min(Z), np.max(Z)
                    print(
                        f"X range: {x_min:.3f} to {x_max:.3f} | "
                        f"Y range: {y_min:.3f} to {y_max:.3f} | "
                        f"Z range: {z_min:.3f} to {z_max:.3f}"
                    )

                    ax.clear()

                    # Limites grossières en mètres (à ajuster selon ta scène)
                    ax.set_xlim3d(-0.4, 0.4)
                    ax.set_ylim3d(-0.4, 0.4)
                    ax.set_zlim3d(0.1, 1.2)
                    ax.set_xlabel("X (m)")
                    ax.set_ylabel("Y (m)")
                    ax.set_zlabel("Z (m)")

                    # Multiplier Y par -1 pour remettre la main à l'endroit
                    data_plot_fixed = data_plot_valid.copy()
                    data_plot_fixed[:,1] *= -1  # Inverse Y
                    data_plot_fixed[:,2] *= -1  # Inverse Z
                    ax.scatter3D(data_plot_fixed[:,0], data_plot_fixed[:,1], data_plot_fixed[:,2], s=40)

                    # Connexions des doigts (même topologie, indices 0..20)
                    edges = [
                        (1, 2),
                        (2, 3),
                        (3, 4),
                        (0, 5),
                        (5, 6),
                        (5, 9),
                        (1, 0),
                        (6, 7),
                        (7, 8),
                        (0, 9),
                        (9, 10),
                        (10, 11),
                        (11, 12),
                        (9, 13),
                        (13, 14),
                        (14, 15),
                        (15, 16),
                        (13, 17),
                        (17, 18),
                        (18, 19),
                        (19, 20),
                        (0, 17),
                    ]

                    n = data_plot_valid.shape[0]
                    # Si deux mains, on suppose 21 points par main
                    if n == 42:
                        for offset in [0, 21]:
                            for edge in edges:
                                i, j = edge[0] + offset, edge[1] + offset
                                ax.plot3D(
                                    [data_plot_fixed[i, 0], data_plot_fixed[j, 0]],
                                    [data_plot_fixed[i, 1], data_plot_fixed[j, 1]],
                                    [data_plot_fixed[i, 2], data_plot_fixed[j, 2]],
                                )
                    elif n == 21:
                        for edge in edges:
                            i, j = edge
                            ax.plot3D(
                                [data_plot_fixed[i, 0], data_plot_fixed[j, 0]],
                                [data_plot_fixed[i, 1], data_plot_fixed[j, 1]],
                                [data_plot_fixed[i, 2], data_plot_fixed[j, 2]],
                            )

                    plt.draw()
                    plt.pause(0.0001)
                else:
                    ax.clear()
                    ax.text2D(
                        0.5,
                        0.5,
                        "Aucun landmark 3D valide",
                        transform=ax.transAxes,
                        ha="center",
                    )
                    plt.draw()
                    plt.pause(0.0001)

            # Display
            detector.displayFPS(img_disp)
            cv2.imshow("Image", img_disp)
            cv2.imshow("Depth", depth_disp)

            key = cv2.waitKey(1)
            if key & 0xFF == 27:  # ESC pour quitter
                break

    cam.zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()