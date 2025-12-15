import argparse
import json
import os
import socket
import sys
import time
from queue import Queue
from threading import Thread

import cv2
import pyzed.sl as sl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from HandTrackingModule.HandTracking import HandTracking
from HandTrackingModule.Zed import Zed
from HandOrientationDetector import (
    compute_hand_axes,
    compute_finger_curls,
    draw_finger_curl_ui,
    draw_horizontal_bars,
    draw_orientation_bars,
    smooth_scalar,
    smooth_vector,
    HorizontalFingerTracker,
    WristOrientationTracker,
    build_udp_payload,
)


UDP_IP = "127.0.0.1"
UDP_PORT = 5005
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Configuration performance
DISPLAY_EVERY_N_FRAMES = 2  # Afficher 1 frame sur N pour gagner en FPS

# Queue pour envoi UDP asynchrone
udp_queue: Queue = Queue(maxsize=2)


def udp_sender_thread():
    """Thread dédié à l'envoi UDP pour ne pas bloquer la boucle principale."""
    while True:
        payload = udp_queue.get()
        if payload is None:  # Signal d'arrêt
            break
        try:
            message = json.dumps(payload).encode("utf-8")
            udp_socket.sendto(message, (UDP_IP, UDP_PORT))
        except OSError:
            pass  # Ignorer les erreurs UDP silencieusement


class ZedLandmark:
    __slots__ = ("x", "y", "z")

    def __init__(self, coords):
        self.x = float(coords[0])
        self.y = float(coords[1])
        self.z = float(coords[2])


def array_to_landmarks(array):
    return [ZedLandmark(point) for point in array]


def send_udp_metrics(curls, horizontal_values, orientation_tracker):
    """Envoie les métriques via le thread UDP (non-bloquant)."""
    payload = build_udp_payload(curls, horizontal_values, orientation_tracker)
    try:
        # Non-bloquant: si la queue est pleine, on drop le paquet
        udp_queue.put_nowait(payload)
    except Exception:
        pass  # Queue pleine, on skip ce paquet


def main():
    parser = argparse.ArgumentParser(description="ZED orientation + finger metrics streamer")
    parser.add_argument("--svo", type=str, default=None, help="Chemin vers un fichier SVO (optionnel)")
    parser.add_argument("--show-depth", action="store_true", help="Affiche la vue profondeur colorisée")
    args = parser.parse_args()

    detector = HandTracking()
    cam = Zed(args.svo)
    cam.print_information()
    camera_params = cam.camera_params

    smoothed_axes = {}
    smoothed_curls = {}
    horizontal_tracker = HorizontalFingerTracker()
    orientation_tracker = WristOrientationTracker()
    calibration_requested = False
    stats_timer = time.perf_counter()
    loop_counter = 0
    udp_counter = 0
    loop_fps = 0.0
    udp_fps = 0.0
    display_enabled = True
    frame_counter = 0  # Pour l'affichage 1 frame sur N

    # Démarrer le thread UDP
    udp_thread = Thread(target=udp_sender_thread, daemon=True)
    udp_thread.start()

    try:
        while True:
            err = cam.zed.grab(cam.runtime_parameters)
            if err != sl.ERROR_CODE.SUCCESS:
                continue

            loop_counter += 1
            frame_counter += 1
            sent_udp = False

            cam.get_image()
            img = cam.img.copy()
            depth_img = cam.depth_img
            point_cloud = cam.point_cloud

            img = detector.findHands(img)
            _, data_right = detector.findpostion(depth_img, point_cloud, camera_params)

            axes_for_tracker = None
            if data_right is not None and data_right.shape == (21, 3):
                hand_landmarks = array_to_landmarks(data_right)

                axes = compute_hand_axes(hand_landmarks)
                if axes:
                    blended_axes = {}
                    for axis_name, axis_vec in axes.items():
                        blended = smooth_vector(smoothed_axes.get(axis_name), axis_vec)
                        if blended is None:
                            continue
                        blended_axes[axis_name] = blended
                    if blended_axes:
                        smoothed_axes = blended_axes
                        axes_for_tracker = blended_axes
                    else:
                        axes_for_tracker = axes

                if calibration_requested and axes_for_tracker:
                    horiz_ok = horizontal_tracker.calibrate(hand_landmarks, axes_for_tracker)
                    orient_ok = orientation_tracker.calibrate(axes_for_tracker)
                    if horiz_ok or orient_ok:
                        print(" Calibration enregistrée (ZED)")
                        calibration_requested = False

                curls = compute_finger_curls(hand_landmarks)
                if curls:
                    for finger, value in curls.items():
                        smoothed_curls[finger] = smooth_scalar(smoothed_curls.get(finger), value)
                    draw_finger_curl_ui(img, smoothed_curls)

                horizontal_tracker.update(hand_landmarks, axes_for_tracker)
                orientation_tracker.update(axes_for_tracker)
                send_udp_metrics(smoothed_curls, horizontal_tracker.values, orientation_tracker)
                sent_udp = True

            if sent_udp:
                udp_counter += 1

            now = time.perf_counter()
            if now - stats_timer >= 1.0:
                elapsed = now - stats_timer
                loop_fps = loop_counter / elapsed
                udp_fps = udp_counter / elapsed
                loop_counter = 0
                udp_counter = 0
                stats_timer = now

                if not display_enabled:
                    stats_text_snapshot = f"Loop FPS: {loop_fps:4.1f} | UDP: {udp_fps:4.1f} pkt/s"
                    print(f"[ZED] {stats_text_snapshot}", flush=True)

            draw_horizontal_bars(img, horizontal_tracker.values, horizontal_tracker.is_calibrated)
            draw_orientation_bars(img, orientation_tracker.values, orientation_tracker.is_calibrated)

            stats_text = f"Loop FPS: {loop_fps:4.1f} | UDP: {udp_fps:4.1f} pkt/s"

            # Afficher seulement 1 frame sur N pour économiser du CPU
            should_display = display_enabled and (frame_counter % DISPLAY_EVERY_N_FRAMES == 0)

            if should_display:
                cv2.putText(img, stats_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                preview = cv2.resize(img, (0, 0), None, 0.5, 0.5)
                cv2.imshow("ZED Image", preview)

                if args.show_depth and depth_img is not None:
                    depth_preview = cv2.resize(depth_img, (0, 0), None, 0.5, 0.5)
                    cv2.imshow("ZED Depth", depth_preview)

            # Utiliser pollKey (non-bloquant) quand on n'affiche pas
            if should_display:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.pollKey() & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                calibration_requested = True
            if key == ord("v"):
                display_enabled = not display_enabled
                state = "activée" if display_enabled else "coupée"
                print(f"[ZED] {stats_text} | Fenêtre {state}")

    finally:
        # Arrêter proprement le thread UDP
        udp_queue.put(None)
        udp_thread.join(timeout=1.0)
        cam.close()
        cv2.destroyAllWindows()
        udp_socket.close()


if __name__ == "__main__":
    main()
