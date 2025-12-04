import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import mediapipe as mp
import cv2
import pyzed.sl as sl
import numpy as np
import json
import socket
import matplotlib.pyplot as plt

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from HandTrackingModule.Zed import Zed
from HandTrackingModule.Vis3D import Vis3D

HAND_CONNECTIONS = [
    [0,1], [1,2], [2,3], [3,4],
    [0,5], [5,6], [6,7], [7,8],
    [0,9], [9,10], [10,11], [11,12],
    [0,13], [13,14], [14,15], [15,16],
    [0,17], [17,18], [18,19], [19,20],
]

# =========================================================
# ----------------------- UDP SETUP -----------------------
# =========================================================
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_udp(handedness, pts3d):
    """Send 21 XYZ joints via UDP as JSON."""
    print("Sent UDP:", pts3d)


    if pts3d is None:
        return

    packet = {
        "handedness": handedness,
        "landmarks": pts3d.tolist()
    }

    msg = json.dumps(packet).encode("utf-8")
    sock.sendto(msg, (UDP_IP, UDP_PORT))



# =========================================================
# --------------- MEDIAPIPE TASKS INIT --------------------
# =========================================================
MODEL_PATH = "hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

mp_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    running_mode=RunningMode.VIDEO,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

mp_detector = HandLandmarker.create_from_options(mp_options)


def mp_image_from_bgr(frame):
    """Convert BGR to MediaPipe SRGB image."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


# =========================================================
# ------- Convert MediaPipe landmarks to ZED 3D ----------
# =========================================================
def get_hands_3d(result, cam):
    """Convert MediaPipe 2D → ZED 3D coordinates."""
    if not result.hand_landmarks:
        return None, None

    cam.get_image()
    depth = cam.depth_img
    pcl = cam.point_cloud

    # Handle missing depth
    if depth is None or depth.ndim != 3:
        return None, None

    H, W, P = depth.shape

    left_hand = None
    right_hand = None

    for i, handedness in enumerate(result.handedness):
        lm = result.hand_landmarks[i]
        pts3d = []

        for p in lm:
            x = int(p.x * W)
            y = int(p.y * H)

            # Bounds check
            if x < 0 or y < 0 or x >= W or y >= H:
                pts3d.append([np.nan, np.nan, np.nan])
                continue

            err, p3d = pcl.get_value(x, y)

            if err == sl.ERROR_CODE.SUCCESS:
                X = float(p3d[0])
                Y = float(p3d[1])
                Z = float(p3d[2])
            else:
                X = Y = Z = np.nan

            pts3d.append([X, Y, Z])

        pts3d = np.array(pts3d, dtype=np.float32)
        print("pts3d:", pts3d)

        if handedness[0].category_name == "Right":
            right_hand = pts3d
        else:
            left_hand = pts3d

    return left_hand, right_hand


# =========================================================
# --------------------------- MAIN ------------------------
# =========================================================
def main():

    print("Camera Mode: Live Streaming")

    # ZED init
    cam = Zed()
    cam.print_information()

    # 3D visualizer
    vis3d = Vis3D()

    # Webcam for MediaPipe
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    timestamp = 0

    while True:

        # ZED grab
        err = cam.zed.grab(cam.runtime_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            continue

        cam.get_image()
        zed_img = cam.img
        depth_img = cam.depth_img

        # Webcam frame for MediaPipe
        ok, frame = cap.read()
        if not ok:
            continue

        timestamp += int(1000 / fps)
        mp_img = mp_image_from_bgr(frame)

        # MediaPipe detection
        result = mp_detector.detect_for_video(mp_img, timestamp)

        # Extract 3D hands
        left_pts, right_pts = get_hands_3d(result, cam)

        # -------------------------
        # UDP SEND
        # -------------------------
        # send_udp("Left", left_pts)
        send_udp("Right", right_pts)

        # -------------------------
        # VIS3D DISPLAY
        # -------------------------
        # if left_pts is not None:
        #     vis3d.show_hand(left_pts, vis3d.blue)

        # if right_pts is not None:
        #     vis3d.show_hand(right_pts, vis3d.red)

        # -------------------------
        # ZED DISPLAY (image + depth)
        # -------------------------
        show_img = cv2.resize(zed_img, (0, 0), fx=0.5, fy=0.5)
        show_depth = cv2.resize(depth_img, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow("Image", show_img)
        cv2.imshow("Depth", show_depth)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cam.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
