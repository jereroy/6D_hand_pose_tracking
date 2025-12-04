import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import mediapipe as mp
import cv2
import pyzed.sl as sl
import numpy as np
import socket
import json
import matplotlib.pyplot as plt

from HandTrackingModule.HandTracking import HandTracking
from HandTrackingModule.Zed import Zed
from HandTrackingModule.Vis3D import Vis3D


# -------------------------
# UDP Setup
# -------------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_udp_right(hand_pts):
    """Send right hand 21 × 3 points as JSON via UDP."""
    if hand_pts is None or hand_pts.shape != (21,3):
        return
    
    packet = {
        "handedness": "Right",
        "landmarks": hand_pts.tolist()
    }
    msg = json.dumps(packet).encode("utf-8")
    sock.sendto(msg, (UDP_IP, UDP_PORT))
    print("Sent UDP right hand.")


# -------------------------
# Main
# -------------------------
def main():

    if len(sys.argv) == 2:
        print("Camera Mode: SVO")
        filename = sys.argv[1]
    else:
        print("Camera Mode: Live Streaming")
        filename = None
        
    # bring detector
    detector = HandTracking()
    # bring zed
    cam = Zed(filename)
    # bring visualizer
    vis3d = Vis3D()
    
    # print camera information
    cam.print_information()
    camera_params = cam.camera_params


    while True:
        err = cam.zed.grab(cam.runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS:
   
            # extract images from ZED
            cam.get_image()
            img = cam.img
            depth_img = cam.depth_img

            # Point cloud data
            pcl = cam.point_cloud

            # find hands
            img = detector.findHands(img)
            # find position
            data_left, data_right = detector.findpostion(depth_img, pcl, camera_params)

            # ----------- VISUALIZATION -----------
            # vis3d.show_hand(data_left,  vis3d.blue)
            # vis3d.show_hand(data_right, vis3d.red)

            # ----------- UDP SEND -----------
            send_udp_right(data_right)

            # half size images
            img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
            depth_img = cv2.resize(depth_img, (0, 0), None, 0.5, 0.5)

            detector.displayFPS(img)
            cv2.imshow("Image", img)
            cv2.imshow("Depth", depth_img)
     
            if cv2.waitKey(1) == 27:
                break

    cam.close()
    cv2.destroyAllWindows()

 
if __name__ == "__main__":
    main()
