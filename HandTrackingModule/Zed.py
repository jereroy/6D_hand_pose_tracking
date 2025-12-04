
import sys
import pyzed.sl as sl
import numpy as np

class Zed():
    def __init__(self, filename=None, depth_confidence=40):

        print("Bringing Up ZED CAMERA Information...")
        # Decide if SVO or Live
        if filename is None:
            print("Using Live stream from ZED camera")
            self.input_type = sl.InputType()
            self.svo_mode = False
        else:
            filepath = filename
            print(f"Reading SVO file: {filepath}")
            self.input_type = sl.InputType()
            self.input_type.set_from_svo_file(filepath)
            self.svo_mode = True

        # Initialize the ZED camera
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters(input_t=self.input_type)

        self.init_params.camera_resolution = sl.RESOLUTION.HD1080
        self.init_params.camera_fps = 30
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_minimum_distance = 0.3
        self.init_params.depth_maximum_distance = -1

        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            print("If using SVO, check if the path is correct")
            self.zed.close()
            sys.exit(1)

        # RuntimeParameters (API SDK 5)
        self.runtime_parameters = sl.RuntimeParameters()
        self.runtime_parameters.enable_fill_mode = True
        self.runtime_parameters.confidence_threshold = depth_confidence
        self.runtime_parameters.texture_confidence_threshold = depth_confidence

        cam_info = self.zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters
        self.camera_params = calib.left_cam

        self.fx = self.camera_params.fx
        self.fy = self.camera_params.fy
        self.cx = self.camera_params.cx
        self.cy = self.camera_params.cy

        # mats
        self.image = sl.Mat()
        self.depth = sl.Mat()          # pour l'affichage colorisé éventuel
        self.depth_measure = sl.Mat()  # profondeur métrique (float32)
        self.point_cloud = sl.Mat()
        self.confidence_map = sl.Mat()

        self.img = None                # image couleur
        self.depth_img = None          # depth pour affichage (uint8/BGR)
        self.depth_map_metric = None   # depth en mètres (float32)

    def print_information(self):
        cam_info = self.zed.get_camera_information()
        res = cam_info.camera_configuration.resolution
        print(f"Resolution: {res.width}, {res.height}.")
        print(f"Depth mode: {self.init_params.depth_mode}.")
        print(f"Fill mode enabled: {self.runtime_parameters.enable_fill_mode}")
        if self.svo_mode:
            print(f"Frame count: {self.zed.get_svo_number_of_frames()}.\n")

    def get_image(self):
        # Image gauche
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)

        # Mesure de profondeur métrique (en mètres)
        self.zed.retrieve_measure(self.depth_measure, sl.MEASURE.DEPTH, sl.MEM.CPU)

        # Point cloud (si tu en as besoin)
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)

        # Carte de confiance numérique
        self.zed.retrieve_measure(self.confidence_map, sl.MEASURE.CONFIDENCE, sl.MEM.CPU)

        # Numpy arrays
        self.img = self.image.get_data()
        depth_array = self.depth_measure.get_data().astype(np.float32)

        # On garde la depth métrique brute pour get_landmarks_3d
        self.depth_map_metric = depth_array

        # Pour l'affichage : on colorise nous-mêmes
        depth_vis = depth_array.copy()
        # Remplacer NaN par 0
        depth_vis[np.isnan(depth_vis)] = 0.0
        # Clamp à une distance max pour le display (par ex. 1.5 m)
        max_display_dist = 1.5  # à ajuster selon ta scène
        depth_vis = np.clip(depth_vis, 0.0, max_display_dist)
        depth_vis = (depth_vis / max_display_dist * 255.0).astype(np.uint8)

        import cv2
        self.depth_img = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    def close(self):
        if hasattr(self, 'zed'):
            self.zed.close()
