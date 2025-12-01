import sys
import mediapipe as mp  # <- pas utilisé dans cette classe, tu peux l’enlever si tu veux
import pyzed.sl as sl
import numpy as np

class Zed():
    def __init__(self, filename=None, depth_confidence=100):

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

        # Résolution & FPS (ça n’a pas changé)
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080
        self.init_params.camera_fps = 30

        # Profondeur – en SDK 5, NEURAL marche encore.
        # Tu peux tester NEURAL_PLUS qui est recommandé dans les exemples récents.
        # self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL

        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_minimum_distance = 0.3
        self.init_params.depth_maximum_distance = 40

        # Open the camera
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            print("If using SVO, check if the path is correct")
            self.zed.close()
            sys.exit(1)

        # RuntimeParameters (API SDK 5)
        self.runtime_parameters = sl.RuntimeParameters()

        # ⚠️ Changement important : SENSING_MODE supprimé.
        # Ancien:
        #   self.runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL
        # Nouveau:
        self.runtime_parameters.enable_fill_mode = True  # équivalent de FILL

        # Réglage de la confiance (toujours dispo)
        self.runtime_parameters.confidence_threshold = depth_confidence
        self.runtime_parameters.texture_confidence_threshold = depth_confidence

        # Get Camera Calibration Parameters (API SDK 5)
        cam_info = self.zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters
        self.camera_params = calib.left_cam

        self.fx = self.camera_params.fx  # Focal length in pixels (x-axis)
        self.fy = self.camera_params.fy  # Focal length in pixels (y-axis)
        self.cx = self.camera_params.cx  # X-coordinate of the principal point
        self.cy = self.camera_params.cy  # Y-coordinate of the principal point

        # declare image, depth, point cloud, confidence
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()
        self.confidence_map = sl.Mat()

        self.img = None
        self.depth_img = None

    def print_information(self):
        cam_info = self.zed.get_camera_information()
        # En SDK récents on passe par camera_configuration.resolution
        res = cam_info.camera_configuration.resolution
        print(f"Resolution: {res.width}, {res.height}.")
        # print(f"Camera FPS: {cam_info.camera_fps}")
        print(f"Depth mode: {self.init_params.depth_mode}.")
        print(f"Fill mode enabled: {self.runtime_parameters.enable_fill_mode}")
        if self.svo_mode:
            print(f"Frame count: {self.zed.get_svo_number_of_frames()}.\n")

    def get_image(self):
        # Retrieve left rectified image
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)

        # Si tu voulais vraiment la “carte de confiance” en image :
        self.zed.retrieve_image(self.depth, sl.VIEW.CONFIDENCE)

        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)

        # Retrieve confidence map (valeurs numériques)
        self.zed.retrieve_measure(self.confidence_map, sl.MEASURE.CONFIDENCE, sl.MEM.CPU)

        # convert zed image to numpy array
        self.img = self.image.get_data()
        self.depth_img = self.depth.get_data()
