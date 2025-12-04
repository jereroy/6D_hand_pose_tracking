import open3d as o3d
import numpy as np

HAND_CONNECTIONS = [
    [0,1], [1,2], [2,3], [3,4],
    [0,5], [5,6], [6,7], [7,8],
    [0,9], [9,10], [10,11], [11,12],
    [0,13], [13,14], [14,15], [15,16],
    [0,17], [17,18], [18,19], [19,20],
]

class Vis3D():
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        # Axes
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(self.mesh_frame)

        # Unique point cloud like your original code
        self.pcd_hand = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd_hand)

        # Unique LineSet for bones
        self.line_set = o3d.geometry.LineSet()
        self.vis.add_geometry(self.line_set)

        # Colors
        self.blue = [0, 0, 1]
        self.red = [1, 0, 0]

    def show_hand(self, data, color=[0,0,1]):
        if data is None or data.shape != (21, 3):
            return

        # Update points
        self.pcd_hand.points = o3d.utility.Vector3dVector(data)
        self.pcd_hand.colors = o3d.utility.Vector3dVector(
            np.tile(color, (21,1))
        )

        # Update bones
        self.line_set.points = o3d.utility.Vector3dVector(data)
        self.line_set.lines = o3d.utility.Vector2iVector(HAND_CONNECTIONS)
        self.line_set.colors = o3d.utility.Vector3dVector(
            np.tile(color, (len(HAND_CONNECTIONS),1))
        )

        # Update scene
        self.vis.update_geometry(self.pcd_hand)
        self.vis.update_geometry(self.line_set)
        self.vis.poll_events()
        self.vis.update_renderer()
