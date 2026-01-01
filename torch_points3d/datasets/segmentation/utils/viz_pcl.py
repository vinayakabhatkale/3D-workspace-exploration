import open3d as o3d
import h5py
import numpy as np
import sys


class Visualizations:
    def __init__(self):
        # TODO not that hardcoded colors
        self.possible_colors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1]
            ]

    def visualize_point_cloud(self, points: np.ndarray= None, colors: np.ndarray=None, point_labels: np.ndarray=None, filename: str=None, show_labels: bool=False) -> None:

        if show_labels is False:
            point_colors = colors
        else:
            point_colors = list()
            label_list = point_labels.tolist()
            map_label_to_color = lambda i: self.possible_colors[int(i)]
            point_colors = list(map(map_label_to_color, label_list))

        print(f'Point cloud contains {len(point_colors)} points')

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        self._vis_point_cloud(pcd)

    def _vis_point_cloud(self, pcd: o3d.geometry.PointCloud) -> None:
        """
            visualize point cloud with open3d standard visualizer
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        view_ctl = vis.get_view_control()
        view_ctl.set_front((1, 1, 0))
        view_ctl.set_up((0, -1, -1))
        #view_ctl.set_lookat(pcd.get_center())
        vis.run()
        vis.destroy_window()



    def _import_from_hdf(self, filename):
        """
        import point cloud from hdf file format
        :return x,y,z, label
        """ 
        f = h5py.File(filename)
        x = f['location_x']
        y = f['location_y']
        z = f['location_z']
        label = f['categoryID']

        return np.array(x), np.array(y), np.array(z), np.array(label)



if __name__ == '__main__':
    viz = Visualizations()
    viz.visualize_point_cloud(filename=sys.argv[1])
