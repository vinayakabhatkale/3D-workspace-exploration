from segments import SegmentsClient
import open3d as o3d
import requests
import os
import numpy as np
from sys import maxsize
import pickle

class PCLProcessing:
    def __init__(self, api_key: str, dataset_name: str, camera: str):
        self.client = SegmentsClient(api_key=api_key)
        self.camera = camera
        self.samples = self.client.get_samples(dataset_identifier=dataset_name)
        self.list_pcds = list()
        self.ds_name = dataset_name
        self.possible_colors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1]
            ]

    def download_all_pcls(self):
        for sample in self.samples:
            url = sample.attributes.pcd.url
            os.makedirs(f'./download/{self.camera}',exist_ok=True)
            filename = f'./download/{self.camera}/{sample.name}'

            information = dict()
            information['absfilename'] = filename
            information['filename'] = sample.name
            information['uuid'] = sample.uuid

            self.list_pcds.append(information)

            if os.path.exists(filename):
                continue

            # here the actual download happens
            pcd = requests.get(url=url)
            with open(filename, 'wb') as f:
                f.write(pcd.content)

    def create_labeled_pcls(self):
        list_pcls = list()
        number_points_min = maxsize
        for pcd in self.list_pcds:
            abs_file_name = pcd['absfilename']
            filename = pcd['filename']

            uuid = pcd['uuid']

            pcl = o3d.io.read_point_cloud(abs_file_name)

            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.100, origin=(0,0,0))

            # realsense cameras have the coordinate system different than zivid or azure kinect
            if not 'kinect' in self.ds_name:
                print('Turning point cloud into zivid coordinate frame')
                R = pcl.get_rotation_matrix_from_quaternion((0,1,0,0))
                pcl = pcl.rotate(R, center=(0,0,0))
            # remove noise from point cloud
            #pcl, ind_sor = pcl.remove_statistical_outlier(nb_neighbors=100,
            #                                        std_ratio=3.0)

            #o3d.visualization.draw_geometries([pcl, origin])

            #pcl.translate(-1 * translation)
            #pcl.rotate(rot_mat.transpose())
             
            points = pcl.points
            colors = pcl.colors

            label = self.client.get_label(uuid)

            annotations = label.attributes.annotations
            point_annotations = label.attributes.point_annotations
            conversion_dict = dict()

            # there might be points in the point cloud that are not labeled
            # the default label for these points is storage rack
            conversion_dict[0] = 0

            # there are several objects for each class
            # each object has its own id but refers to a class (category_id)
            for annotation in annotations:
                conversion_dict[annotation.id] = annotation.category_id
                if annotation.category_id == 4:
                    conversion_dict[annotation.id] = 0

            # lambda function to get the semantic label from the object id
            convert_id_to_label = lambda id: conversion_dict[id] 

            semantic_labels = list(map(convert_id_to_label, point_annotations))

            map_label_to_color = lambda i: self.possible_colors[i]

            np_points = np.asarray(points) 

            # if points are in mm then convert to m
            if abs(np.mean(np_points[0])) > 10:
                np_points = np_points / 1000.

            np_label = np.asarray(semantic_labels)#[ind_sor]
            np_color = np.asarray(colors)

            np_pcl = np.concatenate(
                (
                np.expand_dims(np_label, axis=1), 
                np_points,
                np_color
                ),
                axis=1
            )

            # get minimum of points needed in numpy array
            if np_pcl.shape[0] < number_points_min:
                number_points_min = np_pcl.shape[0]

            list_pcls.append(np_pcl)
        
        rng = np.random.default_rng()
        for idx, pcl in enumerate(list_pcls):
            a = np.arange(pcl.shape[0])
            point_idxs = rng.choice(a, size=number_points_min, replace=False)
            reduced_pcl = pcl[point_idxs]
            list_pcls[idx] = np.expand_dims(reduced_pcl, axis=0)

            # only visualization
            #point_cloud = o3d.geometry.PointCloud()
            #point_cloud.points = o3d.utility.Vector3dVector(reduced_pcl[:,1:4])
            #point_cloud.colors = o3d.utility.Vector3dVector(reduced_pcl[:,4:])
            #o3d.visualization.draw_geometries([point_cloud])

        return np.concatenate(list_pcls, axis=0)

if __name__ == '__main__':
    api_key = "b47bdc4382bfff4990ebe25258dc594447f63935"
    camera = 'kinect'
    dataset = f'anwie/azure_{camera}'

    pcl_processing = PCLProcessing(api_key=api_key, dataset_name=dataset, camera=camera)
    pcl_processing.download_all_pcls()
    labeled_pcls = pcl_processing.create_labeled_pcls()
    np.save(f'./np/{camera}/synth/{camera}_pclsbasic', labeled_pcls)
    print(labeled_pcls.shape)
