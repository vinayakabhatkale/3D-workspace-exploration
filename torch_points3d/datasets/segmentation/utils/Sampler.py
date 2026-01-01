import torch
import random
import numpy as np
from sklearn.metrics import pairwise_distances
import pickle
from os.path import exists

from typing import Tuple

random.seed(0)

class Sampler:
    def __init__(self):
        pass

    @staticmethod
    def sample_point_clouds(point_clouds: torch.tensor, number_output_points: int=1024, sampling_strategy: str="random", n:int=2, n_pcls: int=1):
        sampled_point_clouds = torch.zeros(point_clouds.shape[0] * n_pcls, number_output_points, point_clouds.shape[2])
        for i in range(point_clouds.shape[0]):
            point_cloud = point_clouds[i]
            for idx in range(n_pcls):
                store_idx = i * n_pcls + idx 
                sampled_point_cloud = Sampler.sample_point_cloud(point_cloud, 
                    number_output_points=number_output_points, sampling_strategy=sampling_strategy, n=n)
                sampled_point_clouds[store_idx] = sampled_point_cloud
        return sampled_point_clouds


    @staticmethod
    def sample_point_cloud(point_cloud: torch.tensor, number_output_points: int=1024, sampling_strategy: str="random", n:int=2):
        """
            sample points from point cloud

            params:
                point_cloud: Tuple containing points of point cloud and their labels
                number_output_points: number of points of samples point cloud 
                sampling_strategy: currently supported: "random"
                n: sample every nth element -> only used for this one sampling strategy
                n_pcls: as the input point clouds usually have way more points than the number of output points we can extract more than one point cloud from each input pcl
        """

        if sampling_strategy == "random":
            sampled_point_cloud = Sampler.sample_random(point_cloud, number_output_points)
        elif sampling_strategy == "greedy_farthest_point":
            sampled_point_cloud = Sampler.farthest_point_sampling(point_cloud, number_output_points)
        elif sampling_strategy == "every_nth_element":
            sampled_point_cloud = Sampler.sample_every_nth_element(point_cloud, number_output_points, n=n)
        else:
            raise Exception('Invalid sampling_strategy given')

        return sampled_point_cloud

    @staticmethod
    def sample_every_nth_element(point_cloud: torch.tensor, number_output_points: int, n: int):
        """
            samples every nth element of a point cloud
            first every nth element will be sampled, 
            then starting in the middle to make sure the points in the center stay in the point cloud

            params:
                point_cloud: Tuple containing points of point cloud and their labels
                number_output_points:   number of points of samples point cloud. If number output points is None just every nth
                                        element will be chosen and no specific number of output points will be selected. Can 
                                        be used if the point cloud should be reduced by size if another following sampling strategy 
                                        takes too much time
                n: every nth element in the point cloud will be sampled, starting with the first one 
                TODO: Does it make sense to randomly choose a starting point?   
                TODO: if number_output_points > #points in pointcloud / n then it fails
        """
        # sample every nth element
        sampled_point_clouds = point_cloud[0::n]

        if number_output_points is not None:
            # take the middle number_output_points points from the point cloud
            number_points = sampled_point_clouds.shape[0]
            idxs = (number_points - number_output_points) / 2
            idxs = int(idxs)
            sampled_point_clouds = sampled_point_clouds[idxs:-idxs-1,:]

        return sampled_point_clouds
    
    @staticmethod
    def sample_random(point_cloud: torch.tensor, number_output_points: int):
        """
            sample random points from point cloud

            params:
                point_cloud: Tuple containing points of point cloud and their labels
                number_output_points: number of points of samples point cloud 
        """

        if len(point_cloud.shape) == 3:
            point_cloud = point_cloud.squeeze()

        points_shape = point_cloud.shape
            
        if type(point_cloud) == torch.Tensor:
            sampled_points = torch.zeros(number_output_points, points_shape[1])
        elif type(point_cloud) == np.ndarray:
            sampled_points = np.zeros((number_output_points, points_shape[1]))

        # TODO Use the default rng for this sampling
        # https://stackoverflow.com/questions/8505651/non-repetitive-random-number-in-numpy
        rng = np.random.default_rng()
        a = np.arange(point_cloud.shape[0])
        point_idxs = rng.choice(a, size=number_output_points, replace=False)
        reduced_pcl = point_cloud[point_idxs]
        #number_points_pcl = point_cloud.shape[0]
        #for i in range(number_output_points):
        #    # number_points_pcl - 1 because start and end are both included in this function
        #    idx = random.randint(0, number_points_pcl-1)
        #    sampled_points[i] = point_cloud[idx]

        return reduced_pcl 

    @staticmethod
    def farthest_point_sampling(point_cloud: Tuple[torch.tensor, torch.tensor], number_output_points: int, points_per_cycle: int=256):
        """
            samples always the point that is farthest away from the last point. Therefor first the distances will be calculated
            Currently distances will be calculated in 3 dimensional space (xyz) with euclidean distance
        """
        # only use x,y,z coordinates?
        # TODO does it make sense to take color information into consideration?
        n_cycles = int(number_output_points / points_per_cycle)
        n_points = point_cloud.shape[0]
        points_per_iteration = n_points/n_cycles
        selected_points = list()
        for i in range(n_cycles):
            start, end = i * points_per_iteration, i * points_per_iteration + points_per_iteration
            current_point_cloud = point_cloud[int(start):int(end)]
            distances = pairwise_distances(current_point_cloud[:, :3])
            
            (perm, _) = Sampler.get_greedy_perm(distances)
            sel_idxs = list(perm[0:int(number_output_points/n_cycles)])
            selected_points.append(current_point_cloud[sel_idxs])

        return torch.cat(selected_points, dim=0)

    @staticmethod
    def get_greedy_perm(D: np.ndarray):
        """
        A Naive O(N^2) algorithm to do furthest points sampling
        Source: https://gist.github.com/ctralie/128cc07da67f1d2e10ea470ee2d23fe8
        
        Parameters
        ----------
        D : ndarray (N, N) 
            An NxN distance matrix for points
        Return
        ------
        tuple (list, list) 
            (permutation (N-length array of indices), 
            lambdas (N-length array of insertion radii))
        """
    
        N = D.shape[0]
        #By default, takes the first point in the list to be the
        #first point in the permutation, but could be random
        perm = np.zeros(N, dtype=np.int)
        lambdas = np.zeros(N)
        ds = D[0, :]
        for i in range(1, N):
            idx = np.argmax(ds)
            perm[i] = idx
            lambdas[i] = ds[idx]
            ds = np.minimum(ds, D[idx, :])
        return (perm, lambdas)


