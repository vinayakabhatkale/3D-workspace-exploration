from scipy import spatial
import numpy as np
from data.compare_ds import viz_point_cloud
from Sampler import Sampler

def main():
    possible_colors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1]
            ]

    basic = np.load('./data/synth/training_data0.npy')
    basic = basic[:, :, :4]
    chosen_pcl = 1
    xyz = basic[:,:, 1:][chosen_pcl]
    colors = np.zeros(shape=xyz.shape)

    sampled_points = Sampler.sample_random(point_cloud=basic[chosen_pcl], number_output_points=4096) 
    new_xyz = sampled_points[:, 1:]
    new_label = sampled_points[:, 0]
    tree = spatial.KDTree(xyz)

    results = tree.query_ball_point(new_xyz, r=0.03)
    for idx, result in enumerate(results):
        label = new_label[idx]
        colors[result] = possible_colors[int(label)]
    viz_point_cloud(xyz, colors)

if __name__ == '__main__':
    main()
