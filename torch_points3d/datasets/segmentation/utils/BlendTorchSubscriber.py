import blendtorch.btt as btt
import numpy as np
import h5py
import os
import torch

from Sampler import Sampler

class BlendTorchSubscriber:
    def __init__(self, launch_info_fp: str, max_items: int, store_fp: str=None, verbose: bool=True, dtype: np.dtype=np.float32, max_pcls_per_file: int=256, instances: bool=False):
        """
            params:
                launch_info_fp: file path of launch_info file
                max_items: maximum number of items to receive
                store_fp: file path to store received data
                dtype: data type for received data
        """
        self._launch_info_fp = launch_info_fp
        self._max_items = max_items
        self._store_fp = store_fp
        self._verbose = verbose
        self._dtype = dtype
        self._max_pcls_per_file = max_pcls_per_file
        self._instances = instances

    def receive_data(self) -> list:
        """
            receive point clouds
        """
        list_recv_pcls = list()
        least_points = 1e6

        launch_info = btt.LaunchInfo.load_json(self._launch_info_fp)
        ds = btt.RemoteIterableDataset(launch_info.addresses['DATA'], max_items=self._max_items)
        try:
            pcl_dict = next(iter(ds))['pcl']
        except AssertionError:
            print('Assertion catched.. exiting now')
            exit()

        basic_pcl, colors = self._process_pcl_dict(pcl_dict)

        if basic_pcl is not None:
            least_points = basic_pcl.shape[1]
            list_recv_pcls.append((basic_pcl, colors))

        for item in range(self._max_items - 1):
            try:
                data_received = next(iter(ds))['pcl']
            except AssertionError:
                break

            basic_pcl, colors = self._process_pcl_dict(data_received)
            if basic_pcl is None:
                # no valid point cloud was received
                continue
            list_recv_pcls.append((basic_pcl, colors))
            if basic_pcl.shape[1] < least_points:
                least_points = basic_pcl.shape[1]
            if item % 50 == 0:
                if self._verbose:
                    print(f'{item} pcls received')

        pcls = None
        pcls_colors = None
        for basic_pcl, colors in list_recv_pcls:
            pcl = np.concatenate((basic_pcl, colors), axis=2)
            pcls = self._reduce_pcl(pcls, pcl, number_points=least_points)

        if self._store_fp is not None:
            self._store_data(pcls, self._store_fp, 'basic')
            
        return pcls, pcls_colors

    def _reduce_pcl(self, current_pcl: np.ndarray, new_data: np.ndarray, number_points: int) -> np.ndarray:
        """
            reduce given pcl so that all point clouds have the same amount of points
        """
        if current_pcl is None:
            current_pcl = Sampler.sample_point_cloud(new_data, number_points)
            current_pcl = np.expand_dims(current_pcl, axis=0)
        else:
            next_pcl = Sampler.sample_point_cloud(new_data, number_points)
            next_pcl = np.expand_dims(next_pcl, axis=0)
            current_pcl = np.concatenate((current_pcl, next_pcl), axis=0)
        return current_pcl

    def _process_pcl_dict(self, pcl_dict: dict):
        """

        """
        try:
            colors = np.expand_dims(pcl_dict['rgb'], axis=0).astype(self._dtype)
            points = np.expand_dims(pcl_dict['xyz'], axis=0).astype(self._dtype)
            labels = np.expand_dims(pcl_dict['label'], axis=0).astype(np.int8)
            if self._instances:
                instances = np.expand_dims(pcl_dict['instance'], axis=0).astype(np.int8)
                basic_pcl = np.concatenate((labels, instances, points), axis=2)
            else:
                basic_pcl = np.concatenate((labels, points), axis=2)
        except KeyError:
            basic_pcl, colors = None, None
        return basic_pcl, colors 

    def _store_data(self, data: list, filename: str, post_fix: str) -> None:
        """
            store data as numpy file
            params:
                data: data to store
                filename: path where data should be stored


        """ 
        if self._verbose:
            print(f'Storing received data in {filename + post_fix}')

        base_post_fix = post_fix
        for i in range(0, data.shape[0], self._max_pcls_per_file):
            post_fix = base_post_fix + str(i)
            count = 1
            while os.path.exists(filename + post_fix + '.npy'):
                post_fix = base_post_fix + str(i+count)
                count += 1
            np.save(filename + post_fix, data[i:i+self._max_pcls_per_file,:,:])


if __name__ == '__main__':
    # number_frames * number_episodes * number_scenes
    ITEMS_TO_RECEIVE = 20*2*1
    subscriber = BlendTorchSubscriber(
            launch_info_fp='./launch_info.json', max_items=ITEMS_TO_RECEIVE,
            store_fp='./data/sb_training_data', max_pcls_per_file=64, instances=False)

    subscriber.receive_data()
