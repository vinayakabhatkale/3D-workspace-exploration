from venv import create
import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch
import glob
import pickle
import os
import pandas as pd

# imports for cross validation
from sklearn.model_selection import KFold
import copy


# depends on which 
try:
    from torch_points3d.datasets.segmentation.utils.bhatkale45Dataset import bhatkale45Dataset
    from torch_points3d.datasets.segmentation.utils.Sampler import Sampler
except:
    from Sampler import Sampler
    from bhatkale45Dataset import bhatkale45Dataset
    from viz_pcl import Visualizations
    import matplotlib.pyplot as plt

import yaml
from pathlib import Path

class DataProvider:
    def __init__(self, config: dict, data: np.ndarray = None, pcl_color: np.ndarray = None, pcl_intensity: np.ndarray = None, verbose: bool = True):
        """
        Either data or filepath have to be given
            params:
                data: numpy array containing the basic pcl (label, x, y, z)
                pcl_color: numpy array containing the colors that belong to basic pcl
                pcl_intensity: numpy array containing the intensities that belong to basic pcl
                filepath: filepath to file containing the dataset 
        """
        self._verbose = verbose

        self.config = config
        self.include_real_data_to_training = config['include_real_data_to_training']
        if data is not None:
            self._data = data
            self._pcl_color = pcl_color
            self._pcl_intensities = pcl_intensity
            self.filepath = None
        elif config['filepath'] is not None:
            self.filepath = config['filepath']
            self._data = self._load_data_from_np_file(os.path.join(self.filepath, config['camera'] if 'camera' in config else '', 'synth'))
            self._real_data = self._load_data_from_np_file(os.path.join(self.filepath, 'real')) if config['include_real_data_to_training'] is True else None
        else:
            raise Exception('No source for data given!')

        # float16 would work for tensors but torch.set_default_dtype works only
        # with float32 and float64 
        self.dtype = torch.float 
        self.ds_name = dataset_opt['camera']

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def _load_data_from_np_file(self, filepath: str):
        n_points = list()
        pcls = list()
        filepath = filepath.replace('zivid/', '')
        file_list = glob.glob(filepath + f'/*.npy')
        n_points_required = self.config['number_points']
        for file_ in sorted(file_list):
            data = np.load(file_)
            if data.shape[1] < n_points_required:
                continue
            pcls.append(data)
            n_points.append(data.shape[1])

            # if small dataset for testing stuff in training are necessary
            #break

        n_points_required = min(n_points)
        list_pcls = list()

        # attention:
        # only sampled so that all pcls have the same number 
        #viz = Visualizations()
        for pcl in pcls:
            for x in pcl:
              #viz.visualize_point_cloud(x[:, 1:4], x[:, 4:7])
              torch_pcls = torch.tensor(pcl)
            if not torch_pcls.shape[1] == n_points_required:
                torch_pcls = Sampler.sample_point_clouds(torch_pcls, number_output_points=n_points_required)
            list_pcls.append(torch_pcls)

        data = torch.cat(list_pcls, dim=0)
        return data
   
    def _get_ds_for_model(self, config: dict) -> bhatkale45Dataset:
        """
            get dataset with specified channels
            params:
                config: config contains the configuration for dataset
        """

        base_path = os.path.join(config['filepath_sampled_pcl'], str(config['number_points']))
        pre_fix = os.path.join(base_path, 'training_data_sampled')

        fp_sampled_pcl = f"{pre_fix}_{config['sampling_strategy']}.pkl"

        # check if data is already available
        #if os.path.exists(fp_sampled_pcl) is True:
        #    with open(fp_sampled_pcl, "rb") as f:
        #        self.x, self.y = pickle.load(f)
        #    return bhatkale45Dataset(self.x, self.y)

        self._get_ds_info(self._data, dataset='synth') 
        if config['include_real_data_to_training'] is True:
            self._get_ds_info(self._real_data, dataset='real') 
        data = self.sample_point_clouds(
            self._data, number_output_points=config['number_points'], n=config['n'], n_pcls=config['n_pcls'])

        real_data = self.sample_point_clouds(
                self._real_data, number_output_points=config['number_points'], n=config['n'], n_pcls=config['n_pcls']) if config['include_real_data_to_training'] is True else None

        self.x = data[:,:,1:]
        self.y = data[:,:,0].type(torch.LongTensor)

        self.x_real = real_data[:,:,1:] if config['include_real_data_to_training'] is True else None
        self.y_real = real_data[:,:,0].type(torch.LongTensor) if config['include_real_data_to_training'] is True else None

        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)

        with open(fp_sampled_pcl, "wb") as f:
            pickle.dump((self.x, self.y), f)

        # we created a new torch.tensor containing the data from the following
        # tensors, so we can delete them
        del self._data
        del self._real_data

        return bhatkale45Dataset(self.x, self.y), bhatkale45Dataset(self.x_real, self.y_real)

    def _get_ds_info(self, pcls: np.ndarray, dataset: str='synth') -> None:
        """
            Get information about the dataset like distribution of classes of number of point clouds ...
            params:
                pcls: np.ndarray containing the data
                dataset: which dataset is investigated. Possible strings are synth, real, merged (contains real and synthetic data)
        """

        title_prefix = 'Class distribution for'
        labels = ['Table', 'PCB','Tablar','Adapter_Box','Floor']
        if dataset == 'synth':
            title = f'{title_prefix} synthetic dataset'
        elif dataset == 'real':
            title = f'{title_prefix} real dataset'
        elif dataset == 'merged':
            title = f'{title_prefix} for merged dataset'
        else:
            print('Not supported dataset string, no pdf will be generated')
            return

        print(f'The dataset consists of {pcls.shape[0]} point clouds')
        y = pcls[:,:,0]
        distribution_info = np.unique(y, return_counts=True)
        sum_points = np.sum(distribution_info[1])
        print(f'The dataset has {sum_points} points')
        percentages = list()
        for class_, count in zip(np.nditer(distribution_info[0]), np.nditer(distribution_info[1])):
            class_percentage = 100 * (count/sum_points)
            percentages.append(class_percentage)
            print(f'The dataset has {count} points with class {labels[int(class_)]}, that is {class_percentage} % of the dataset')

        df = pd.DataFrame.from_dict({self.ds_name: percentages}, orient='index')
        df.to_csv('class_percentages_cameras', mode='a', header=False)

        #fig1, ax1 = plt.subplots()
        #ax1.pie(percentages, labels=labels, autopct='%1.1f%%',
            #shadow=False, startangle=90)
    
        #ax1.axis('equal')
        #ax1.set_title(title)
        #filename = 'pie' 
                
        #plt.savefig(f'./{filename}_{dataset}.pdf')

    def get_device(self):
        return self.device

    def sample_point_clouds(self, pcls: torch.tensor, number_output_points: int = 1024, n: int=2, n_pcls: int=1):
        """
            sample all given point clouds
            params:
                pcls: all point clouds 
                labels: labels of points in point clouds
                number_output_points: number of output points in each point cloud 
        """
        n_resulting_pcls = pcls.shape[0] * n_pcls
        sampled_point_clouds = torch.zeros(
            n_resulting_pcls, number_output_points, pcls[0].shape[1], dtype=self.dtype)
        for i in range(pcls.shape[0]):
            pcl = pcls[i]

            if self.config['sampling_strategy'] == "greedy_farthest_point":
                # point clouds are too big, at least when using farthest point sampling
                # the calculations for pairwise distances would take forever 
                # so first take only every nth element and then calculate the pairwise distances
                pcl = Sampler.sample_point_cloud(pcl, sampling_strategy="every_nth_element", n=4, number_output_points=None)
            
            for n_pcl in range(n_pcls):
                store_idx = i * n_pcls + n_pcl
                sampled_point_clouds[store_idx] = Sampler.sample_point_cloud(
                    pcl, number_output_points=number_output_points, sampling_strategy=self.config['sampling_strategy'])

        return sampled_point_clouds

    def get_train_test_set(self, test_size: float = 0.15, config: dict= None, valid_split: float = None, no_split: bool=False):
        """
            Returns a train, (validation) and a test set
            params:
                test_size: size of test set in percent -> values possible [0:1]
                config: DataConfig that contains information which data should be in the dataset
                valid_split: if valid_split is None, None is returned for valid_ds, if it is not None
                part of the train_ds will be returned as valid_ds

            return: 
                bhatkale45Dataset, (None, bhatkale45Dataset), bhatkale45Dataset
        """
        if config is None:
            config = self.config

        valid_ds = None
        self.ds, self.ds_real = self._get_ds_for_model(config)

        if no_split is True:
            test_ds = Subset(self.ds, np.arange(len(self.ds)))
            return test_ds, test_ds, None, test_ds

        if self.include_real_data_to_training is True:
            self.ds.extend(self.ds_real.x, self.ds_real.labels)
            if test_size < 0. or test_size > 1.:
                raise Exception('Invalid value for test_size')
            train_idx, test_idx = train_test_split(
                list(range(len(self.ds))), test_size=test_size, random_state=0)
            train_ds = Subset(self.ds, train_idx)
            test_ds = Subset(self.ds, test_idx)

            # create a dataset consisting of all training data if cross validation should
            # be used
            train_val_x, train_val_y = self.ds[train_idx]
            train_val_ds = bhatkale45Dataset(train_val_x, train_val_y)

            if valid_split is not None:
                train_idx, valid_idx = train_test_split(
                    train_idx, test_size=valid_split, random_state=0)
                train_ds = Subset(self.ds, train_idx)
                valid_ds = Subset(self.ds, valid_idx)
        else:
            train_idx, val_idx = train_test_split(
                list(range(len(self.ds))), test_size=test_size, random_state=0)
            train_ds = Subset(self.ds, train_idx)
            valid_ds = Subset(self.ds, val_idx)
            train_val_ds = copy.deepcopy(self.ds)

            test_idx = list(range(len(self.ds_real)))  #Here, the real dataset is used in the else condition, which should not be there. FIX!
            test_ds = Subset(self.ds_real, test_idx) #Here, the real dataset is used in the else condition, which should not be there. FIX!

        return train_ds, valid_ds, train_val_ds, test_ds

    def get_num_classes(self):
        if self.ds is not None:
            return self.ds.num_classes
        else:
            raise Exception(
                'DataProvider.ds not initialized. Call get_train_test_set first')


def create_ds_from_subset(ds: Subset):
    data_list = list()
    # need this because Subset creates a dataset that is as big as the whole set but stores the indices
    # needed to create the desired Subset
    for idx in ds.indices:
        data_point = ds.dataset.x[idx]
        data_list.append((data_point, ds.dataset.labels[idx]))

    return data_list

def create_cross_validation_datasets(train_val_dataset: bhatkale45Dataset, dataset_opt: dict):
    k=2 #The K can be taken from config file, this can be changed. 
    splits=KFold(n_splits=k,shuffle=True,random_state=42)
    
    if dataset_opt['include_real_data_to_training'] is True:
        filepath = os.path.join(dataset_opt['filepath_torch'], 'real_in_train')
    else:
        filepath = os.path.join(dataset_opt['filepath_torch'], 'no_real_in_train')

    base_path = os.path.join(filepath, 'cross_val')

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_val_dataset)))):
        train_ds = Subset(train_val_dataset, train_idx)
        valid_ds = Subset(train_val_dataset, val_idx)
        train_list = create_ds_from_subset(train_ds)
        valid_list = create_ds_from_subset(valid_ds)

        filepath = os.path.join(base_path, str(fold))
        os.makedirs(filepath, exist_ok=True)
        torch.save(train_list, os.path.join(filepath, f'./torch_training_data_training.torchds'))
        torch.save(valid_list, os.path.join(filepath, f'./torch_training_data_valid.torchds'))


if __name__ == '__main__':
    #dataset_opt= yaml.safe_load(Path('/home/developer/deepviewaggregation/config/data/segmentation/bhatkale45.yaml').read_text())
    dataset_opt= yaml.safe_load(Path('/home/developer/deepviewaggregation/conf/data/segmentation/bhatkale45.yaml').read_text())
    dp = DataProvider(dataset_opt)

    train_dataset, valid_dataset, train_val_dataset, test_dataset = dp.get_train_test_set(config=dataset_opt, valid_split=0.2, no_split=False)

    create_cross_validation_datasets(train_val_dataset=train_val_dataset, dataset_opt=dataset_opt)

    ds_dict = dict()
    list_ds = ['training', 'valid', 'test']
    list_bhatkale45_ds = [train_dataset, valid_dataset, test_dataset]
    for ds_name, dataset in zip(list_ds, list_bhatkale45_ds):
        if dataset is None:
            continue
        data_list = create_ds_from_subset(dataset)
        ds_dict[ds_name] = data_list

    for ds_name in list_ds:
        if dp.include_real_data_to_training is True:
            filepath = os.path.join(dataset_opt['filepath_torch'], 'real_in_train')
        else:
            filepath = os.path.join(dataset_opt['filepath_torch'], 'no_real_in_train')

        filepath = os.path.join(filepath, dataset_opt['camera'])

        os.makedirs(filepath, exist_ok=True)
        
        if ds_name in ds_dict:
            torch.save(ds_dict[ds_name], os.path.join(filepath, f'./torch_training_data_{ds_name}.torchds'))

