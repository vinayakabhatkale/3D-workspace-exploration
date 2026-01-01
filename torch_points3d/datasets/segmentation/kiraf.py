import os
import os.path as osp
import shutil
import json
from tqdm.auto import tqdm as tq
from itertools import repeat, product
import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import read_txt_array
import torch_geometric.transforms as T
from torch_points3d.core.data_transform.grid_transform import SaveOriginalPosId
from torch_points3d.metrics.bhatkale45_tracker import bhatkale45Tracker
from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties
from torch_points3d.datasets.segmentation.utils.viz_pcl import Visualizations

from pickle import dump, load



class bhatkale45(InMemoryDataset):
    url = None
    """
    category_ids = {
        "PCB": [1],
        "Adapter_Box": [2],
        "Tablar": [3],
        "Table": [4],
        "Floor": [5],
    }

    seg_classes = {
        "PCB": [0],
        "Adapter_Box": [1],
        "Tablar": [2],
        "Table": [3],
        "Background": [4],
    }
    """

    category_ids = {
        "Table": [0],
        "PCB": [1],
        "Tablar": [2],
        "Adapter_Box": [3],
        "Floor": [4],
    }

    seg_classes = {
        "Table": [0],
        "PCB": [1],
        "Tablar": [2],
        "Adapter_Box": [3],
        "Floor": [4],
    }
    
    def __init__(
        self,
        root,
        categories=None,
        include_normals=True,
        split="trainval",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        training=True,
        fold: int=None,
        camera: str='zivid'
    ):
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories
        self.is_training = training

        super(bhatkale45, self).__init__(
            root, transform, pre_transform, pre_filter)

        self.split = split
        self.fold = fold
        self.camera = camera

        if split == "train":
            #path = self.processed_paths[0]
            raw_path = self.processed_raw_paths[0]
        elif split == "val":
            #path = self.processed_paths[1]
            raw_path = self.processed_raw_paths[1]
        elif split == "test":
            #path = self.processed_paths[2]
            raw_path = self.processed_raw_paths[2]
        elif split == "train_val":
            #path = self.processed_paths[3]
            raw_path = self.processed_raw_paths[3]
        else:
            raise ValueError(
                (f"Split {split} found, but expected either " "train, val, trainval or test"))

        self.data, self.slices = self.load_data(raw_path)

        # We have perform a slighly optimzation on memory space of no pre-transform was used.
        # c.f self._process_filenames
        # if os.path.exists(raw_path):
        #     self.raw_data, self.raw_slices, _ = self.load_data(
        #         raw_path, include_normals)
        # else:
        #     self.get_raw_data = self.get

    def load_data(self, path):
        '''This function is used twice to load data for both raw and pre_transformed
        '''
        ds_list = torch.load(path)

        data_list = list()
        #viz = Visualizations()
        for data_point, label in ds_list:
            xyz = data_point[:, :3]
            features = data_point[:, 3:6]
            data = Data(x=features, pos=xyz, coords=xyz, y=label)
            #viz.visualize_point_cloud(xyz, label, from_hdf=False)
            data_list.append(data)

        data, slices = self.collate(data_list)
        return data, slices

    @property
    def raw_file_names(self):
        return list(self.category_ids.values()) + ["train_test_split"]

    @property
    def processed_raw_paths(self):
        ds_list = ["training", "valid", "test", "train_val"]
        if self.fold is not None:
            filepath = os.path.join(self.root, 'cross_val', str(self.fold))
        else:
            filepath = self.root
        processed_raw_paths = [os.path.join(filepath, "torch_training_data_{}.torchds".format(
            s)) for s in ds_list]

        return processed_raw_paths

    @property
    def processed_file_names(self):
        cats = "_".join([cat[:3].lower() for cat in self.categories])
        return [os.path.join("{}_{}.pt".format(cats, split)) for split in ["train", "val", "test", "trainval"]]

    def get_raw_data(self, idx, **kwargs):
        data = self.raw_data.__class__()

        if hasattr(self.raw_data, '__num_nodes__'):
            data.num_nodes = self.raw_data.__num_nodes__[idx]

        for key in self.raw_data.keys:
            item, slices = self.raw_data[key], self.raw_slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            # print(slices[idx], slices[idx + 1])
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.raw_data.__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]
        return data

    def _process_filenames(self, filenames):
        data_raw_list = []
        data_list = []
        categories_ids = [self.category_ids[cat] for cat in self.categories]
        cat_idx = {categories_ids[i]: i for i in range(len(categories_ids))}

        has_pre_transform = self.pre_transform is not None

        id_scan = -1
        for name in tq(filenames):
            cat = name.split(osp.sep)[0]
            if cat not in categories_ids:
                continue
            id_scan += 1
            data = read_txt_array(osp.join(self.raw_dir, name))
            pos = data[:, :3]
            x = data[:, 3:6]
            y = data[:, -1].type(torch.long)
            category = torch.ones(x.shape[0], dtype=torch.long) * cat_idx[cat]
            id_scan_tensor = torch.from_numpy(np.asarray([id_scan])).clone()
            data = Data(pos=pos, x=x, y=y, category=category,
                        id_scan=id_scan_tensor)
            data = SaveOriginalPosId()(data)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            data_raw_list.append(data.clone() if has_pre_transform else data)
            if has_pre_transform:
                data = self.pre_transform(data)
                data_list.append(data)
        if not has_pre_transform:
            return [], data_raw_list
        return data_raw_list, data_list

    def _save_data_list(self, datas, path_to_datas, save_bool=True):
        if save_bool:
            torch.save(self.collate(datas), path_to_datas)

    def _re_index_trainval(self, trainval):
        if len(trainval) == 0:
            return trainval
        train, val = trainval
        for v in val:
            v.id_scan += len(train)
        assert (train[-1].id_scan + 1 ==
                val[0].id_scan).item(), (train[-1].id_scan, val[0].id_scan)
        return train + val

    def __repr__(self):
        return "{}({}, categories={})".format(self.__class__.__name__, len(self), self.categories)


class bhatkale45Dataset(BaseDataset):
    """ Wrapper around bhatkale45 that creates train and test datasets.

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - category: List of categories or All
            - normal: bool, include normals or not
            - pre_transforms
            - train_transforms
            - test_transforms
            - val_transforms
    """

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        self.dataset_opt = dataset_opt
        camera = dataset_opt.camera

        if dataset_opt['include_real_data_to_training'] is True:
            self._data_path = os.path.join(self._data_path, 'real_in_train')
        else:
            self._data_path = os.path.join(self._data_path, 'no_real_in_train')

        self._data_path = os.path.join(self._data_path, camera)

        if hasattr(dataset_opt, 'fold') and camera == 'zivid':
            fold = getattr(dataset_opt, 'fold')
        else:
            fold = None

        # generate training data and data loader
        self.train_dataset = bhatkale45(
            self._data_path,
            camera=camera,
            #self._category,
            include_normals=dataset_opt.normal,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            fold=fold
        )

        self.val_dataset = bhatkale45(
            self._data_path,
            include_normals=dataset_opt.normal,
            split="val",
            camera=camera,
            pre_transform=self.pre_transform,
            transform=self.val_transform,
            fold=fold
        )

        self.test_dataset = bhatkale45(
            self._data_path,
            camera=camera,
            include_normals=dataset_opt.normal,
            split="test",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
        )
        self._categories = self.train_dataset.categories

    @property  # type: ignore
    @save_used_properties
    def class_to_segments(self):
        classes_to_segment = {}
        for key in self._categories:
            classes_to_segment[key] = bhatkale45.seg_classes[key]
        return classes_to_segment

    @property
    def is_hierarchical(self):
        return len(self._categories) > 1

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return bhatkale45Tracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log, dataset_opt=self.dataset_opt)
