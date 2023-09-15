import numpy as np
import glob, os
from pathlib import Path
import logging
import cv2
import json

from dataset.base_dataset import BaseDataset, BaseDatasetSplit

log = logging.getLogger(__name__)

class ImageSplit(BaseDatasetSplit):
    """This class is used to create a custom dataset split.
    Initialize the class.
    Args:
        dataset: The dataset to split.
        split: A string identifying the dataset split that is usually one of
        'training', 'test', 'validation', or 'all'.
        **kwargs: The configuration of the model as keyword arguments.
    Returns:
        A dataset split object providing the requested subset of the data.
    """

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)

        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} images for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)


    def get_data(self, idx):

        image_path = self.path_list[idx]
        anot_path = image_path.split('.png')[0] + '.json'

        img = cv2.imread(image_path)

        with open(anot_path, 'r') as j:
            annotation = json.loads(j.read())

        labels = []
        boxes = []

        for item in annotation:
            labels.append(item['class'])
            boxes.append(item['box'])
   
        labels = np.asarray(labels).reshape(-1)
        boxes = np.asarray(boxes).reshape(-1,4).astype(np.float32)
     
        return {'image': img, 'labels': labels, 'boxes': boxes}

    def get_attr(self, idx):

        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.npy', '')

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}

        return attr
    
class Dataset(BaseDataset):

    def __init__(self,
                 dataset_path,
                 **kwargs):

        super().__init__(dataset_path=dataset_path,
                         **kwargs)

        cfg = self.cfg
        self.dataset_path = cfg.dataset_path 

        self.train_dir = str(Path(self.dataset_path) / 'training')
        self.val_dir = str(Path(self.dataset_path) / 'validation')
        self.test_dir = str(Path(self.dataset_path) / 'testing')
        
        self.train_files = [f for f in glob.glob(self.train_dir + "/*.png")]
        self.val_files = [f for f in glob.glob(self.val_dir + "/*.png")]
        self.test_files = [f for f in glob.glob(self.test_dir + "/*.png")]
        
    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.
        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            1: 'License plate'
        }

        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.
        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return ImageSplit(self, split=split)

    def get_split_list(self, split):
        """Returns a dataset split.
        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'testing', 'validation'.
        Returns:
            A dataset split object providing the requested subset of the data.
        Raises:
             ValueError: Indicates that the split name passed is incorrect. The
             split name should be one of 'training', 'testing', 'validation'.
        """
        
        if split in ['test', 'testing']:
            return self.test_files
        elif split in ['val', 'validation']:
            self.rng.shuffle(self.val_files)
            return self.val_files
        elif split in ['train', 'training']:
            self.rng.shuffle(self.train_files)
            return self.train_files
        else:
            raise ValueError("Invalid split {}".format(split))