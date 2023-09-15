from torch.utils.data import Dataset
import torch

class TorchDataloader(Dataset):
    """
    This class allows you to load datasets for a PyTorch framework (probably from numpy objects).
    Takes dataset object and performs operations defined with model (preprocessing, transformation).
    """

    def __init__(self,
                 dataset=None,
                 preprocess=None,
                 transform=None,
                 steps_per_epoch=None,
                 **kwargs):

        """Initialize.
        Args:
            dataset: The 2D image dataset class.
            preprocess: The model's pre-process method.
            transform: The model's transform method.
            steps_per_epoch: The number of steps per epoch that indicates the batches of samples to train. If it is None, then the step number will be the number of samples in the data.
        Returns:
            class: The corresponding class.
        """
        self.dataset = dataset
        self.preprocess = preprocess
        self.steps_per_epoch = steps_per_epoch
        self.transform = transform

    def __getitem__(self, index):

        """Returns the item at index position (idx)."""
        index = index % len(self.dataset)
        
        attr = self.dataset.get_attr(index)
        # If datum just preprocessed
        if self.preprocess:
            data = self.preprocess(self.dataset.get_data(index), attr)
        # Nothing happens
        else:
            data = self.dataset.get_data(index)
        # Transform datum if it's required
        if self.transform is not None:
            data = self.transform(data, attr)

        return {'data': data, 'attr': attr}

    def __len__(self):
        
        """Returns the number of steps for an epoch."""
        if self.steps_per_epoch is not None:
            steps_per_epoch = self.steps_per_epoch
        else:
            steps_per_epoch = len(self.dataset)
            
        return steps_per_epoch

class ObjectDetectBatch(object):

    def __init__(self, batch):
        """Initialize.
        Args:
            batches: A batch of data
        Returns:
            class: The corresponding class.
        """
        self.images = []
        self.labels = []
        self.boxes = []
        self.centers = []
        self.radius = []
        self.directions = []
        self.attr = []

        for batch_item in batch:

            self.attr.append(batch_item['attr'])
            
            data = batch_item['data']

            self.images.append(torch.tensor(data['image'], dtype=torch.float32))
            self.labels.append(torch.tensor(data['labels'], dtype=torch.int32) if 'labels' in data else None)
            
            if len(data.get('boxes', [])) > 0:

                self.boxes.append(
                    torch.tensor(data['boxes'], dtype=torch.float32) if 'boxes' in data else None)
                self.centers.append(
                    torch.tensor(data['centers'], dtype=torch.float32) if 'centers' in data else None)
                self.radius.append(
                    torch.tensor(data['radius'], dtype=torch.float32) if 'radius' in data else None)
                self.directions.append(
                    torch.tensor(data['directions'], dtype=torch.float32) if 'directions' in data else None)

            else:
                self.boxes.append(torch.zeros((0, 4)))
                self.centers.append(torch.zeros((0, 2)))
                self.radius.append(torch.zeros((0, 1)))
                self.directions.append(torch.zeros((0, 1)))

        self.images = torch.stack(self.images,0).permute((0,3,1,2))    

    def pin_memory(self):

        self.images = self.images.pin_memory()

        for i in range(len(self.boxes)):
            if self.labels[i] is not None:
                self.labels[i] = self.labels[i].pin_memory()
            if self.boxes[i] is not None:
                self.boxes[i] = self.boxes[i].pin_memory()
            if self.centers[i] is not None:
                self.centers[i] = self.centers[i].pin_memory()
            if self.radius[i] is not None:
                self.radius[i] = self.radius[i].pin_memory()
            if self.directions[i] is not None:
                self.directions[i] = self.directions[i].pin_memory()

        return self

    def to(self, device):

        self.images = self.images.to(device)

        for i in range(len(self.boxes)):

            if self.labels[i] is not None: self.labels[i] = self.labels[i].to(device)
            if self.boxes[i] is not None: self.boxes[i] = self.boxes[i].to(device)
            if self.centers[i] is not None: self.centers[i] = self.centers[i].to(device)
            if self.radius[i] is not None: self.radius[i] = self.radius[i].to(device)
            if self.directions[i] is not None: self.directions[i] = self.directions[i].to(device)

    
class ConcatBatcher(object):
    """
        ConcatBatcher selects batch generator according to selected model.
        Provides function collate_fn, which can be provided to torch.DataLoader
        to define how custom batching should be executed.
    """

    def __init__(self, device):

        """Initialize.
        Args:
            device: torch device 'gpu' or 'cpu'
        Returns:
            class: The corresponding class.
        """
        super(ConcatBatcher, self).__init__()
        self.device = device

    def collate_fn(self, batches):
        """Collate function called by original PyTorch dataloader.
        Args:
            batches: a batch of data
        Returns:
            class: the batched result
        """

        batching_result = ObjectDetectBatch(batches)
        return batching_result