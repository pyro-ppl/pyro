import os

from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets


EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXAMPLES_DIR, 'data')
RESULTS_DIR = os.path.join(EXAMPLES_DIR, 'results')


def get_data_loader(dataset_name,
                    batch_size=1,
                    dataset_transforms=None,
                    is_training_set=True,
                    shuffle=True):
    if not dataset_transforms:
        dataset_transforms = []
    trans = transforms.Compose([transforms.ToTensor()] + dataset_transforms)
    dataset = getattr(datasets, dataset_name)
    return DataLoader(
        dataset(root=DATA_DIR,
                train=is_training_set,
                transform=trans,
                download=True),
        batch_size=batch_size,
        shuffle=shuffle
    )
