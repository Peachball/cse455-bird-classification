import torch
from torchvision.io import read_image
from torchvision import transforms
import shutil
from pathlib import Path
import csv
import os
from tqdm import tqdm

ZIPPED_DATA_PATH = Path('birds21sP.zip')
DATA_DIR = Path('data/')


def unzip_zipped_data():
    print('Unzipping data...')
    shutil.unpack_archive(ZIPPED_DATA_PATH, DATA_DIR)


class BirdDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_type='train', augmentation=None):
        super().__init__()
        self.dataset_type = dataset_type
        self.examples = []
        self.augmentation = augmentation
        if dataset_type in ['train', 'validate']:
            with (DATA_DIR / 'birds' / 'labels.csv').open() as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    fname = f'train/{row[1]}/{row[0]}'
                    self.examples.append((fname, int(row[1])))
                self.validate_ind = int(len(self.examples) * 0.8)
        elif dataset_type == 'test':
            with (DATA_DIR / 'birds' / 'sample.csv').open() as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    root_fname = os.path.basename(row[0])
                    fname = f'test/0/{root_fname}'
                    self.examples.append((fname, int(row[1])))
        else:
            raise ValueError(f'Unknown type of dataset: {dataset_type}')

    def __getitem__(self, key):
        if self.dataset_type == 'validate':
            key += self.validate_ind
        fname, label = self.examples[key]
        image = read_image(str(DATA_DIR / 'birds' / fname))
        aug = transforms.Resize((224, 224))
        if self.augmentation is not None:
            aug = transforms.Compose([self.augmentation, aug])
        return {'image': aug(image), 'label': label}

    def __len__(self):
        if self.dataset_type == 'validate':
            return len(self.examples) - self.validate_ind
        if self.dataset_type == 'train':
            return self.validate_ind
        return len(self.examples)


def get_data(batch_size=32, augmentation=None):
    if not DATA_DIR.exists():
        unzip_zipped_data()

    train_dir = DATA_DIR / 'birds' / 'train'

    if any(train_dir.iterdir()) == 0:
        # empty train directory, meaning that data has not been extracted yet
        unzip_zipped_data()

    train_dataset = BirdDataset(dataset_type='train',
                                augmentation=augmentation)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=4)
    validate_dataset = BirdDataset(dataset_type='validate')
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=4)
    test_dataset = BirdDataset(dataset_type='test')
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              num_workers=4)
    return {
        'train': train_loader,
        'validate': validate_loader,
        'test': test_loader
    }


def train():
    data = get_data()
    total = torch.zeros((3,))
    for sample in tqdm(data['train']):
        total += sample['image'].type(torch.float32).mean(dim=[0, 2, 3])
    sample_mean = total / len(data['train'])


if __name__ == '__main__':
    train()
