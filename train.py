import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
import shutil
from pathlib import Path
import csv
import cv2
import os
import argparse
from tqdm import tqdm
from constants import *

device = torch.device('cuda')
parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--logDir', default='logs/', help='output directory for log files')
parser.add_argument('--saveDir', default='./', help='output directory for saved model')

args = parser.parse_args()

def unzip_zipped_data():
    print('Unzipping data...')
    shutil.unpack_archive(ZIPPED_DATA_PATH, DATA_DIR)


class BirdDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_type='train',
                 augmentation=None,
                 use_resized=False,
                 normalize=None):
        super().__init__()
        self.dataset_type = dataset_type
        self.examples = []
        self.augmentation = augmentation
        self.t = transforms.ToTensor()
        self.normalize = normalize
        default_bird_dir = DATA_DIR / 'birds'
        resized_dir = DATA_DIR / 'birds_resized'
        train_ratio = 0.9
        if dataset_type in ['train', 'validate']:
            with (DATA_DIR / 'birds' / 'labels.csv').open() as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    if use_resized:
                        fname = resized_dir / row[0]
                    else:
                        fname = default_bird_dir / f'train/{row[1]}/{row[0]}'
                    self.examples.append((fname, int(row[1])))
                self.validate_ind = int(len(self.examples) * train_ratio)
        elif dataset_type == 'test':
            with (DATA_DIR / 'birds' / 'sample.csv').open() as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    root_fname = os.path.basename(row[0])
                    if use_resized:
                        fname = resized_dir / row[0]
                    else:
                        fname = default_bird_dir / f'test/0/{root_fname}'
                    self.examples.append((fname, int(row[1])))
        else:
            raise ValueError(f'Unknown type of dataset: {dataset_type}')

    def __getitem__(self, key):
        if self.dataset_type == 'validate':
            key += self.validate_ind
        fname, label = self.examples[key]

        # pytorch will throw an error when reading
        # uncompressed training data if this isn't here
        im = cv2.imread(str(fname))
        assert im.shape[2] == 3
        image = self.t(im).float() / 255.0
        if image.shape[0] == 1:
            image = image.expand(3, -1, -1)

        aug = transforms.Compose([
            transforms.Resize((224, 224)),
            self.normalize
        ])

        if self.augmentation is not None:
            aug = transforms.Compose([self.augmentation, aug])
        return {'image': aug(image), 'label': label}

    def __len__(self):
        if self.dataset_type == 'validate':
            return len(self.examples) - self.validate_ind
        if self.dataset_type == 'train':
            return self.validate_ind
        return len(self.examples)


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(pretrained=True)
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, 555)
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)


def get_data(batch_size=32, augmentation=None, normalize=None):
    if not DATA_DIR.exists():
        unzip_zipped_data()

    train_dir = DATA_DIR / 'birds' / 'train'

    if any(train_dir.iterdir()) == 0:
        # empty train directory, meaning that data has not been extracted yet
        unzip_zipped_data()

    workers = 1
    train_dataset = BirdDataset(dataset_type='train',
                                augmentation=augmentation,
                                normalize=normalize)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=workers)
    validate_dataset = BirdDataset(dataset_type='validate',
                                   normalize=normalize)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=workers)
    test_dataset = BirdDataset(dataset_type='test',
                               normalize=normalize)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              num_workers=workers)
    return {
        'train': train_loader,
        'validate': validate_loader,
        'test': test_loader
    }


def compute_accuracy(model, dataloader):
    with torch.no_grad():
        total_correct = 0
        total = 0
        for sample in tqdm(dataloader):
            pred = model(sample['image'].to(device))
            labels = pred.argmax(dim=1)
            total_correct += (labels == sample['label'].to(device)).sum()
            total += labels.shape[0]
        return total_correct.item() / total


def train(modelname='test', epochs=10):
    log_dir = Path(f'{args.logDir}/{modelname}')

    save_path = Path(f'{args.saveDir}/{modelname}.pt')
    normalize = transforms.Compose([
        # Statistics are based off ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    data = get_data(augmentation=None, normalize=normalize)
    model = CNNModel().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    sw = SummaryWriter(log_dir=log_dir)
    global_step = 0

    if save_path.exists():
        print('Loading previous model')
        info = torch.load(save_path)
        global_step = info['global_step']
        optim.load_state_dict(info['optim'])
        model.load_state_dict(info['model'])

    model.train()
    for _ in range(epochs):
        print('Computing validate accuracy...')
        model.eval()
        validate_acc = compute_accuracy(model, data['validate'])
        sw.add_scalar('validate/acc', validate_acc, global_step)

        print('Training')
        model.train()
        for sample in tqdm(data['train']):
            optim.zero_grad()
            pred = model(sample['image'].to(device))
            err = loss_fn(pred, sample['label'].to(device))
            err.backward()

            optim.step()
            global_step += 1

            sw.add_scalar('train/err', err, global_step)

        print('Saving...')
        info = {
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'global_step': global_step
                }
        torch.save(info, save_path)



if __name__ == '__main__':
    train()
