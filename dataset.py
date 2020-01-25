import os
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import csv
import random

class DA(data.Dataset):
    def __init__(self, dir, name, img_size, train, real_val=False):
        self.name = name
        self.labels = []
        self.fnames = []
        self.dir = dir
        self.train = train
        self.image_size = img_size
        self.real_val = real_val
        self.transform = transforms.Compose([

            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        ])

        if self.train:
            file = os.path.join(self.dir, self.name, '{}_train.csv'.format(self.name))
        else:
            file = os.path.join(self.dir, self.name,  '{}_test.csv'.format(self.name))

        with open(file) as f:
            csv_reader = csv.reader(f, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    self.fnames.append(row[0])
                    self.labels.append(int(row[1]))
                    line_count += 1
                else:
                    line_count += 1

        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        l = self.labels[idx]
        img = Image.open(os.path.join(self.dir + fname)).convert('RGB')
        if not self.real_val:
            if self.train:
                if random.random() < 0.5:
                    t = transforms.RandomAffine(degrees=(-20,20))
                    img = t(img)

        img = self.transform(img)


        return img, l

    def __len__(self):
        return self.num_samples

class DA_test(data.Dataset):
    def __init__(self, dir, img_size):

        self.dir = dir
        self.image_size = img_size
        self.transform = transforms.Compose([

            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        ])

        self.fnames = os.listdir(self.dir)

        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.dir, fname)).convert('RGB')
        img = self.transform(img)

        return img

    def __len__(self):
        return self.num_samples

# Test
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    r = 'data/test/'

    r = os.path.abspath(r)
    A = DA_test(dir=r, img_size=(224,224))
    B = DataLoader(A, batch_size=10, num_workers=2)

    v = iter(B)

    for i in range(5):
        t = next(v)
        print(t.shape)

