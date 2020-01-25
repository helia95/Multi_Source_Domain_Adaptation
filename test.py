import os
from torch.utils.data import DataLoader
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import models
import csv

class test_dataset(data.Dataset):
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
        name = fname
        return img, name

    def __len__(self):
        return self.num_samples

class DA(data.Dataset):
    def __init__(self, dir, name, img_size, train):
        self.name = name
        self.fnames = []
        self.dir = dir
        self.train = train
        self.image_size = img_size
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
                    line_count += 1
                else:
                    line_count += 1

        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.dir + fname)).convert('RGB')
        img = self.transform(img)
        name = fname

        return img, name

    def __len__(self):
        return self.num_samples

tasks = ['infograph', 'sketch', 'real', 'quickdraw']

def taskSelect(target):
    tasks.remove(target)
    return tasks[0], tasks[1], tasks[2], target
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output_name')
    parser.add_argument('target_domain')
    parser.add_argument('model_path')

    a = parser.parse_args()

    root = 'data/'

    target = a.target_domain
    source1, source2, source3, target = taskSelect(target)
    adaptive_weight = True

    if not target == 'real':
        dataset_t = DA(dir=root,name=target,img_size=(224,224),train=False)
    else:
        dataset_t = test_dataset(dir='data/test', img_size=(224,224))

    dataloader_t = DataLoader(dataset_t, batch_size=64, shuffle=False,num_workers=8)

    path = a.model_path      #you may change the path

    feature_extractor = models.feature_extractor()
    classifier_1 = models.class_classifier()
    classifier_2 = models.class_classifier()
    classifier_3 = models.class_classifier()

    state = torch.load(path)

    feature_extractor.load_state_dict(state['feature_extractor'])
    classifier_1.load_state_dict(state['{}_classifier'.format(source1)])
    classifier_2.load_state_dict(state['{}_classifier'.format(source2)])
    classifier_3.load_state_dict(state['{}_classifier'.format(source3)])

    if adaptive_weight:
        w1_mean = state['{}_weight'.format(source1)]
        w2_mean = state['{}_weight'.format(source2)]
        w3_mean = state['{}_weight'.format(source3)]
    else:
        w1_mean = 1/3
        w2_mean = 1/3
        w3_mean = 1/3

    if torch.cuda.is_available():
        feature_extractor = feature_extractor.cuda()
        classifier_1 = classifier_1.cuda()
        classifier_2 = classifier_2.cuda()
        classifier_3 = classifier_3.cuda()

    feature_extractor.eval()
    classifier_1.eval(), classifier_2.eval(), classifier_3.eval()

    ans = open(a.output_name,'w')
    ans.write('image_name,label\n')
    with torch.no_grad():
        for idx, (img, name) in enumerate(dataloader_t):
            if torch.cuda.is_available():
                img = img.cuda()

            ft_t = feature_extractor(img)

            pred1 = classifier_1(ft_t)
            pred2 = classifier_2(ft_t)
            pred3 = classifier_3(ft_t)

            pred = (pred1*w1_mean+pred2*w2_mean+pred3*w3_mean)
            _, pred = torch.max(pred, dim=1)
            
            print('\r Predicting... Progress: %.1f %%' %(100*(idx+1)/len(dataloader_t)),end='')

            for i in range(len(name)):
                ans.write('{},{}\n'.format(os.path.join('test/',name[i]), pred[i]))
            
    ans.close()

if __name__=='__main__':
    main()