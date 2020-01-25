import models
import torch
import dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy('file_system')
tasks = ['infograph', 'sketch', 'quickdraw']
def taskSelect(target):
    tasks.remove(target)
    return tasks[0], tasks[1], target

# IUPUT
root = 'data/'
target = 'quickdraw'
pretrain_path = 'beta_qkr/quickdraw-19-16.11.pth'
batch_size = 48

source1, source2, target = taskSelect(target)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dataset_s1 = dataset.DA(dir=root, name=source1, img_size=(224, 224), train=False)
dataset_s2 = dataset.DA(dir=root, name=source2, img_size=(224, 224), train=False)
dataset_t = dataset.DA(dir=root, name=target, img_size=(224, 224), train=False)

dataloader_s1 = DataLoader(dataset_s1, batch_size=batch_size, shuffle=True, num_workers=3)
dataloader_s2 = DataLoader(dataset_s2, batch_size=batch_size, shuffle=True, num_workers=3)
dataloader_t = DataLoader(dataset_t, batch_size=batch_size, shuffle=True, num_workers=3)

feature_extractor = models.feature_extractor().to(device)
state = torch.load(pretrain_path)
feature_extractor.load_state_dict(state['feature_extractor'])

feature_extractor.eval()

tot_acc = 0
n_samples = 0

latents = []
labels = []
domains = []

with torch.no_grad():
    for i, (imgt, lbt) in enumerate(dataloader_s1):
        b_size = imgt.shape[0]
        imgt = imgt.to(device)
        ft_t = feature_extractor(imgt)
        latents.append(ft_t.detach().cpu().numpy())
        labels.append(lbt.detach().cpu().numpy())
        domains.append(np.zeros(b_size, dtype = int))
        print('\rEvaluation... Progress: %.1f %%'
                % (100*(i+1)/len(dataloader_s1)),end='')
        # if i == 100:
        #     break
    print('')

    for i, (imgt, lbt) in enumerate(dataloader_s2):
        b_size = imgt.shape[0]
        imgt = imgt.to(device)
        ft_t = feature_extractor(imgt)
        latents.append(ft_t.detach().cpu().numpy())
        labels.append(lbt.detach().cpu().numpy())
        domains.append(np.zeros(b_size, dtype = int)+1)
        print('\rEvaluation... Progress: %.1f %%'
                % (100*(i+1)/len(dataloader_s2)),end='')
        # if i == 100:
        #     break

    print('')
    for i, (imgt, lbt) in enumerate(dataloader_t):
        b_size = imgt.shape[0]
        imgt = imgt.to(device)
        ft_t = feature_extractor(imgt)
        latents.append(ft_t.detach().cpu().numpy())
        labels.append(lbt.detach().cpu().numpy())
        domains.append(np.zeros(b_size, dtype = int)+2)
        print('\rEvaluation... Progress: %.1f %%'
                % (100*(i+1)/len(dataloader_t)),end='')
        # if i == 100:
        #     break
    print('')

    # print(latents[])
    latents = np.concatenate(latents)
    labels = np.concatenate(labels)
    domains = np.concatenate(domains)

    # uniquelabels = np.unique(labels)
    # choice_labels = np.random.choice(uniquelabels,10, replace = False)
    choice_labels = [7, 261, 26, 337, 262, 165, 230, 255, 50, 335]

    selected_latents = []
    selected_labels = []
    selected_domains = []
    for i in range(10):
        choice_label = choice_labels[i]
        selected_latents.append(latents[labels == choice_label])
        selected_labels.append(labels[labels == choice_label])
        selected_domains.append(domains[labels == choice_label])

    selected_latents = np.concatenate(selected_latents)
    selected_labels = np.concatenate(selected_labels)
    selected_domains = np.concatenate(selected_domains)

    embed = TSNE(n_components=2).fit_transform(selected_latents)

    # plot by labels
    plt.figure()
    for i in range(10):
        choice_label = choice_labels[i]
        plt.scatter(embed[selected_labels == choice_label, 0], 
            embed[selected_labels == choice_label, 1], 
            c = 'C' + str(i),
            s = 4,
            label = choice_label,
            marker = '.',
            )
    plt.legend(loc='upper right')
    plt.savefig('tsne_{}_1.png'.format(target))
    plt.close()


    plt.figure()
    domain_name = ['src1', 'src2', 'tgt']
    for i in range(3):
        plt.scatter(embed[selected_domains == i, 0],
            embed[selected_domains == i, 1],
            c = 'C' + str(i),
            s = 4,
            label = domain_name[i],
            marker = '.')
    plt.legend(loc='upper right')
    plt.savefig('tsne_{}_2.png'.format(target))
    plt.close()

    # print(latents.shape)

