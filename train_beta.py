import os
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam, SGD
import torch
from torch.autograd import Variable
import models
from loss import momentumLoss, discrepancyLoss
import dataset
import pickle as pkl
import time
from utils import weight_init
import argparse
import numpy as np

# Seed
manualSeed = 123
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.backends.cudnn.enablesd = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(manualSeed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)
if use_cuda:
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

# Create output directory
path_output = './checkpoints/'
if not os.path.exists(path_output):
    os.makedirs(path_output)

root = 'data/'
tasks = ['infograph', 'sketch', 'real', 'quickdraw']

def get_args():
    parser = argparse.ArgumentParser(description = '.')
    parser.add_argument('--pretrain', default = None)
    parser.add_argument('--target', default = 'infograph', help = 'specify the target name')
    parser.add_argument('--bs', type = int, default = 10, help = 'batch_size')
    parser.add_argument('--ep', type = int, default = 20, help = 'epoches')
    parser.add_argument('--alpha', type = float, default = 1.0, help = 'alpha')
    parser.add_argument('--log_interval', type = int, default = 200, help = 'log interval')
    parser.add_argument('--eval_interval', type = int, default = 2000, help = 'eval and save interval')
    opt = parser.parse_args()
    return opt

def taskSelect(target):
    tasks.remove(target)
    return tasks[0], tasks[1], tasks[2], target

def eval(saved_weight, feature_extractor, classifier_1_, classifier_2_, classifier_3_,classifier_1, classifier_2, classifier_3, dataloader_t):
    feature_extractor.eval()
    classifier_1.eval(), classifier_2.eval(), classifier_3.eval()
    classifier_1_.eval(), classifier_2_.eval(), classifier_3_.eval()

    tot_acc = 0
    n_samples = 0
    with torch.no_grad():
        for i, (imgt, lbt) in enumerate(dataloader_t):
            print('\rTraining... Progress: %.1f %%'
                % (100*(i+1)/len(dataloader_t)),end='')

            cur_batch = imgt.shape[0]

            imgt = imgt.to(device)
            lbt = lbt.to(device)

            # Forward the test images
            ft_t = feature_extractor(imgt)
            pred1 = classifier_1(ft_t)
            pred2 = classifier_2(ft_t)
            pred3 = classifier_3(ft_t)
            pred1_ = classifier_1_(ft_t)
            pred2_ = classifier_2_(ft_t)
            pred3_ = classifier_3_(ft_t)

            # Compute accuracy
            preds = torch.stack((pred1, pred2, pred3, pred1_, pred2_, pred3_)).permute(1,0,2) # [bs,6,325]
            saved_weight_ = saved_weight.repeat(cur_batch, 1).unsqueeze(1) # [bs, 1, 6]
            output = torch.bmm(saved_weight_, preds) # [bs, 1, 325]
            output = output.squeeze(1)
         
            _, pred = torch.max(output, dim=1)
            correct = pred.eq(lbt.data.view_as(pred))
            # print(correct)
            accuracy = torch.mean(correct.type(torch.FloatTensor))
            tot_acc += accuracy.item() * cur_batch
            n_samples += cur_batch

    tot_t_acc = tot_acc / n_samples
    feature_extractor.train()
    classifier_1.train(), classifier_2.train(), classifier_3.train()
    classifier_1_.train(), classifier_2_.train(), classifier_3_.train()
    return tot_t_acc

def train(opt):
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(path_output)

    source1, source2, source3, target = taskSelect(opt.target)

    dataset_s1 = dataset.DA(dir=root, name=source1, img_size=(224, 224), train=True)
    dataset_s2 = dataset.DA(dir=root, name=source2, img_size=(224, 224), train=True)
    dataset_s3 = dataset.DA(dir=root, name=source3, img_size=(224, 224), train=True)
    dataset_t = dataset.DA(dir=root, name=target, img_size=(224, 224), train=True)
    dataset_tt = dataset.DA(dir=root, name=target, img_size=(224,224), train=False,real_val=False)

    dataloader_s1 = DataLoader(dataset_s1, batch_size=opt.bs, shuffle=True, num_workers=2)
    dataloader_s2 = DataLoader(dataset_s2, batch_size=opt.bs, shuffle=True, num_workers=2)
    dataloader_s3 = DataLoader(dataset_s3, batch_size=opt.bs, shuffle=True, num_workers=2)
    dataloader_t = DataLoader(dataset_t, batch_size=opt.bs, shuffle=True, num_workers=2)
    dataloader_tt = DataLoader(dataset_tt, batch_size=opt.bs, shuffle=False, num_workers=2)


    # dataset_s1 = dataset.DA(dir=root, name=source1, img_size=(224, 224), train=True)
    # dataset_s2 = dataset.DA(dir=root, name=source2, img_size=(224, 224), train=True)
    # dataset_s3 = dataset.DA(dir=root, name=source3, img_size=(224, 224), train=True)
    # dataset_t = dataset.DA(dir=root, name=target, img_size=(224, 224), train=True)

    # if target == 'real':
    #     tmp = os.path.join(root, 'test')
    #     dataset_tt = dataset.DA_test(dir=tmp, img_size=(224,224))
    # else:
    #     dataset_tt = dataset.DA(dir=root, name=target, img_size=(224, 224), train=False)

    # dataloader_s1 = DataLoader(dataset_s1, batch_size=opt.bs, shuffle=True, num_workers=2)
    # dataloader_s2 = DataLoader(dataset_s2, batch_size=opt.bs, shuffle=True, num_workers=2)
    # dataloader_s3 = DataLoader(dataset_s3, batch_size=opt.bs, shuffle=True, num_workers=2)
    # dataloader_t = DataLoader(dataset_t, batch_size=opt.bs, shuffle=True, num_workers=2)
    # dataloader_tt = DataLoader(dataset_tt, batch_size=opt.bs, shuffle=False, num_workers=2)


    len_data = min(len(dataset_s1), len(dataset_s2), len(dataset_s3), len(dataset_t))           # length of "shorter" domain
    len_bs = min(len(dataloader_s1), len(dataloader_s2), len(dataloader_s3), len(dataloader_t))

    # Define networks
    feature_extractor = models.feature_extractor()
    classifier_1 = models.class_classifier()
    classifier_2 = models.class_classifier()
    classifier_3 = models.class_classifier()
    classifier_1_ = models.class_classifier()
    classifier_2_ = models.class_classifier()
    classifier_3_ = models.class_classifier()

    # if torch.cuda.is_available():
    feature_extractor = feature_extractor.to(device)
    classifier_1 = classifier_1.to(device).apply(weight_init)
    classifier_2 = classifier_2.to(device).apply(weight_init)
    classifier_3 = classifier_3.to(device).apply(weight_init)
    classifier_1_ = classifier_1_.to(device).apply(weight_init)
    classifier_2_ = classifier_2_.to(device).apply(weight_init)
    classifier_3_ = classifier_3_.to(device).apply(weight_init)

    # Define loss
    mom_loss = momentumLoss()
    cl_loss = nn.CrossEntropyLoss()
    disc_loss = discrepancyLoss()

    # Optimizers
    # Change the LR
    optimizer_features = SGD(feature_extractor.parameters(), lr=0.0001,momentum=0.9,weight_decay=5e-4)
    optimizer_classifier = SGD(([{'params': classifier_1.parameters()},
                    {'params': classifier_2.parameters()},
                    {'params': classifier_3.parameters()}]), lr=0.002,momentum=0.9,weight_decay=5e-4)

    optimizer_classifier_ = SGD(([{'params': classifier_1_.parameters()},
                    {'params': classifier_2_.parameters()},
                    {'params': classifier_3_.parameters()}]), lr=0.002,momentum=0.9,weight_decay=5e-4)

    # optimizer_features = SGD(feature_extractor.parameters(), lr=0.0001)
    # optimizer_classifier = Adam(([{'params': classifier_1.parameters()},
    #                    {'params': classifier_2.parameters()},
    #                    {'params': classifier_3.parameters()}]), lr=0.002)
    # optimizer_classifier_ = Adam(([{'params': classifier_1_.parameters()},
    #                    {'params': classifier_2_.parameters()},
    #                    {'params': classifier_3_.parameters()}]), lr=0.002)

    if opt.pretrain is not None:
        state = torch.load(opt.pretrain)
        feature_extractor.load_state_dict(state['feature_extractor'])
        classifier_1.load_state_dict(state['{}_classifier'.format(source1)])
        classifier_2.load_state_dict(state['{}_classifier'.format(source2)])
        classifier_3.load_state_dict(state['{}_classifier'.format(source3)])
        classifier_1_.load_state_dict(state['{}_classifier_'.format(source1)])
        classifier_2_.load_state_dict(state['{}_classifier_'.format(source2)])
        classifier_3_.load_state_dict(state['{}_classifier_'.format(source3)])

    # Lists
    train_loss = []
    acc_on_target = []

    tot_loss, tot_clf_loss, tot_mom_loss, tot_s2_loss, tot_s3_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    n_samples, iteration = 0, 0
    tot_correct = [0, 0, 0, 0, 0, 0]
    saved_time = time.time()
    feature_extractor.train()
    classifier_1.train(), classifier_2.train(), classifier_3.train()
    classifier_1_.train(), classifier_2_.train(), classifier_3_.train()

    for epoch in range(opt.ep):

        if epoch+1 == 5:
            optimizer_classifier = SGD(([{'params': classifier_1.parameters()},
                    {'params': classifier_2.parameters()},
                    {'params': classifier_3.parameters()}]), lr=0.001,momentum=0.9,weight_decay=5e-4)

            optimizer_classifier_ = SGD(([{'params': classifier_1_.parameters()},
                            {'params': classifier_2_.parameters()},
                            {'params': classifier_3_.parameters()}]), lr=0.001,momentum=0.9,weight_decay=5e-4)

        if epoch+1 == 10:
            optimizer_classifier = SGD(([{'params': classifier_1.parameters()},
                    {'params': classifier_2.parameters()},
                    {'params': classifier_3.parameters()}]), lr=0.0001,momentum=0.9,weight_decay=5e-4)

            optimizer_classifier_ = SGD(([{'params': classifier_1_.parameters()},
                            {'params': classifier_2_.parameters()},
                            {'params': classifier_3_.parameters()}]), lr=0.0001,momentum=0.9,weight_decay=5e-4)


        for i, (data_1, data_2, data_3, data_t) in enumerate(zip(dataloader_s1, dataloader_s2, dataloader_s3, dataloader_t)):

            img1, lb1 = data_1
            img2, lb2 = data_2
            img3, lb3 = data_3
            imgt, _ = data_t

            # Prepare data
            cur_batch = min(img1.shape[0], img2.shape[0], img3.shape[0], imgt.shape[0])
            # print(i, cur_batch)
            img1, lb1 = Variable(img1[0:cur_batch,:,:,:]).to(device), Variable(lb1[0:cur_batch]).to(device)
            img2, lb2 = Variable(img2[0:cur_batch,:,:,:]).to(device), Variable(lb2[0:cur_batch]).to(device)
            img3, lb3 = Variable(img3[0:cur_batch,:,:,:]).to(device), Variable(lb3[0:cur_batch]).to(device)
            imgt = Variable(imgt[0:cur_batch,:,:,:]).to(device)

            ### STEP 1 ### train G and C pairs
            # Forward
            optimizer_features.zero_grad()
            optimizer_classifier.zero_grad()
            optimizer_classifier_.zero_grad()

            # Extract Features
            ft1 = feature_extractor(img1)
            ft2 = feature_extractor(img2)
            ft3 = feature_extractor(img3)
            ft_t = feature_extractor(imgt)

            # Class Prediction [bs, 345]
            cl1, cl1_ = classifier_1(ft1), classifier_1_(ft1)
            cl2, cl2_ = classifier_2(ft2), classifier_2_(ft2)
            cl3, cl3_ = classifier_3(ft3), classifier_3_(ft3)

            # Compute "momentum loss"
            loss_mom = mom_loss(ft1, ft2, ft3, ft_t)

            # Cross entropy loss
            l1, l1_ = cl_loss(cl1, lb1), cl_loss(cl1_, lb1)
            l2, l2_ = cl_loss(cl2, lb2), cl_loss(cl2_, lb2)
            l3, l3_ = cl_loss(cl3, lb3), cl_loss(cl3_, lb3)
            # total loss
            s1loss = l1 + l2 + l3 + l1_ + l2_ + l3_ + opt.alpha * loss_mom

            s1loss.backward()
            optimizer_features.step()
            optimizer_classifier.step()
            optimizer_classifier_.step()

            ### STEP 2 ### fix G, and train C pairs 
            optimizer_classifier.zero_grad()
            optimizer_classifier_.zero_grad()

            # Class Prediction on each src domain
            cl1, cl1_ = classifier_1(ft1.detach()), classifier_1_(ft1.detach())
            cl2, cl2_ = classifier_2(ft2.detach()), classifier_2_(ft2.detach())
            cl3, cl3_ = classifier_3(ft3.detach()), classifier_3_(ft3.detach())

            # discrepancy on tgt domain
            clt1, clt1_ = classifier_1(ft_t.detach()), classifier_1_(ft_t.detach())
            clt2, clt2_ = classifier_2(ft_t.detach()), classifier_2_(ft_t.detach())
            clt3, clt3_ = classifier_3(ft_t.detach()), classifier_3_(ft_t.detach())

            # classification loss
            l1, l1_ = cl_loss(cl1, lb1), cl_loss(cl1_, lb1)
            l2, l2_ = cl_loss(cl2, lb2), cl_loss(cl2_, lb2)
            l3, l3_ = cl_loss(cl3, lb3), cl_loss(cl3_, lb3)

            # print(clt1.shape)
            dl1 = disc_loss(clt1, clt1_)
            dl2 = disc_loss(clt2, clt2_)
            dl3 = disc_loss(clt3, clt3_)
            # print(dl1, dl2, dl3)

            # backward
            s2loss = l1 + l2 + l3 + l1_ + l2_ + l3_ - dl1 - dl2 - dl3
            s2loss.backward()
            optimizer_classifier.step()
            optimizer_classifier_.step()

            ### STEP 3 #### fix C pairs, train G
            optimizer_features.zero_grad()

            ft_t = feature_extractor(imgt)
            clt1, clt1_ = classifier_1(ft_t), classifier_1_(ft_t)
            clt2, clt2_ = classifier_2(ft_t), classifier_2_(ft_t)
            clt3, clt3_ = classifier_3(ft_t), classifier_3_(ft_t)

            dl1 = disc_loss(clt1, clt1_)
            dl2 = disc_loss(clt2, clt2_)
            dl3 = disc_loss(clt3, clt3_)

            s3loss = dl1 + dl2 + dl3
            s3loss.backward()
            optimizer_features.step()
            


            pred = torch.stack((cl1, cl2, cl3, cl1_, cl2_, cl3_), 0) # [6, bs, 345]
            _, pred = torch.max(pred, dim = 2) # [6, bs]
            gt = torch.stack((lb1, lb2, lb3, lb1, lb2, lb3), 0) # [6, bs]
            correct = pred.eq(gt.data)
            correct = torch.mean(correct.type(torch.FloatTensor), dim = 1).cpu().numpy()

            tot_loss += s1loss.item() * cur_batch
            tot_clf_loss += (s1loss.item() - opt.alpha * loss_mom.item()) * cur_batch
            tot_s2_loss += s2loss.item() * cur_batch
            tot_s3_loss += s3loss.item() * cur_batch
            tot_mom_loss += loss_mom.item() * cur_batch
            tot_correct += correct * cur_batch
            n_samples += cur_batch

            # print(cur_batch)
            if iteration % opt.log_interval == 0:
                current_time = time.time()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tClfLoss: {:.4f}\tMMLoss: {:.4f}\t \
                    S2Loss: {:.4f}\tS3Loss: {:.4f}\t \
                    Accu: {:.4f}\\{:.4f}\\{:.4f}\\{:.4f}\\{:.4f}\\{:.4f}\tTime: {:.3f}'.format(\
                        epoch, i * opt.bs, len_data, 100. * i / len_bs, \
                        tot_clf_loss / n_samples, 
                        tot_mom_loss / n_samples,
                        tot_s2_loss / n_samples,
                        tot_s3_loss / n_samples,
                        tot_correct[0] / n_samples,
                        tot_correct[1] / n_samples,
                        tot_correct[2] / n_samples,
                        tot_correct[3] / n_samples,
                        tot_correct[4] / n_samples,
                        tot_correct[5] / n_samples,
                        current_time - saved_time))
                writer.add_scalar('Train/ClfLoss', tot_clf_loss / n_samples, iteration * opt.bs)
                writer.add_scalar('Train/MMLoss', tot_mom_loss / n_samples, iteration * opt.bs)
                writer.add_scalar('Train/s2Loss', tot_s2_loss / n_samples, iteration * opt.bs)
                writer.add_scalar('Train/s3Loss', tot_s3_loss / n_samples, iteration * opt.bs)
                writer.add_scalar('Train/Accu0', tot_correct[0] / n_samples, iteration * opt.bs)
                writer.add_scalar('Train/Accu1', tot_correct[1] / n_samples, iteration * opt.bs)
                writer.add_scalar('Train/Accu2', tot_correct[2] / n_samples, iteration * opt.bs)
                writer.add_scalar('Train/Accu0_', tot_correct[3] / n_samples, iteration * opt.bs)
                writer.add_scalar('Train/Accu1_', tot_correct[4] / n_samples, iteration * opt.bs)
                writer.add_scalar('Train/Accu2_', tot_correct[5] / n_samples, iteration * opt.bs)

                saved_weight = torch.FloatTensor([tot_correct[0], tot_correct[1], tot_correct[2], tot_correct[3], tot_correct[4], tot_correct[5]]).to(device)
                if torch.sum(saved_weight) == 0.:
                    saved_weight = torch.FloatTensor(6).to(device).fill_(1)/6.
                else:
                    saved_weight = saved_weight/torch.sum(saved_weight)
                
                saved_time = time.time()
                tot_clf_loss, tot_mom_loss, tot_correct, n_samples = 0, 0, [0, 0, 0, 0, 0, 0], 0
                tot_s2_loss, tot_s3_loss = 0, 0
                train_loss.append(tot_loss)

            # evaluation and save
            if iteration % opt.eval_interval == 0 and iteration >= 0 and target != 'real':
                print('weight = ', saved_weight.cpu().numpy())
                evalacc = eval(saved_weight, feature_extractor, classifier_1_, classifier_2_, classifier_3_,
                classifier_1, classifier_2, classifier_3, dataloader_tt)
                writer.add_scalar('Test/Accu', evalacc, iteration * opt.bs)
                acc_on_target.append(evalacc)
                print('Eval Acc = {:.2f}\n'.format(evalacc*100))
                torch.save({
                        'epoch': epoch,
                        'feature_extractor': feature_extractor.state_dict(),
                        '{}_classifier'.format(source1): classifier_1.state_dict(),
                        '{}_classifier'.format(source2): classifier_2.state_dict(),
                        '{}_classifier'.format(source3): classifier_3.state_dict(),
                        '{}_classifier_'.format(source1): classifier_1_.state_dict(),
                        '{}_classifier_'.format(source2): classifier_2_.state_dict(),
                        '{}_classifier_'.format(source3): classifier_3_.state_dict(),
                        'features_optimizer': optimizer_features.state_dict(),
                        'classifier_optimizer': optimizer_classifier.state_dict(),
                        'loss': tot_loss,
                        'saved_weight': saved_weight
               }, os.path.join(path_output, target + '-{}-{:.2f}.pth'.format(epoch, evalacc*100)))

            iteration += 1

    pkl.dump(train_loss, open('{}train_loss.p'.format(path_output), 'wb'))
    if target != 'real':
        pkl.dump(acc_on_target, open('{}target_accuracy.p'.format(path_output), 'wb'))


if __name__ == '__main__':
    opt = get_args()
    train(opt)