import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import torch
from torch.autograd import Variable
import models
from loss import momentumLoss
import dataset
import pickle as pkl
import torch.nn.init as init

def weight_init(m):
    
    if isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)

def main():
    # Create output directory
    path_output = './checkpoints/'
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    # Hyperparameters, to change
    epochs = 50
    batch_size = 24
    alpha = 1                   # it's the trade-off parameter of loss function, what values should it take?

    # Source domains name
    save_interval = 10          # save every 10 epochs
    root = 'data/'

    source1 = 'sketch'
    source2 = 'quickdraw'
    source3 = 'real'
    target = 'infograph'

    # Dataloader
    dataset_s1 = dataset.DA(dir=root, name=source1, img_size=(224, 224), train=True)
    dataset_s2 = dataset.DA(dir=root, name=source2, img_size=(224, 224), train=True)
    dataset_s3 = dataset.DA(dir=root, name=source3, img_size=(224, 224), train=True)

    if target == 'real':
        tmp = os.path.join(root, 'test')
        dataset_t = dataset.DA_test(dir=tmp, img_size=(224,224))
    else:
        dataset_t = dataset.DA(dir=root, name=target, img_size=(224, 224), train=False)

    dataloader_s1 = DataLoader(dataset_s1, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_s2 = DataLoader(dataset_s2, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_s3 = DataLoader(dataset_s3, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_t = DataLoader(dataset_t, batch_size=batch_size, shuffle=True, num_workers=2)


    len_data = min(len(dataset_s1), len(dataset_s2), len(dataset_s3), len(dataset_t))           # length of "shorter" domain

    # Define networks
    feature_extractor = models.feature_extractor()
    classifier_1 = models.class_classifier()
    classifier_2 = models.class_classifier()
    classifier_3 = models.class_classifier()

    # Weight initialization
    classifier_1.apply(weight_init)
    classifier_2.apply(weight_init)
    classifier_3.apply(weight_init)
    
    if torch.cuda.is_available():
        feature_extractor = feature_extractor.cuda()
        classifier_1 = classifier_1.cuda()
        classifier_2 = classifier_2.cuda()
        classifier_3 = classifier_3.cuda()

    # Define loss
    mom_loss = momentumLoss()
    cl_loss = nn.CrossEntropyLoss()

    # Optimizers
    # Change the LR
    optimizer_features = Adam(feature_extractor.parameters(), lr=0.0001)
    optimizer_classifier = Adam(([{'params': classifier_1.parameters()},
                    {'params': classifier_2.parameters()},
                    {'params': classifier_3.parameters()}]), lr=0.002)


    # Lists
    train_loss = []
    acc_on_target = []

    for epoch in range(epochs):
        tot_loss = 0.0
        feature_extractor.train()
        classifier_1.train(), classifier_2.train(), classifier_3.train()

        for i, (data_1, data_2, data_3, data_t) in enumerate(zip(dataloader_s1, dataloader_s2, dataloader_s3, dataloader_t)):
            img1, lb1 = data_1
            img2, lb2 = data_2
            img3, lb3 = data_3
            if target == 'real':
                imgt = data_t
            else:
                imgt, _ = data_t

            # Prepare data
            cur_batch = min(img1.shape[0], img2.shape[0], img3.shape[0], imgt.shape[0])

            img1, lb1 = Variable(img1[0:cur_batch,:,:,:]).cuda(), Variable(lb1[0:cur_batch]).cuda()
            img2, lb2 = Variable(img2[0:cur_batch,:,:,:]).cuda(), Variable(lb2[0:cur_batch]).cuda()
            img3, lb3 = Variable(img3[0:cur_batch,:,:,:]).cuda(), Variable(lb3[0:cur_batch]).cuda()
            imgt = Variable(imgt[0:cur_batch,:,:,:]).cuda()


            # Forward
            optimizer_features.zero_grad()
            optimizer_classifier.zero_grad()

            # Extract Features
            ft1 = feature_extractor(img1)
            ft2 = feature_extractor(img2)
            ft3 = feature_extractor(img3)
            ft_t = feature_extractor(imgt)

            # Class Prediction
            cl1 = classifier_1(ft1)
            cl2 = classifier_2(ft2)
            cl3 = classifier_3(ft3)

            # Compute "momentum loss"
            loss_mom = mom_loss(ft1, ft2, ft3, ft_t)

            # Cross entropy loss
            l1 = cl_loss(cl1, lb1)
            l2 = cl_loss(cl2, lb2)
            l3 = cl_loss(cl3, lb3)

            # total loss
            loss = l1 + l2 + l3 + alpha * loss_mom
            #print(loss_mom,(l1+l2+l3))
            loss.backward()
            optimizer_features.step()
            optimizer_classifier.step()

            tot_loss += loss.item() * cur_batch
            
        tot_t_loss = tot_loss / (len_data)

        # Print
        train_loss.append(tot_t_loss)
        print('*************************************************')
        print('Epoch [%d/%d], Training loss: %.4f'
            % (epoch + 1, epochs, tot_t_loss))
        ####################################################################################################################
        # Compute the accuracy at the end of each epoch
        if target != 'real':

            feature_extractor.eval()
            classifier_1.eval(), classifier_2.eval(), classifier_3.eval()

            tot_acc = 0
            with torch.no_grad():
                for i, (imgt, lbt) in enumerate(dataloader_t):

                    cur_batch = imgt.shape[0]

                    imgt = imgt.cuda()
                    lbt = lbt.cuda()

                    # Forward the test images
                    ft_t = feature_extractor(imgt)

                    pred1 = classifier_1(ft_t)
                    pred2 = classifier_2(ft_t)
                    pred3 = classifier_3(ft_t)

                    # Compute accuracy
                    output = torch.mean(torch.stack((pred1, pred2, pred3)), 0)
                    _, pred = torch.max(output, dim=1)
                    correct = pred.eq(lbt.data.view_as(pred))
                    accuracy = torch.mean(correct.type(torch.FloatTensor))
                    tot_acc += accuracy.item() * cur_batch

            tot_t_acc = tot_acc / (len(dataset_t))

            # Print
            acc_on_target.append(tot_t_acc)
            print('Epoch [%d/%d], Accuracy on target: %.4f'
                % (epoch + 1, epochs, tot_t_acc))

        # Save every save_interval
        if epoch % save_interval == 0 or epoch == epochs-1:
            torch.save({
                        'epoch': epoch,
                        'feature_extractor': feature_extractor.state_dict(),
                        '{}_classifier'.format(source1): classifier_1.state_dict(),
                        '{}_classifier'.format(source2): classifier_2.state_dict(),
                        '{}_classifier'.format(source3): classifier_3.state_dict(),
                        'features_optimizer': optimizer_features.state_dict(),
                        'classifier_optimizer': optimizer_classifier.state_dict(),
                        'loss': tot_loss,
            }, os.path.join(path_output, target + '-{}.pth'.format(epoch)))

    # Save training loss and accuracy on target (if not 'real')
    pkl.dump(train_loss, open('{}train_loss.p'.format(path_output), 'wb'))
    if target != 'real':
        pkl.dump(acc_on_target, open('{}target_accuracy.p'.format(path_output), 'wb'))

if __name__ == '__main__':
    main()
