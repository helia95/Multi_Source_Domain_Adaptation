import os
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam,SGD
import torch
from torch.autograd import Variable
from utils import weight_init
import models 
import dataset
import pickle as pkl
import numpy as np
import timeit
import torch.nn.init as init


def main():
    # Create output directory
    path_output = './checkpoints/'
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    # Hyperparameters, to change
    epochs = 20
    batch_size = 16
    alpha = 1                   # it's the trade-off parameter of loss function, what values should it take?
    gamma = 1
    # Source domains name
    root = 'data/'

    source1 = 'sketch'
    source2 = 'infograph'
    source3 = 'real'
    target = 'quickdraw'

    # Dataloader
    dataset_s1 = dataset.DA(dir=root, name=source1, img_size=(224, 224), train=True)
    dataset_s2 = dataset.DA(dir=root, name=source2, img_size=(224, 224), train=True)
    dataset_s3 = dataset.DA(dir=root, name=source3, img_size=(224, 224), train=True)
    dataset_t = dataset.DA(dir=root, name=target, img_size=(224, 224), train=True)
    dataset_val = dataset.DA(dir=root, name=target, img_size=(224,224), train=False,real_val=False)

    dataloader_s1 = DataLoader(dataset_s1, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_s2 = DataLoader(dataset_s2, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_s3 = DataLoader(dataset_s3, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_t = DataLoader(dataset_t, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)


    len_data = min(len(dataset_s1), len(dataset_s2), len(dataset_s3), len(dataset_t))           # length of "shorter" domain
    len_dataloader = min(len(dataloader_s1),len(dataloader_s2),len(dataloader_s3),len(dataloader_t))
 
    # Define networks
    feature_extractor = models.feature_extractor()
    classifier_1 = models.class_classifier()
    classifier_2 = models.class_classifier()
    classifier_3 = models.class_classifier()
    classifier_1.apply(weight_init)
    classifier_2.apply(weight_init)
    classifier_3.apply(weight_init)

    discriminator = models.discriminator()
    discriminator.apply(weight_init)

    if torch.cuda.is_available():
        feature_extractor = feature_extractor.cuda()
        classifier_1 = classifier_1.cuda()
        classifier_2 = classifier_2.cuda()
        classifier_3 = classifier_3.cuda()
        discriminator = discriminator.cuda()

    # Define loss
    cl_loss = nn.CrossEntropyLoss()
    disc_loss = nn.NLLLoss()

    # Optimizers
    # Change the LR
    optimizer_features = SGD(feature_extractor.parameters(), lr=0.0001,momentum=0.9,weight_decay=5e-4)
    optimizer_classifier = SGD(([{'params': classifier_1.parameters()},
                    {'params': classifier_2.parameters()},
                    {'params': classifier_3.parameters()}]), lr=0.002,momentum=0.9,weight_decay=5e-4)

    optimizer_discriminator = SGD(([{'params': discriminator.parameters()},
                    ]), lr=0.002,momentum=0.9,weight_decay=5e-4)



    # Lists
    train_loss = []
    train_class_loss = []
    train_domain_loss = []
    val_class_loss = []
    val_domain_loss = []
    acc_on_target = []
    best_acc = 0.0
    for epoch in range(epochs):
        epochTic = timeit.default_timer()
        tot_loss = 0.0
        tot_c_loss = 0.0
        tot_d_loss = 0.0
        tot_val_c_loss = 0.0
        tot_val_d_loss = 0.0        
        w1_mean = 0.0
        w2_mean = 0.0
        w3_mean = 0.0
        feature_extractor.train()
        classifier_1.train(), classifier_2.train(), classifier_3.train()
        discriminator.train()
        if epoch+1 == 5:
            optimizer_classifier = SGD(([{'params': classifier_1.parameters()},
                {'params': classifier_2.parameters()},
                {'params': classifier_3.parameters()}]), lr=0.001,momentum=0.9,weight_decay=5e-4)

            optimizer_discriminator = SGD(([{'params': discriminator.parameters()}
            ]), lr=0.001,momentum=0.9,weight_decay=5e-4)

        if epoch+1 == 10:
            optimizer_classifier = SGD(([{'params': classifier_1.parameters()},
                {'params': classifier_2.parameters()},
                {'params': classifier_3.parameters()}]), lr=0.0001,momentum=0.9,weight_decay=5e-4)

            optimizer_discriminator = SGD(([{'params': discriminator.parameters()}
            ]), lr=0.0001,momentum=0.9,weight_decay=5e-4)
        print('*************************************************')
        for i, (data_1, data_2, data_3, data_t) in enumerate(zip(dataloader_s1, dataloader_s2, dataloader_s3, dataloader_t)):

            p = float(i + epoch * len_data) / epochs / len_data
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            img1, lb1 = data_1
            img2, lb2 = data_2
            img3, lb3 = data_3
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
            optimizer_discriminator.zero_grad()

            # Extract Features
            ft1 = feature_extractor(img1)
            ft2 = feature_extractor(img2)
            ft3 = feature_extractor(img3)
            ft_t = feature_extractor(imgt)

            # Train the discriminator
            ds_s1 = discriminator(torch.cat((ft1,ft2,ft3)), alpha)
            ds_t = discriminator(ft_t,alpha)


            # Class Prediction
            cl1 = classifier_1(ft1)
            cl2 = classifier_2(ft2)
            cl3 = classifier_3(ft3)

            # Compute the "discriminator loss"
            ds_label = torch.zeros(cur_batch*3).long()
            dt_label = torch.ones(cur_batch).long()

            d_s = disc_loss(ds_s1, ds_label.cuda())
            d_t = disc_loss(ds_t,dt_label.cuda())

            # Cross entropy loss
            l1 = cl_loss(cl1, lb1)
            l2 = cl_loss(cl2, lb2)
            l3 = cl_loss(cl3, lb3)

            # Classifier Weight
            total_class_loss = 1/l1+1/l2+1/l3
            w1 = (1/l1)/total_class_loss
            w2 = (1/l2)/total_class_loss
            w3 = (1/l3)/total_class_loss
            w1_mean += w1
            w2_mean += w2
            w3_mean += w3

            # total loss
            Class_loss = l1 + l2 + l3
            Domain_loss = gamma*(d_s + d_t)
            loss = Class_loss + Domain_loss

            loss.backward()
            optimizer_features.step()
            optimizer_classifier.step()
            optimizer_discriminator.step()

            tot_loss += loss.item() * cur_batch
            tot_c_loss += Class_loss.item()
            tot_d_loss += Domain_loss.item()
            # Progress indicator
            print('\rTraining... Progress: %.1f %%'
                % (100*(i+1)/len_dataloader),end='')
            
            # Save Class loss and Domain loss
            if i % 50 == 0:
                train_class_loss.append(tot_c_loss/(i+1))
                train_domain_loss.append(tot_d_loss/(i+1))
                train_loss.append(tot_loss/(i+1)/cur_batch)

        tot_t_loss = tot_loss / (len_data)

        w1_mean /= len_dataloader
        w2_mean /= len_dataloader
        w3_mean /= len_dataloader
        #print(w1_mean,w2_mean,w3_mean)
        
        print('\rEpoch [%d/%d], Training loss: %.4f'
            % (epoch + 1, epochs, tot_t_loss),end='\n')
        ####################################################################################################################
        # Compute the accuracy at the end of each epoch
        feature_extractor.eval()
        classifier_1.eval(), classifier_2.eval(), classifier_3.eval()
        discriminator.eval()

        tot_acc = 0
        with torch.no_grad():
            for i, (imgt, lbt) in enumerate(dataloader_val):

                cur_batch = imgt.shape[0]

                imgt = imgt.cuda()
                lbt = lbt.cuda()

                # Forward the test images
                ft_t = feature_extractor(imgt)

                pred1 = classifier_1(ft_t)
                pred2 = classifier_2(ft_t)
                pred3 = classifier_3(ft_t)
                val_ds_t = discriminator(ft_t,alpha)

                # Compute class loss
                val_l1 = cl_loss(pred1,lbt)
                val_l2 = cl_loss(pred2,lbt)
                val_l3 = cl_loss(pred3,lbt)
                val_CE_loss = val_l1 + val_l2 + val_l3
                
                # Compute domain loss
                val_dt_label = torch.ones(cur_batch).long()
                val_d_t = disc_loss(val_ds_t,val_dt_label.cuda())

                # Compute accuracy
                output = pred1*w1_mean + pred2*w2_mean + pred3*w3_mean
                _, pred = torch.max(output, dim=1)
                correct = pred.eq(lbt.data.view_as(pred))
                accuracy = torch.mean(correct.type(torch.FloatTensor))
                tot_acc += accuracy.item() * cur_batch

                # total loss
                tot_val_c_loss += val_CE_loss.item()
                tot_val_d_loss += val_d_t.item()

                # Progress indicator
                print('\rValidation... Progress: %.1f %%'
                    % (100*(i+1)/len(dataloader_val)),end='')

                # Save validation loss
                if i % 50 == 0:
                    val_class_loss.append(tot_val_c_loss/(i+1))
                    val_domain_loss.append(tot_val_d_loss/(i+1))

            tot_t_acc = tot_acc / (len(dataset_val))

            # Print
            acc_on_target.append(tot_t_acc)
            print('\rEpoch [%d/%d], Accuracy on target: %.4f'
                % (epoch + 1, epochs, tot_t_acc),end='\n')

        # Save every save_interval
        if best_acc < tot_t_acc:
            torch.save({
                        'epoch': epoch,
                        'feature_extractor': feature_extractor.state_dict(),
                        '{}_classifier'.format(source1): classifier_1.state_dict(),
                        '{}_classifier'.format(source2): classifier_2.state_dict(),
                        '{}_classifier'.format(source3): classifier_3.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'features_optimizer': optimizer_features.state_dict(),
                        'classifier_optimizer': optimizer_classifier.state_dict(),
                        'loss': tot_loss,
                        '{}_weight'.format(source1): w1_mean,
                        '{}_weight'.format(source2): w2_mean,
                        '{}_weight'.format(source3): w3_mean,                        
            }, os.path.join(path_output, target + '-{}.pth'.format(epoch)))
            print('Saved best model!')
            best_acc = tot_t_acc
        
        # Pirnt elapsed time per epoch
        epochToc = timeit.default_timer()
        (t_min,t_sec) = divmod((epochToc-epochTic),60)
        print('Elapsed time is: %d min: %d sec' % (t_min,t_sec))
        
        # Save training loss and accuracy on target (if not 'real')
        pkl.dump(train_loss, open('{}total_loss_{}.p'.format(path_output,target), 'wb'))
        pkl.dump(train_class_loss, open('{}class_loss_{}.p'.format(path_output,target), 'wb'))
        pkl.dump(train_domain_loss, open('{}domain_loss_{}.p'.format(path_output,target), 'wb'))
        pkl.dump(acc_on_target, open('{}target_accuracy_{}.p'.format(path_output,target), 'wb'))
        pkl.dump(val_class_loss, open('{}val_class_loss_{}.p'.format(path_output,target), 'wb'))
        pkl.dump(val_domain_loss, open('{}val_domain_loss_{}.p'.format(path_output,target), 'wb'))


if __name__ == '__main__':
    main()