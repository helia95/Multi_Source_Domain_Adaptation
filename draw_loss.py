import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

path = './checkpoints/'   #you can change here
target = 'real'
output_path = './visual/'

if not os.path.exists(output_path):
    os.mkdir(output_path)
#draw target accuracy
file_name = '{}target_accuracy_{}.p'.format(path,target)
f = open(file_name,'rb')
acc = pickle.load(f)

plt.title('{}'.format(target))
plt.xlabel('Epoch')
plt.xticks(np.arange(0,21,2))
plt.ylabel('Accuracy')
plt.plot(acc)
plt.savefig('{}{}_accuracy.jpg'.format(output_path,target))
plt.close()

#draw train loss
file_name = '{}class_loss_{}.p'.format(path,target)
f = open(file_name,'rb')
class_loss = pickle.load(f)
file_name = '{}domain_loss_{}.p'.format(path,target)
f = open(file_name,'rb')
domain_loss = pickle.load(f)
file_name = '{}total_loss_{}.p'.format(path,target)
f = open(file_name,'rb')
total_loss = pickle.load(f)

plt.title('{}'.format(target))
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.plot(domain_loss)
plt.plot(class_loss)
plt.plot(total_loss)
plt.legend(['domain', 'class', 'total'])
plt.savefig('{}{}_train_loss.jpg'.format(output_path,target))
plt.close()

#draw valid loss
file_name = '{}val_class_loss_{}.p'.format(path,target)
f = open(file_name,'rb')
val_class_loss = pickle.load(f)
file_name = '{}val_domain_loss_{}.p'.format(path,target)
f = open(file_name,'rb')
val_domain_loss = pickle.load(f)

plt.title('{}'.format(target))
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.plot(val_domain_loss)
plt.plot(val_class_loss)
plt.legend(['domain', 'class'])
plt.savefig('{}{}_valid_loss.jpg'.format(output_path,target))
plt.close()