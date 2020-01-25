import numpy as np
import pandas as pd
import os
import sys

target = 'infograph'
gt_csv_root = 'data/{}/{}_test.csv'.format(target,target)
pred_csv_root = '{}_pred.csv'.format(target)

gt_landmarks = pd.read_csv(gt_csv_root)
pred_landmarks = pd.read_csv(pred_csv_root)

gt_length = len(gt_landmarks.iloc[:,1])
pred_length = len(pred_landmarks.iloc[:,1])
print('GT has %d labels, Pred has %d labels' % (gt_length,pred_length))

val_acc = 0.0
for i in range(gt_length):
    gt_label = int(gt_landmarks.iloc[i,1])
    pred_label = int(pred_landmarks.iloc[i,1])
    acc = np.sum(gt_label == pred_label)
    val_acc += acc

print('Accuracy on {} is: %.4f'.format(target) % (val_acc/gt_length))
