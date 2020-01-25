import torch
import torch.nn as nn

class momentumLoss(nn.Module):

    def forward(self, src1, src2, src3, y):

        batch_sz = src1.shape[0]

        mom_1_t = torch.mean(y, 0)
        mom_2_t = torch.mean( torch.mul(y, y), 0)

        mom_1_s1 = torch.mean(src1, 0)
        mom_1_s2 = torch.mean(src2, 0)
        mom_1_s3 = torch.mean(src3, 0)

        mom_2_s1 = torch.mean(torch.mul(src1, src1), 0)
        mom_2_s2 = torch.mean(torch.mul(src2, src2), 0)
        mom_2_s3 = torch.mean(torch.mul(src3, src3), 0)


        term1 = (1/3 * (torch.dist(mom_1_s1, mom_1_t,2) + torch.dist(mom_1_s2, mom_1_t, 2) + torch.dist(mom_1_s3, mom_1_t,2)) +
                1/3 * (torch.dist(mom_1_s1, mom_1_s2, 2) + torch.dist(mom_1_s1, mom_1_s3, 2) + torch.dist(mom_1_s2, mom_1_s3, 2))
                 )

        term2 = (1/3 * (torch.dist(mom_2_s1, mom_2_t,2) + torch.dist(mom_2_s2, mom_2_t, 2) + torch.dist(mom_2_s3, mom_2_t,2)) +
                1/3 * (torch.dist(mom_2_s1, mom_2_s2, 2) + torch.dist(mom_2_s1, mom_2_s3, 2) + torch.dist(mom_2_s2, mom_2_s3, 2))
                 )

        loss = term1 + term2

        return loss

class discrepancyLoss(nn.Module):
    def forward(self, clf, clf_):
        clf = nn.functional.softmax(clf, dim = 1)
        clf_ = nn.functional.softmax(clf_, dim = 1)
        loss = torch.dist(clf, clf_, p = 1)/clf.shape[1]
        # print(loss)
        return loss