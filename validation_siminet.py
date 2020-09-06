import torch
from torch.autograd import Variable
import time
import os
import sys
import torch.nn.functional as F
from utils import AverageMeter, calculate_accuracy
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from sklearn.metrics import mean_squared_error as sk_mse

_cosine_pred = [[], []]
_siminet_pred = [[], []]


def cosine_similarity(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    alpha = sk_cosine_similarity(vec1, vec2)
    alpha = alpha[0][0]
    return alpha


def update_sim_accuracy(cosine_alpha, sim_alpha, label):
    # print("Similarity: ", cosine_alpha, sim_alpha, label)A
    print("Label", int(label))
    _cosine_pred[int(label)].append(cosine_alpha)
    _siminet_pred[int(label)].append(sim_alpha)


def print_accuracy():
    # Cosine mse
    y_true = [None, None]
    y_true[0] = [0.0 for i in range(len(_cosine_pred[0]))]
    y_true[1] = [1.0 for i in range(len(_cosine_pred[1]))]
    cosine_dis_mse = sk_mse(_cosine_pred[0], y_true[0])
    cosine_sim_mse = sk_mse(_cosine_pred[1], y_true[1])
    cosine_tot_mse = sk_mse(
        _cosine_pred[0] + _cosine_pred[1], y_true[0] + y_true[1])

    siminet_dis_mse = sk_mse(_siminet_pred[0], y_true[0])
    siminet_sim_mse = sk_mse(_siminet_pred[1], y_true[1])
    siminet_tot_mse = sk_mse(
        _siminet_pred[0] + _siminet_pred[1], y_true[0] + y_true[1])
    print("Disimilarity error: cosine={}, siminet={}",
          cosine_dis_mse, siminet_dis_mse)
    print("Similarity error: cosine={}, siminet={}",
          cosine_sim_mse, siminet_sim_mse)
    print("Total error: cosine={}, siminet={}",
          cosine_tot_mse, siminet_tot_mse)


def val_epoch(epoch, data_loader, sim_model, feature_model, criterion, opt,
              epoch_logger):
    '''
    @note. This code is not tested.
    '''
    print('train at epoch {}'.format(epoch))

    feature_model.eval()
    sim_model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    total_videos = len(data_loader)
    for i, (inputs1, inputs2, targets, _) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs1 = Variable(inputs1)
        inputs2 = Variable(inputs2)

        inputs2 = inputs2.cpu()
        outputs1 = feature_model(inputs1)
        inputs2 = inputs2.cuda()
        outputs2 = feature_model(inputs2)

        targets = Variable(targets)
        targets = targets.view(-1, 1)
        inputs = torch.cat([outputs1, outputs2], dim=-1)
        outputs = sim_model(inputs)
        outputs = torch.sigmoid(outputs)

        outputs1 = F.softmax(outputs1, dim=1)
        outputs2 = F.softmax(outputs2, dim=1)
        outputs1 = outputs1.cpu().detach().numpy()
        outputs2 = outputs2.cpu().detach().numpy()
        for j in range(len(outputs)):
            cosine_alpha = cosine_similarity(outputs1[j], outputs2[j])
            sim_alpha = outputs[j].item()
            label = targets[j].item()
            update_sim_accuracy(cosine_alpha, sim_alpha, label)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i % 100) == 0:
            print("Finished; {}/{}".format(i, total_videos))

    print_accuracy()
