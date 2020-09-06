import torch
from torch.autograd import Variable
import time
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import AverageMeter, calculate_accuracy

_teacher_outputs_dict = {}


def loss_kd(outputs, teacher_outputs, targets):
    # "alpha": 0.95,
    # "temperature": 6,
    alpha = 0.95
    T = 6
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
        F.cross_entropy(outputs, targets) * (1. - alpha)

    return KD_loss


def teacher_predictions(model, data_loader, opt):
    assert not _teacher_outputs_dict
    model.eval()
    data_len = len(data_loader)
    for i, (inputs, targets, indices) in enumerate(data_loader):
        if not opt.no_cuda:
            inputs, targets = inputs.cuda(async=True), targets.cuda(async=True)

        inputs = Variable(inputs)
        targets = Variable(targets)
        teacher_output = model(inputs).data.cpu().numpy()
        print("Teacher prediction ({}/{})".format(i, data_len))
        # teacher_outputs.append(teacher_output)
        for i in range(len(indices)):
            _teacher_outputs_dict[indices[i].item()] = teacher_output[i]
    return


def train_epoch(epoch, data_loader, model, teacher_model, optimizer, opt,
                epoch_logger, batch_logger):
    '''
    @note. This function is not tested. And may be implemented or conceptualized incorrectly.
    '''
    print('train at epoch {}'.format(epoch))
    model.train()
    teacher_model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    # Get teacher model outputs
    # if not _teacher_outputs_dict:
    #    print("Executing teacher model...")
    #    for i, (inputs, targets, indices) in enumerate(data_loader):
    #        if not opt.no_cuda:
    #            inputs, targets = inputs.cuda(async=True), targets.cuda(async=True)

    #        inputs = Variable(inputs)
    #        targets = Variable(targets)
    #        teacher_output = teacher_model(inputs).data.cpu().numpy()
    # print("Teacher output shape", teacher_output.shape)
    # teacher_outputs.append(teacher_output)
    #        for i in range(len(indices)):
    #            _teacher_outputs_dict[indices[i].item()] = teacher_output[i]
    # else:
    #    print("Skipping teacher model execution...")

    # Check if it is teacher output
    # print("teacher output", _teacher_outputs_dict.keys())

    end_time = time.time()
    for i, (inputs, targets, indices) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        if not opt.no_cuda:
            inputs = inputs.cuda(async=True)
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = model(inputs)
        # loss = criterion(outputs, targets)
        # teacher_outputs = []
        # for idx in indices:
        # print("idx, teacher output", idx, _teacher_outputs_dict[idx.item()])
        #    teacher_outputs.append(_teacher_outputs_dict[idx.item()])
        # teacher_outputs = np.asarray(teacher_outputs)

        # teacher_output = torch.from_numpy(teacher_outputs)
        # if not opt.no_cuda:
        #    teacher_output = teacher_output.cuda(async=True)
        #    teacher_output = Variable(teacher_output, requires_grad=False)

        # get teacher output.
        teacher_output = teacher_model(inputs)
        loss = loss_kd(outputs, teacher_output, targets)
        acc = calculate_accuracy(outputs, targets)

        # losses.update(loss.data[0], inputs.size(0))
        losses.update(loss.data.item(), inputs.size(0))

        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        if not opt.no_cuda:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
