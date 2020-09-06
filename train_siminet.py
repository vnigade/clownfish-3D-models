import torch
from torch.autograd import Variable
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, sim_model, feature_model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    feature_model.eval()
    sim_model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
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
        # outputs = torch.sigmoid(outputs) # comment this when using loss with logits
        # print("Inputs", inputs.shape, "Outputs", outputs.shape, "Targets", targets.shape)
        loss = criterion(outputs, targets)
        # acc = calculate_accuracy(outputs, targets)

        # losses.update(loss.data[0], inputs.size(0))
        losses.update(loss.data.item(), inputs.size(0))

        # accuracies.update(acc, inputs.size(0))

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
            state_dict = sim_model.module.state_dict()
        else:
            state_dict = sim_model.state_dict()

        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
