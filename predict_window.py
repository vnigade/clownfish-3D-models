import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model, generate_sim_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test_windows as test
import timeit
from utils import AverageMeter

torch.backends.cudnn.enabled = False


def model_time(model, opt):
    avg_time = AverageMeter()
    for i in range(100):
        input = torch.randn(
            [1, 3, opt.sample_duration, opt.sample_size, opt.sample_size]).cuda()
        start_time = timeit.default_timer()
        with torch.no_grad():
            _ = model(input)
            torch.cuda.synchronize()
            prediction_time = timeit.default_timer() - start_time
            if i > 10:
                avg_time.update(prediction_time * 1000)
    print("Model inference time:", avg_time.avg)


if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    if opt.model == 'mobilenet' or opt.model == 'mobilenetv2':
        opt.arch = opt.model
    else:
        opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    model_time(model, opt)
    # criterion = nn.CrossEntropyLoss()
    # if not opt.no_cuda:
    #    criterion = criterion.cuda()
    criterion = None

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    assert opt.no_train
    # assert opt.no_val

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        if opt.no_cuda_predict:
            checkpoint = torch.load(opt.resume_path, map_location='cpu')
        else:
            checkpoint = torch.load(opt.resume_path)

        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        res = [val for key, val in checkpoint['state_dict'].items()
               if 'module' in key]
        # if not opt.no_cuda:
        if len(res) == 0:
            # Model wrapped around DataParallel but checkpoints are not
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])

        # if not opt.no_train:
        #    optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    # Perform validation to check the accuracy on clipped videos
    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])
        validation_loss = val_epoch(opt.begin_epoch, val_loader, model, criterion, opt,
                                    val_logger)

    # similarity model for testing
    sim_model = None
    if opt.resume_path_sim != '':
        opt.n_finetune_classes = opt.n_classes
        sim_model, _ = generate_sim_model(opt)
        print('loading similarity model checkpoint {}'.format(opt.resume_path_sim))
        checkpoint = torch.load(opt.resume_path_sim)
        print(opt.arch, checkpoint['arch'])
        assert opt.arch == checkpoint['arch']
        if not opt.no_cuda:
            sim_model.module.load_state_dict(checkpoint['state_dict'])
        else:
            sim_model.load_state_dict(checkpoint['state_dict'])
        sim_model.eval()

    if opt.test:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.window_size)
        target_transform = ClassLabel()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        # test.test(test_loader, model, opt, test_data.class_names, sim_model=sim_model)
        print("Predicting batch size.. 32")
        test.test_batch(test_data, model, opt, test_data.class_names, 32)
