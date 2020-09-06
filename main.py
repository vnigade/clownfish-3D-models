import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from utils import adjust_learning_rate
from train import train_epoch
from KD_train import train_epoch as kd_train_epoch
from KD_train import teacher_predictions
from validation import val_epoch
import test
import copy


def load_teacher_model(opt):
    assert opt.teacher_model is not None
    local_opt = copy.deepcopy(opt)
    local_opt.model = opt.teacher_model
    local_opt.model_depth = opt.teacher_model_depth
    local_opt.resnet_shortcut = opt.teacher_resnet_shortcut
    local_opt.resnext_cardinality = opt.teacher_resnext_cardinality
    local_opt.arch = '{}-{}'.format(local_opt.model, local_opt.model_depth)
    local_opt.resume_path = os.path.join(opt.root_path, opt.teacher_model_path)
    local_opt.pretrain_path = os.path.join(
        opt.root_path, opt.teacher_pretrain_path)
    model, _ = generate_model(local_opt)

    print('loading teacher model checkpoint {}'.format(local_opt.resume_path))
    checkpoint = torch.load(local_opt.resume_path)
    assert local_opt.arch == checkpoint['arch']
    res = [val for key, val in checkpoint['state_dict'].items()
           if 'module' in key]
    # if not opt.no_cuda:
    if len(res) == 0:
        # Model wrapped around DataParallel but checkpoints are not
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model


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
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        print("Total training data:", len(training_data))
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        if opt.kd_train:
            teacher_train_loader = torch.utils.data.DataLoader(
                training_data,
                batch_size=opt.teacher_batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)
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

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        assert os.path.exists(
            opt.resume_path), "Resume path does not exist".format(opt.resume_path)
        checkpoint = torch.load(opt.resume_path)
        print(opt.arch, checkpoint['arch'])
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
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('Starting run from epoch ', opt.begin_epoch)
    if opt.kd_train:
        teacher_model = load_teacher_model(opt)
        # teacher_predictions(teacher_model, teacher_train_loader, opt)
        # del teacher_train_loader
        # del teacher_model

    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            if opt.kd_train:
                # teacher_model = load_teacher_model(opt)
                # @note. This code is not tested well.
                kd_train_epoch(i, train_loader, model, teacher_model, optimizer, opt,
                               train_logger, train_batch_logger)
            else:
                if opt.model == 'mobilenet' or opt.model == 'mobilenetv2':
                    adjust_learning_rate(optimizer, i, opt)
                train_epoch(i, train_loader, model, criterion, optimizer, opt,
                            train_logger, train_batch_logger)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)

        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)

    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = VideoID()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt, test_data.class_names)
