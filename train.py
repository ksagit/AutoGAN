# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
import models
import datasets
from functions import train, validate, LinearLrDecay, load_params, copy_params
from utils.utils import set_log_dir, save_checkpoint, create_logger
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception

import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
import subprocess


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import yaml
from ncsn.models.cond_refinenet_dilated import CondRefineNetDilated
from torch.nn import DataParallel
import torch


def get_ncsn_model():
    state_dict = torch.load('checkpoint.pth')[0]
    config = yaml.load(open('config.yml'))
    model = DataParallel(CondRefineNetDilated(config))
    # model.load_state_dict(state_dict)
    return model.module.train().cuda()


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # import network
    gen_net = eval('models.'+args.gen_model+'.Generator')(args=args).cuda()
    # dis_net = eval('models.'+args.dis_model+'.Discriminator')(args=args).cuda()

    # Get score models for data and mixture distributions
    score_p_d = get_ncsn_model()
    score_p_m = get_ncsn_model()

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    torch.manual_seed(0)
    gen_net.apply(weights_init)
    print(sum(param.sum().item() for param in gen_net.parameters()))
    # dis_net.apply(weights_init)

    torch.manual_seed(0)
    gen_net.apply(weights_init)
    print(sum(param.sum().item() for param in gen_net.parameters()))

    # set optimizer
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic)

    # p_d doesn't need an optimizer
    score_p_m_optim = torch.optim.Adam(score_p_m.parameters(), lr=.001, weight_decay=0.000,
                                       betas=(.9, 0.999), amsgrad=False)

    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)

    # epoch number for dis_net
    args.max_epoch = args.max_epoch * args.n_critic
    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
    gen_avg_param = copy_params(gen_net)
    start_epoch = 0
    best_fid = 1e4

    # set writer
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint['best_fid']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        avg_gen_net = deepcopy(gen_net)
        avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(avg_gen_net)
        del avg_gen_net

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    # Version specific details
    environment = subprocess.check_output(["conda", "list"]).decode()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()
    logger.info("Training with environment: {}\n\nModel revision: {}\n\n".format(environment, revision))

    # train loop
    for epoch in tqdm(range(int(start_epoch), int(args.max_epoch)), desc='total progress'):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        train(args, gen_net, score_p_d, score_p_m, gen_optimizer, score_p_m_optim, gen_avg_param, train_loader, epoch, writer_dict,
              lr_schedulers)

        if True:  # epoch and epoch % args.val_freq == 0 or epoch == int(args.max_epoch)-1:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param)

            # inception_score, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict)
            inception_score, fid_score = 0, 0

            torch.cuda.empty_cache()
            logger.info(f'Inception score: {inception_score}, FID score: {fid_score} || @ epoch {epoch}.')
            load_params(gen_net, backup_param)
            if fid_score < best_fid:
                best_fid = fid_score
                is_best = True
            else:
                is_best = False
        else:
            is_best = False

        avg_gen_net = deepcopy(gen_net)
        load_params(avg_gen_net, gen_avg_param)
        save_checkpoint({
            'epoch': epoch + 1,
            'gen_model': args.gen_model,
            'dis_model': args.dis_model,
            'gen_state_dict': gen_net.state_dict(),
            'avg_gen_state_dict': avg_gen_net.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'best_fid': best_fid,
            'path_helper': args.path_helper
        }, is_best, args.path_helper['ckpt_path'])
        del avg_gen_net


if __name__ == '__main__':
    main()
