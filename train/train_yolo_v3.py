# coding='utf-8'

import os
import sys
MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))

from train.train_yolo_v3_config import yolo_v3_train_config

os.environ["MKL_NUM_THREADS"] = str(yolo_v3_train_config['max_threads'])
os.environ["NUMEXPR_NUM_THREADS"] = str(yolo_v3_train_config['max_threads'])
os.environ["OMP_NUM_THREADS"] = str(yolo_v3_train_config['max_threads'])

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, yolo_v3_train_config["parallels"]))

import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from nets.yolo_v3 import Yolo_v3
from nets.yolo_loss import YOLOLoss
from dac_sdc_2020.dac_dataset import DAC_SDC_2020_Dataset
from test.yolo_v3_evaluate import Evaluate


def train(config):
    device = config['device']

    net = Yolo_v3(config)

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        net.LoadPretrainedModel(config['pretrain_snapshot'])
        logging.info("Loaded checkpoint from {}".format(config["pretrain_snapshot"]))
    else:
        logging.info('No pretrained model to use.')

    # Optimizer and learning rate
    optimizer = _get_optimizer(config, net)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["lr"]["decay_step"],
        gamma=config["lr"]["decay_gamma"])
    
    # Set data parallel
    if device == 'cuda':
        net = nn.DataParallel(net)
    net = net.to(device)
    
    params = [param for param in net.parameters() if len(param.size()) == 4 and param.view(param.size(0), -1).size(1) > 20]

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                    config["yolo"]["classes"],
                                    (config["img_w"], config["img_h"]),
                                    device))

    with open(config['class_names'], 'r') as f:
        class_names = f.read().split('\n')[:-1]
    print(class_names)
    dataset = DAC_SDC_2020_Dataset(dataset_path=config['train_path'],
                                   class_names=class_names,
                                   img_w=config['img_w'],
                                   img_h=config['img_h'],
                                   is_training=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=config['train_batch_size'],
                                             shuffle=True)
    print('here')
    # Start the training loop
    logging.info("Start training.")
    for epoch in range(config["epochs"]):
        logging.info(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~ epoch = {epoch} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        net.train()
        for step, samples in enumerate(dataloader):
            start_time = time.time()

            images, labels = samples["image"], samples["label"]
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)

            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = []
            for _ in range(len(losses_name)):
                losses.append([])
            for i in range(3):
                _loss_item = yolo_losses[i](outputs[i], labels)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]

            loss.backward()
            optimizer.step()

            if step > 0 and step % config['display_interval']  == 0:
                _loss = loss.item()
                duration = float(time.time() - start_time)
                example_per_second = config["train_batch_size"] / duration
                lr = optimizer.param_groups[0]['lr']
                logging.info(
                    "epoch [%.3d] iter = %d loss = %.6f example/sec = %.3f lr = %.5f "%
                    (epoch, step, _loss, example_per_second, lr)
                )

        info = Evaluate(net, config)
        logging.info('net   iou: ' + str(info['mean_iou']) + '  acc: ' + str(info['accuracy']))
        info['state_dict'] = net.state_dict()
        info['epoch'] = epoch
        checkpoint_path = os.path.join(config["sub_working_dir"], 'epoch_' + str(epoch) + '_net.pth')
        torch.save(info, checkpoint_path)
        logging.info("Model checkpoint saved to %s" % checkpoint_path)

        lr_scheduler.step()

    logging.info("Bye~")


def _get_optimizer(config, net):
    optimizer = None

    # Assign different lr for each layer
    params = None
    base_params = list(
        map(id, net.backbone.parameters())
    )
    logits_params = filter(lambda p: id(p) not in base_params, net.parameters())

    if not config["lr"]["freeze_backbone"]:
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
            {"params": net.backbone.parameters(), "lr": config["lr"]["backbone_lr"]},
        ]
    else:
        logging.info("freeze backbone's parameters.")
        for p in net.backbone.parameters():
            p.requires_grad = False
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
        ]

    # Initialize optimizer class
    if config["optimizer"]["type"] == "adam":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"])
    elif config["optimizer"]["type"] == "amsgrad":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"],
                               amsgrad=True)
    elif config["optimizer"]["type"] == "rmsprop":
        optimizer = optim.RMSprop(params, weight_decay=config["optimizer"]["weight_decay"])
    else:
        # Default to sgd
        logging.info("Using SGD optimizer.")
        optimizer = optim.SGD(params, momentum=0.9,
                              weight_decay=config["optimizer"]["weight_decay"],
                              nesterov=(config["optimizer"]["type"] == "nesterov"))

    return optimizer

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    config = yolo_v3_train_config

    # Create sub_working_dir
    sub_working_dir = '{}/{}_size{}x{}_try_{}/{}'.format(
        config['working_dir'], config['model_params']['backbone_name'], 
        config['img_w'], config['img_h'], config['try'],
        time.strftime("%Y%m%d%H%M%S", time.localtime()))
    if not os.path.exists(sub_working_dir):
        os.makedirs(sub_working_dir)
    config["sub_working_dir"] = sub_working_dir
    logging.info("sub working dir: %s" % sub_working_dir)

    torch.save(config, os.path.join(sub_working_dir, 'config.pth'))
    with open(os.path.join(sub_working_dir, 'config.txt'), 'w') as f:
        for key, value in config.items():
            f.write(str(key) + ': ' + str(value) + '\n')
    
    if config['device'] == 'cuda':
        assert torch.cuda.is_available()
    logging.info('Device: ' + config['device'])

    train(config)

if __name__ == "__main__":
    main()
