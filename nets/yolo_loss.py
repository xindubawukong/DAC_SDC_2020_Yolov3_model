import os
import sys
MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
import torch
import torch.nn as nn
import numpy as np
import math

from dac_sdc_2020.dac_dataset import DAC_SDC_2020_Dataset
from nets.yolo_v3 import Yolo_v3
from common.utils import bbox_iou


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, device):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

        self.ignore_threshold = 0.5
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self.device = device

    def forward(self, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w
        # print(f'stride_h: {stride_h}  stride_w: {stride_w}')
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        # print('anchors:', self.anchors)
        # print('scaled_anchors:', scaled_anchors)
        prediction = input.view(bs,  self.num_anchors,
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        # print('prediction:', prediction.size())
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = torch.sigmoid(prediction[..., 4])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        # Calculate offsets for each grid
        grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_h, 1).repeat(
            bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_w, 1).t().repeat(
            bs * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
        # Calculate anchor w, h
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        # print(pred_boxes.shape)

        if targets is not None:
            mask, noobj_mask, tx, ty, tw, th, gwxh, tconf, tcls =\
                self.get_target(targets, scaled_anchors, in_w, in_h, self.ignore_threshold, pred_boxes)
            mask, noobj_mask = mask.to(self.device), noobj_mask.to(self.device)
            tx, ty, tw, th = tx.to(self.device), ty.to(self.device), tw.to(self.device), th.to(self.device)
            gwxh = gwxh.to(self.device)
            tconf, tcls = tconf.to(self.device), tcls.to(self.device)
            #  losses.
            loss_x = self.bce_loss(x * mask, tx * mask)
            loss_y = self.bce_loss(y * mask, ty * mask)
            temp = (2 - gwxh) ** 2
            loss_w = self.mse_loss(w * mask * temp, tw * mask * temp)
            loss_h = self.mse_loss(h * mask * temp, th * mask * temp)
            loss_conf = self.bce_loss(conf * mask, mask) + \
                0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)
            loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1])
            #  total loss = losses * weight
            loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

            return loss, loss_x.item(), loss_y.item(), loss_w.item(),\
                loss_h.item(), loss_conf.item(), loss_cls.item()
        else:
            # Results
            _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                                conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data

    def get_target(self, target, anchors, in_w, in_h, ignore_threshold, pred_boxes):
        bs = target.size(0)

        mask = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        gwxh = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False)
        for b in range(bs):
            # print(pred_shapes.size())
            for t in range(target.shape[1]):
                if target[b, t].sum() == 0:
                    continue
                # Convert to position relative to box
                gx = target[b, t, 1] * in_w
                gy = target[b, t, 2] * in_h
                gw = target[b, t, 3] * in_w
                gh = target[b, t, 4] * in_h
                # Get grid box indices
                gi = int(gx)
                gj = int(gy)
                # Get shape of gt box
                gt_box = torch.FloatTensor([0, 0, gw, gh]).unsqueeze(0)
                # Get shape of anchor box
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(anchors)), 1))
                # print('anchor_shapes:', anchor_shapes)
                # print('gt_box:', gt_box)
                # Calculate iou between gt and anchor shapes
                anch_ious = bbox_iou(gt_box, anchor_shapes, x1y1x2y2=False)
                # Where the overlap is larger than threshold set mask to zero (ignore)
                pred_ious = bbox_iou(torch.FloatTensor([gx, gy, gw, gh]).unsqueeze(0),
                                     pred_boxes[b, :, gj, gi].cpu(),
                                     x1y1x2y2=False)
                # print(pred_ious.size())
                noobj_mask[b, pred_ious > ignore_threshold, gj, gi] = 0
                # noobj_mask[b, anch_ious > ignore_threshold, gj, gi] = 0
                # Find the best matching anchor box
                best_n = np.argmax(anch_ious)

                # Masks
                mask[b, best_n, gj, gi] = 1
                # Coordinates
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                # Width and height
                tw[b, best_n, gj, gi] = math.log(gw/anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = math.log(gh/anchors[best_n][1] + 1e-16)
                gwxh[b, best_n, gj, gi] = torch.sigmoid(gw) * torch.sigmoid(gh)
                # object
                tconf[b, best_n, gj, gi] = 1
                # One-hot encoding of label
                tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1

        return mask, noobj_mask, tx, ty, tw, th, gwxh, tconf, tcls


def main():

    config = {
        "model_params": {
            "backbone_name": "resnet_18",
            # "backbone_name": "darknet_53",
            "depthwise_conv": True
        },
        "pretrain_snapshot": None, #'YOUR_WORKING_DIR/resnet_18/size640x352_try0/20200203210851/epoch_35_step_-1.pth',
        # "pretrain_snapshot": "../weights/yolov3_weights_pytorch.pth",
        "yolo": {
            # "anchors": [[[116, 90], [156, 198], [373, 326]],
            #             [[30, 61], [62, 45], [59, 119]],
            #             [[10, 13], [16, 30], [33, 23]]],
            "anchors": [[[117.92, 57.98], [56.57, 139.19], [163.46, 132.17]],
                        [[48.65, 45.88], [35.41, 81.16], [75.67, 76.64]],
                        [[12.90, 26.01], [29.35, 29.03], [24.03, 52.87]]],
            "classes": 95,
        },
        "lr": {
            "backbone_lr": 0.01,
            "other_lr": 0.01,
            "freeze_backbone": False,   #  freeze backbone wegiths to finetune
            "decay_gamma": 0.5,
            "decay_step": 20,           #  decay lr in every ? epochs
        },
        "optimizer": {
            "type": "sgd",
            "weight_decay": 4e-05,
        },
        "train_batch_size": 16,
        "valid_batch_size": 16,
        "train_path": "../../dac_sdc_2020_dataset/train",
        "valid_path": "../../dac_sdc_2020_dataset/valid",
        "class_names": "../../dac_sdc_2020_dataset/dac.names",
        "epochs": 50,
        "img_h": 352,
        "img_w": 640,
        "parallels": [2],                         #  config GPU device
        "working_dir": "YOUR_WORKING_DIR",              #  replace with your working dir

        # "device": "cuda",
        "device": "cpu",

        "try": 'new_anchors',

        'display_interval': 20,
    }

    net = Yolo_v3(config)

    with open(config['class_names'], 'r') as f:
        class_names = f.read().split('\n')[:-1]
    dataset = DAC_SDC_2020_Dataset(dataset_path=config['train_path'],
                                   class_names=class_names,
                                   img_w=config['img_w'],
                                   img_h=config['img_h'],
                                   is_training=False)

    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                    config["yolo"]["classes"],
                                    (config["img_w"], config["img_h"]), config['device']))
    
    sample = dataset[0]
    img = torch.unsqueeze(sample['image'], 0)

    outputs = net(img)
    o = []
    for i in range(3):
        label = torch.unsqueeze(sample['label'], 0)
        o.append(yolo_losses[i](outputs[i], label))
    
    print(o)


if __name__ == '__main__':
    main()