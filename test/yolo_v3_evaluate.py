import os
import sys
MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
from nets.yolo_v3 import Yolo_v3
from dac_sdc_2020.dac_dataset import DAC_SDC_2020_Dataset
import torch
from nets.yolo_loss import YOLOLoss
from dac_sdc_2020.utils import *
import numpy as np


def GetIOU(bbox1, bbox2):
    from common.utils import bbox_iou
    b1 = torch.Tensor(bbox1).unsqueeze(0)
    b2 = torch.Tensor(bbox2).unsqueeze(0)
    iou = bbox_iou(b1, b2, x1y1x2y2=True)
    return iou.item()


def Evaluate(net, config):
    # logger = GetLogger(output_file=config['output_file'])

    print('Start evaluating.')
    net.eval()

    device = config['device']

    with open(config['class_names'], 'r') as f:
        class_names = f.read().split('\n')[:-1]

    dataset = DAC_SDC_2020_Dataset(dataset_path=config['valid_path'],
                                   class_names=class_names,
                                   img_w=config['img_w'],
                                   img_h=config['img_h'],
                                   is_training=False)
    
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                    config["yolo"]["classes"], (config["img_w"],
                                    config["img_h"]), config['device']))
    
    batch_size = config['valid_batch_size']
    results = []
    right = 0
    for step in range(0, len(dataset), batch_size):
        # print('step:', step)
        samples = []
        for ii in range(step, min(len(dataset), step + batch_size)):
            samples.append(dataset[ii])
        images = torch.cat([sample['image'].unsqueeze(0) for sample in samples], 0)
        images = images.to(device)
        
        with torch.no_grad():
            outputs = net(images)

            tt = []
            for i in range(3):
                tt.append(yolo_losses[i](outputs[i]))
            tt = torch.cat(tt, 1)

        for kk in range(len(samples)):
            t = tt[kk]
            best = t[:, 4].argmax()
            qq = t[best]
            cc = qq[5:].argmax()
            x = qq[0] / samples[kk]['image'].size(2) * samples[kk]['original_image'].shape[1]
            y = qq[1] / samples[kk]['image'].size(1) * samples[kk]['original_image'].shape[0]
            w = qq[2] / samples[kk]['image'].size(2) * samples[kk]['original_image'].shape[1]
            h = qq[3] / samples[kk]['image'].size(1) * samples[kk]['original_image'].shape[0]
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            bbox = [x1, y1, x2, y2]
            iou = GetIOU(bbox, samples[kk]['original_bbox'])
            if samples[kk]['original_label'] == class_names[cc]:
                right += 1

            # logger.info('file: ' + samples[kk]['jpg_path'])
            # logger.info('pred: ' + str(bbox) + ' ' + class_names[cc])
            # logger.info('gt: ' + str(samples[kk]['original_bbox']) + ' ' + samples[kk]['original_label'])
            # logger.info(f'iou: {iou}\n')

            result = {
                'jpg_path': samples[kk]['jpg_path'],
                'pred': {
                    'bbox': bbox,
                    'class': class_names[cc],
                },
                'gt': {
                    'bbox': samples[kk]['original_bbox'],
                    'class': samples[kk]['original_label'],
                },
                'iou': iou,
            }
            results.append(result)
        
    
    mean_iou = np.array([res['iou'] for res in results]).mean()

    # logger.info('Mean IOU: ' + str(ious.mean()))

    info = {
        'mean_iou': mean_iou,
        'accuracy': right / len(results),
        'results': results,
        'config': config,
    }
    return info


def main():
        
    config = {
        "model_params": {
            "backbone_name": "resnet_18",
            "depthwise_conv": False,
        },
        "yolo": {
            "anchors": [[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]],
            "classes": 95,
        },
        "valid_batch_size": 16,
        "confidence_threshold": 0.5,
        "valid_path": "../../dac_sdc_2020_dataset/valid",
        "class_names": "../../dac_sdc_2020_dataset/dac.names",
        "img_h": 352,
        "img_w": 640,
        "parallels": [2],
        "pretrain_snapshot": "../train/epoch_4_step_1000.pth",
        'output_file': 'output/output.txt',
        'device': 'cuda',
    }
    device = config['device']
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    net = Yolo_v3(config)
    net = net.to(device)
    res = Evaluate(net, config)
    torch.save(res, 'temp.pth')


if __name__ == '__main__':
    main()