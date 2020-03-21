import torch
import torch.nn as nn
from collections import OrderedDict
import os
import sys
MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.backbone import resnet
import logging
import math


class Yolo_v3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_params = config["model_params"]
        self.depthwise_conv = self.model_params['depthwise_conv']
        # backbone
        if self.model_params['backbone_name'] == 'resnet_18':
            pretrained = not self.depthwise_conv
            self.backbone = resnet.resnet18(pretrained=pretrained, depthwise_conv=self.depthwise_conv)
        else:
            assert False
        _out_filters = self.backbone.layers_out_filters
        #  embedding0
        final_out_filter0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"])
        self.embedding0 = self._make_embedding([512, 1024], _out_filters[-1], final_out_filter0)
        #  embedding1
        final_out_filter1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.embedding1_cbl = self._make_cbl(512, 256, 1)
        self.embedding1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding1 = self._make_embedding([256, 512], _out_filters[-2] + 256, final_out_filter1)
        #  embedding2
        final_out_filter2 = len(config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"])
        self.embedding2_cbl = self._make_cbl(256, 128, 1)
        self.embedding2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding2 = self._make_embedding([128, 256], _out_filters[-3] + 128, final_out_filter2)

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks - 1) // 2 if ks else 0
        if ks > 1 and self.depthwise_conv:
            return nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(_in, _in, kernel_size=ks, stride=1, padding=pad, bias=False, groups=_in)),
                ("conv", nn.Conv2d(_in, _out, kernel_size=1, stride=1, padding=0, bias=False, groups=1)),
                ("bn", nn.BatchNorm2d(_out)),
                ("relu", nn.LeakyReLU(0.1)),
            ]))
        else:
            return nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False, groups=1)),
                ("bn", nn.BatchNorm2d(_out)),
                ("relu", nn.LeakyReLU(0.1)),
            ]))

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3)])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                           stride=1, padding=0, bias=True, groups=1))
        return m

    def forward(self, x):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch
        #  backbone
        x2, x1, x0 = self.backbone(x)
        #  yolo branch 0
        out0, out0_branch = _branch(self.embedding0, x0)
        #  yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        x1_in = self.embedding1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)
        #  yolo branch 2
        x2_in = self.embedding2_cbl(out1_branch)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)
        return out0, out1, out2
    
    def LoadPretrainedModel(self, path):
        if self.config['device'] == 'cpu':
            info = torch.load(path, map_location='cpu')
        else:
            info = torch.load(path)
        
        if 'config' in info.keys():
            assert info['config']['model_params'] == self.config['model_params']
            assert info['config']['yolo'] == self.config['yolo']
            assert info['config']['img_h'] == self.config['img_h']
            assert info['config']['img_w'] == self.config['img_w']
        
        state_dict = info['state_dict']
        
        # nn.DataParallel后的模型，key会前面多一个"module."
        if list(state_dict.keys())[0].startswith('module.'):
            new_dict = {}
            for key in state_dict.keys():
                new_dict[key[7:]] = state_dict[key]
            state_dict = new_dict

        # # 如果类的个数不匹配，最后几个输出层会不一致
        # if state_dict['embedding0.conv_out.weight'].size() != self.state_dict()['embedding0.conv_out.weight'].size():
        #     logging.info(f'Class number mismatch. Some parameters not loaded.')
        #     new_dict = {}
        #     for key in state_dict.keys():
        #         if key in ['embedding0.conv_out.weight', 'embedding0.conv_out.bias',
        #                    'embedding1.conv_out.weight', 'embedding1.conv_out.bias',
        #                    'embedding2.conv_out.weight', 'embedding2.conv_out.bias']:
        #             continue
        #         new_dict[key] = state_dict[key]
        #     state_dict = new_dict

        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)


if __name__ == "__main__":
    dxy_training_config = {
        "model_params": {
            "backbone_name": "resnet_18",
            # "backbone_name": "darknet_53",
            "depthwise_conv": True,
        },
        "pretrain_snapshot": '../train/YOUR_WORKING_DIR/resnet_18/size640x352_try_use_1_cpu/20200206113629/epoch_25_net.pth',
        # "pretrain_snapshot": "../weights/yolov3_weights_pytorch.pth",
        "yolo": {
            "anchors": [[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]],
            "classes": 95,
        },
        "lr": {
            "backbone_lr": 0.001,
            "other_lr": 0.01,
            "freeze_backbone": False,   #  freeze backbone wegiths to finetune
            "decay_gamma": 0.1,
            "decay_step": 20,           #  decay lr in every ? epochs
        },
        "optimizer": {
            "type": "sgd",
            "weight_decay": 4e-05,
        },
        "batch_size": 1,
        "train_path": "../../dac_sdc_2020_dataset/train",
        "class_names": "../../dac_sdc_2020_dataset/dac.names",
        "epochs": 100,
        "img_h": 352,
        "img_w": 640,
        "parallels": [0,1,2,3],                         #  config GPU device
        "working_dir": "YOUR_WORKING_DIR",              #  replace with your working dir
        "evaluate_type": "", 
        "try": 0,
        "export_onnx": False,
        "device": "cpu"
    }

    config = dxy_training_config

    device = config['device']

    config['model_params']['depthwise_conv'] = False
    m1 = Yolo_v3(config)
    m1.to(device)
    config['model_params']['depthwise_conv'] = True
    m2 = Yolo_v3(config)
    m2.to(device)

    cnt1 = 0
    for param in m1.parameters():
        cnt1 += param.view(-1).size(0)
    cnt2 = 0
    for param in m2.parameters():
        cnt2 += param.view(-1).size(0)
    print(cnt1, cnt2)

    img = torch.randn(1, 3, config['img_h'], config['img_w']).to(device)
    import time
    start = time.time()
    o1 = m1(img)
    print('size:', o1[0].size(), o1[1].size(), o1[2].size())
    end = time.time()
    print('m1:', end - start)
    start = time.time()
    o2 = m2(img)
    end = time.time()
    print('m2:', end - start)
    
    # for name, param in m2.named_parameters():
    #     print(name, param.size())
    
    cnt2 = 0
    for param in m2.parameters():
        if param.view(param.size(0), -1).size(1) > 20:
            cnt2 += param.size(0) * 8
        else:
            cnt2 += param.view(-1).size(0)
    print(cnt1, cnt2)