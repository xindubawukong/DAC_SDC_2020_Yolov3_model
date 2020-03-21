import os


yolo_v3_train_config = {
    "model_params": {
        "backbone_name": "resnet_18",
        "depthwise_conv": True,
    },
    "pretrain_snapshot": None, #'/home1/dingxy/dxy_DAC_SDC_2020_model/train/YOUR_WORKING_DIR/resnet_18/size640x352_try_use_1_cpu/20200206113629/epoch_49_net.pth',
    "yolo": {
        # 这些anchor的大小是在352*640分辨率下的，所以最下面要处理一下
        "anchors": [[[117.92, 57.98], [56.57, 139.19], [163.46, 132.17]],
                    [[48.65, 45.88], [35.41, 81.16], [75.67, 76.64]],
                    [[12.90, 26.01], [29.35, 29.03], [24.03, 52.87]]],
        "classes": 95,  # DAC SDC 2020 的目标种类数
    },
    "lr": {  # 最开始的learning rate是0.01
        "backbone_lr": 0.01,
        "other_lr": 0.01,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.6,
        "decay_step": 10,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "sgd",
        "weight_decay": 4e-05,
    },
    "train_batch_size": 16,
    "valid_batch_size": 16,
    "train_path": os.path.join('..', '..', 'dac_sdc_2020_dataset', 'train'),  # "../../dac_sdc_2020_dataset/train",
    "valid_path": os.path.join('..', '..', 'dac_sdc_2020_dataset', 'valid'),  # "../../dac_sdc_2020_dataset/valid",
    "class_names": "../../dac_sdc_2020_dataset/dac.names",

    "img_h": 352,   # 网络输入图片的h
    "img_w": 640,   # 网络输入图片的w

    "epochs": 150,

    "parallels": [0],  # 训练时占用哪些显卡，即CUDA_VISIBLE_DEVICES
    "max_threads": 2,  # 训练时占用多少个cpu，防止都占满把服务器卡爆了

    "working_dir": "YOUR_WORKING_DIR",              #  replace with your working dir

    "device": "cuda",  # 使用GPU训练
    # "device": "cpu",  # 使用CPU训练

    "try": 'temp',  # 这个参数只影响sub_working_dir的名字

    'display_interval': 10,
}


for i in range(3):
    for j in range(3):
        anchor = yolo_v3_train_config['yolo']['anchors'][i][j]
        anchor[0] = anchor[0] * yolo_v3_train_config['img_w'] / 640
        anchor[1] = anchor[1] * yolo_v3_train_config['img_h'] / 352