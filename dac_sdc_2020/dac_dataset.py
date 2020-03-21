import os
import sys
MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import cv2
from dac_sdc_2020.utils import *
import numpy as np
import random


class DAC_SDC_2020_Dataset(Dataset):

    def __init__(self, dataset_path, class_names, img_w, img_h, is_training=False):
        self.dataset_path = dataset_path
        self.class_names = class_names
        # assert len(self.class_names) == 95
        self.img_w = img_w
        self.img_h = img_h
        self.is_training = is_training
        self.all_data = []
        for file in os.listdir(self.dataset_path):
            if file.endswith('.jpg') and not file.startswith('.'):
                jpg = os.path.join(self.dataset_path, file)
                xml = os.path.join(self.dataset_path, file[:-4] + '.xml')
                self.all_data.append((jpg, xml))
        self.all_data.sort()
        print(f'The dataset has {len(self.all_data)} images.')
        self.aug_ratio = 0.5
        # self.all_data = self.all_data[:10]  # debug
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        jpg, xml = self.all_data[index]
        img = read_image(jpg)
        label, bbox = self.ParseXml(xml)
        sample = {
            'jpg_path': jpg,
            'xml_path': xml,
            'original_image': img.copy(),
            'original_label': label,
            'original_bbox': bbox,
        }
        if label != 0:
            label = self.class_names.index(label)  # str -> int
        
        new_img = img.copy()
        h0, w0, _ = new_img.shape

        # if self.is_training:
        #     new_img, bbox = self.RandomFlip(new_img, bbox)
        #     new_img = self.RandomBlur(new_img)
        #     new_img = self.RandomBrightness(new_img)

        x = (bbox[0] + bbox[2]) / 2.0 / w0
        y = (bbox[1] + bbox[3]) / 2.0 / h0
        w = (bbox[2] - bbox[0]) * 1.0 / w0
        h = (bbox[3] - bbox[1]) * 1.0 / h0
        new_label = torch.Tensor([[label, x, y, w, h]])  # 这样是因为，dac数据集里一张图只有一个目标

        new_img = cv2.resize(new_img, (self.img_w, self.img_h))
        new_img = np.transpose(new_img, (2, 0, 1)).astype(np.float32)
        new_img /= 255
        new_img = torch.Tensor(new_img)

        sample['image'] = new_img
        sample['label'] = new_label

        return sample


    # Return label and bounding box.
    def ParseXml(self, xml_path):
        if not os.path.exists(xml_path):
            return 0, [0, 0, 0, 0]
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj = root.find('object')
        label = obj.find('name').text
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)
        return label, [xmin, ymin, xmax, ymax]
    

    def RandomFlip(self, img, box):
        # 水平翻转
        if random.random() < self.aug_ratio:
            new_img = np.fliplr(img).copy()
            w = img.shape[1]
            x1 = box[0]
            x2 = box[2]
            box[0] = w - x2
            box[2] = w - x1
            return new_img, box
        return img, box
    

    def RandomBlur(self, img):
        if random.random() < self.aug_ratio:
            img = cv2.blur(img, (2, 2))
        return img

    
    def RandomBrightness(self, rgb):
        if random.random() < self.aug_ratio:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            adjust = random.uniform(0.8, 1.2)
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb


if __name__ == '__main__':
    dataset_path = '../../dac_sdc_2020_dataset/valid'
    classes_path = '../../dac_sdc_2020_dataset/dac.names'
    with open(classes_path, 'r') as f:
        class_names = f.read().split('\n')[:-1]
    dataset = DAC_SDC_2020_Dataset(dataset_path, class_names=class_names, img_h=352, img_w=352, is_training=False)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=16,
                                             shuffle=True)
    
    for samples in dataloader:
        print(samples['label'].size())
        break
    
    # dataset.aug_ratio = 1
    # img = read_image('/Users/xdbwk/Desktop/0052.jpg')
    # bbox = [0, 30, 50, 70]
    # img, bbox = dataset.RandomFlip(img, bbox)
    # save_image(img, '/Users/xdbwk/Desktop/temp.jpg')
    # print(bbox)