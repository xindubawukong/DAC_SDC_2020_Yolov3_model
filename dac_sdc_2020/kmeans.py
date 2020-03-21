from dac_dataset import DAC_SDC_2020_Dataset
import torch
from sklearn.cluster import KMeans as sklearn_KMeans
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def Save(bboxes, centers, path):
    plt.figure(figsize=(9, 9))
    x = [b[0] for b in bboxes]
    y = [b[1] for b in bboxes]
    plt.scatter(x, y, s=1)
    x = [c[0] for c in centers]
    y = [c[1] for c in centers]
    plt.scatter(x, y, marker='*', s=30, c='r')
    plt.savefig(path)


def GetIOU(b1, b2):
    if b1[0] < b2[0]:
        t = b1
        b1 = b2
        b2 = t
    # b1[0] >= b2[0]
    # print(b1, b2)
    if b1[1] >= b2[1]:
        return b2[0] * b2[1] / (b1[0] * b1[1])
    in_area = b2[0] * b1[1]
    iou = in_area / (b1[0] * b1[1] + b2[0] * b2[1] - in_area + 1e-16)
    return iou


def MyKmeans(bboxes, centers):
    K = len(centers)
    maxiter = 50
    for step in range(maxiter):
        print('step:', step)
        pp = [[] for k in range(K)]
        for wh in bboxes:
            ious = [GetIOU(wh, c) for c in centers]
            belong = np.array(ious).argmax()
            pp[belong].append(wh)
        pp = [np.array(p) for p in pp]
        centers = [(p[:, 0].mean(), p[:, 1].mean()) for p in pp]
    return centers


def main():
    
    # dataset_path = '../../dac_sdc_2020_dataset/all'
    # classes_path = '../../dac_sdc_2020_dataset/dac.names'
    # img_h = 352
    # img_w = 640
    # with open(classes_path, 'r') as f:
    #     class_names = f.read().split('\n')[:-1]
    # dataset = DAC_SDC_2020_Dataset(dataset_path, class_names=class_names, img_h=img_h, img_w=img_w, is_training=False)

    # bboxes = []
    # for ii, sample in enumerate(dataset):
    #     print(ii)
    #     label = sample['label']
    #     w = label[0][3].item() * img_w
    #     h = label[0][4].item() * img_h
    #     bboxes.append((w, h))
    # torch.save({'bboxes':bboxes}, 'bboxes.pth')

    bboxes = torch.load('bboxes.pth')['bboxes']
    
    euc_kmeans = sklearn_KMeans(n_clusters=9).fit(np.array(bboxes))
    centers = []
    for i in range(9):
        centers.append((euc_kmeans.cluster_centers_[i][0], euc_kmeans.cluster_centers_[i][1]))
    centers.sort()
    print(centers)
    Save(bboxes, centers, 'euc_kmeans.jpg')

    # centers = MyKmeans(bboxes, centers)
    # print(centers)
    # Save(bboxes, centers, 'iou_kmeans.jpg')

    centers = [[117.92, 57.98], [56.57, 139.19], [163.46, 132.17],
                [48.65, 45.88], [35.41, 81.16], [75.67, 76.64],
                [10.00, 10.00], [35.00, 27.00], [20.00, 45.00]]
    print(centers)
    Save(bboxes, centers, 'my_kmeans.jpg')



if __name__ == '__main__':
    main()