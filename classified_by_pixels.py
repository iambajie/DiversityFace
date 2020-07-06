
import os
import operator
import time

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from base import base

#RGB
class ClassifiedByPixels(base):
    def __init__(self, type):
        super().__init__(type)
        
    def rgb_feature(self, image_path):
        """
        计算图像的RGB颜色直方图,并经过归一化和平滑处理生成一个特征向量.
        Args:
            image_path: 图像的路径.
        Return:
            numpy包中的array类型的特征向量.
        Raise:
            当输入的图像路径无效时,抛出异常.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError('hist_feature:path of image is invaild!')
        hist0 = cv2.calcHist([img], [0], None, [256], [0., 255.])
        hist1 = cv2.calcHist([img], [1], None, [256], [0., 255.])
        hist2 = cv2.calcHist([img], [2], None, [256], [0., 255.])
        cv2.normalize(hist0, hist0)
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        hist = []
        hist.extend(hist0.flatten())
        hist.extend(hist1.flatten())
        hist.extend(hist2.flatten())
        return np.array(hist)


    def image2fea(self, folder, feas_dir, labels_dir, feature_extractor):
        '''
        将folder文件下的图片按照way方法转换成特征向量.
        :param folder: 文件夹路径
        :param feas_dir: 存储特征向量的路径
        :param labels_dir: 存储图像标签的路径
        :param way: 图像特征编码方式,此处只使用rgb和gray两种方式
        '''
        feas = []
        labels = []
        file_list = os.listdir(folder)
        for file in file_list:
            labels.append(int(file[0:3]))   #   保存每个特征对应的类别
            img_path = os.path.join(folder, file)   # 读取图像进行特征提取
            fea = feature_extractor(img_path)
            feas.append(fea)
        feas = np.array(feas)
        labels = np.array(labels)
        np.save(feas_dir, feas)
        np.save(labels_dir, labels)
        print('图像特征提取完毕')
        return feas, labels
        

    def image2features(self):
        if self.type == 'RGB':
            self.known_feas, self.known_labels = self.image2fea(self.known_folder, self.known_feas_dir, self.known_labels_dir, self.rgb_feature)
            self.unknown_feas, self.unknown_labels = self.image2fea(folder=self.unknown_folder, 
                    feas_dir=self.unknown_feas_dir, labels_dir=self.unknown_labels_dir,feature_extractor=self.rgb_feature)

        else:
            raise RuntimeError('输入的特征类别有误!')


def test_rgb(first=False):
    test = ClassifiedByPixels('RGB')
    if first:
        start = time.time()
        test.image2features()
        end = time.time()
        used_time = int(end - start)
        print('rgb提取特征的时间是 %d 分, %d 秒.'%(used_time/60, used_time%60))
    else:
        test.load_feas()
    activeList = ['cor','che', 'l2']
    for active in activeList:
        print('度量向量间的距离选择：', active)
        test.get_all_distance(active)
        test.classify()
        test.evaluting()

    
if __name__ == "__main__":
    start = time.time()
    test_rgb(True)
    end = time.time()
    used_time = int(end - start)
    print('使用的总时间是 %d 分, %d 秒.' % (used_time / 60, used_time % 60))

"""
图像特征提取完毕
图像特征提取完毕
rgb提取特征的时间是 0 分, 0 秒.
度量向量间的距离选择： cor
K = 1 时的分类错误率为 0.3600 K = 1 时的分类正确率为 0.6400
K = 2 时的分类错误率为 0.3600 K = 2 时的分类正确率为 0.6400
K = 3 时的分类错误率为 0.4700 K = 3 时的分类正确率为 0.5300
K = 4 时的分类错误率为 0.5200 K = 4 时的分类正确率为 0.4800
K = 5 时的分类错误率为 0.5400 K = 5 时的分类正确率为 0.4600
度量向量间的距离选择： che
K = 1 时的分类错误率为 0.4600 K = 1 时的分类正确率为 0.5400
K = 2 时的分类错误率为 0.4600 K = 2 时的分类正确率为 0.5400
K = 3 时的分类错误率为 0.4700 K = 3 时的分类正确率为 0.5300
K = 4 时的分类错误率为 0.5300 K = 4 时的分类正确率为 0.4700
K = 5 时的分类错误率为 0.5600 K = 5 时的分类正确率为 0.4400
度量向量间的距离选择： l2
K = 1 时的分类错误率为 0.3700 K = 1 时的分类正确率为 0.6300
K = 2 时的分类错误率为 0.3700 K = 2 时的分类正确率为 0.6300
K = 3 时的分类错误率为 0.4400 K = 3 时的分类正确率为 0.5600
K = 4 时的分类错误率为 0.5200 K = 4 时的分类正确率为 0.4800
K = 5 时的分类错误率为 0.5300 K = 5 时的分类正确率为 0.4700
使用的总时间是 0 分, 0 秒.
"""
