import os
import time
import cv2
from base import base
import numpy as np
from sklearn.decomposition import PCA



#pca
class ClassifiedByPCA(base):
    def __init__(self, type):
        super().__init__(type)


    def image2fea(self, folder, feas_dir, labels_dir, folder2, feas_dir2, labels_dir2):
        feas = []
        labels = []
        file_list = os.listdir(folder)
        for file in file_list:
            labels.append(int(file[0:3]))  # 保存每个特征对应的类别
            img_path = os.path.join(folder, file)  # 读取图像进行特征提取
            img = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h,w = img_gray.shape
            img_col = img_gray.reshape(h*w)
            feas.append(img_col)
        feas = np.array(feas)
        n_components = 20
        pca = PCA(n_components=n_components, svd_solver='randomized', #选择一种svd方式
          whiten=True).fit(feas)    #whiten是一种数据预处理方式，会损失一些数据信息，但可获得更好的预测结果
        feas_pca = pca.transform(feas)      #得到训练集投影系数
        labels = np.array(labels)
        np.save(feas_dir, feas_pca)
        np.save(labels_dir, labels)

        feas2 = []
        labels2 = []
        file_list = os.listdir(folder2)
        for file in file_list:
            labels2.append(int(file[0:3]))  # 保存每个特征对应的类别
            img_path = os.path.join(folder2, file)  # 读取图像进行特征提取
            img = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h,w = img_gray.shape
            img_col = img_gray.reshape(h*w)
            feas2.append(img_col)
        feas_pca2 = pca.transform(feas2)
        labels2 = np.array(labels2)
        np.save(feas_dir2, feas_pca2)
        np.save(labels_dir2, labels2)
        print('图像特征提取完毕')
        return feas_pca, labels,feas_pca2,labels2

    def image2features(self):
        if self.type == 'PCA':
            self.known_feas, self.known_labels ,self.unknown_feas, self.unknown_labels= self.image2fea(self.known_folder, self.known_feas_dir,
                                                                self.known_labels_dir, self.unknown_folder,self.unknown_feas_dir,self.unknown_labels_dir)
        else:
            raise RuntimeError('输入的特征类别有误!')


def test_pca(first=False):
    test = ClassifiedByPCA('PCA')
    if first:
        start = time.time()
        test.image2features()
        end = time.time()
        used_time = int(end - start)
        print('pca提取特征的时间是 %d 分, %d 秒.' % (used_time / 60, used_time % 60))
    else:
        test.load_feas()
    # activeList = ['l2', 'l1', 'cos']
    activeList = ['cor','che', 'l2']
    for active in activeList:
        print('度量向量间的距离选择：', active)
        test.get_all_distance(active)
        test.classify()
        test.evaluting()


if __name__ == "__main__":
    start = time.time()
    test_pca(True)
    end = time.time()
    used_time = int(end - start)
    print('使用的总时间是 %d 分, %d 秒.' % (used_time / 60, used_time % 60))
"""
图像特征提取完毕
pca提取特征的时间是 0 分, 7 秒.
度量向量间的距离选择： cor
K = 1 时的分类错误率为 0.2100 K = 1 时的分类正确率为 0.7900
K = 2 时的分类错误率为 0.2100 K = 2 时的分类正确率为 0.7900
K = 3 时的分类错误率为 0.2600 K = 3 时的分类正确率为 0.7400
K = 4 时的分类错误率为 0.2700 K = 4 时的分类正确率为 0.7300
K = 5 时的分类错误率为 0.2900 K = 5 时的分类正确率为 0.7100
度量向量间的距离选择： che
K = 1 时的分类错误率为 0.2700 K = 1 时的分类正确率为 0.7300
K = 2 时的分类错误率为 0.2700 K = 2 时的分类正确率为 0.7300
K = 3 时的分类错误率为 0.3000 K = 3 时的分类正确率为 0.7000
K = 4 时的分类错误率为 0.3100 K = 4 时的分类正确率为 0.6900
K = 5 时的分类错误率为 0.3600 K = 5 时的分类正确率为 0.6400
度量向量间的距离选择： l2
K = 1 时的分类错误率为 0.2300 K = 1 时的分类正确率为 0.7700
K = 2 时的分类错误率为 0.2300 K = 2 时的分类正确率为 0.7700
K = 3 时的分类错误率为 0.2900 K = 3 时的分类正确率为 0.7100
K = 4 时的分类错误率为 0.2900 K = 4 时的分类正确率为 0.7100
K = 5 时的分类错误率为 0.3300 K = 5 时的分类正确率为 0.6700
使用的总时间是 0 分, 7 秒.
"""