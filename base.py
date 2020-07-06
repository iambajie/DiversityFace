import os
import cv2
import operator
import numpy as np
from scipy.spatial.distance import cdist

#基本方法：
#1.设置读取的图片路径、特征向量保存和读取路径
#2.设置knn分类器及分类距离
class base(object):
    def __init__(self, type):
        self.unknown_folder = './unused/'
        self.known_folder = './used/'
        __TYPE = ['RGB', 'PCA', 'VGG','VGGFC']
        if type not in __TYPE:
            raise RuntimeError('输入的参数:type无效!')
        self.type = type
        self.root_dir = os.path.join(os.getcwd(), type)
        feas_dir = self.root_dir

        # 此处是为了直接使用GoogLeNet和VGG提取的特征
        if type in ['VGGFC']:
            feas_dir = os.path.join(os.getcwd(), type[:-2])

        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)

        self.unknown_feas_dir = os.path.join(feas_dir, 'unknown_feas.npy')
        self.unknown_labels_dir = os.path.join(feas_dir, 'unknown_labels.npy')

        self.known_feas_dir = os.path.join(feas_dir, 'known_feas.npy')
        self.known_labels_dir = os.path.join(feas_dir, 'known_labels.npy')

    def load_feas(self):
        self.known_feas = np.load(self.known_feas_dir)
        self.known_labels = np.load(self.known_labels_dir)
        self.unknown_feas = np.load(self.unknown_feas_dir)
        self.unknown_labels = np.load(self.unknown_labels_dir)

    # 选择不同的距离度量方法
    def get_all_distance(self, active):
        if active == 'l2':
            self.get_all_distanceL2()
        if active == 'cor':
            self.get_all_distanceCor()
        if active == 'che':
            self.get_all_distanceChe()
    def get_all_distanceL2(self):
        distance = []
        for unknown_fea in self.unknown_feas:
            dis = cdist(np.expand_dims(unknown_fea, axis=0), self.known_feas, metric='euclidean')[0]
            distance.append(dis)
        self.distances = np.stack(distance)

    def get_all_distanceCor(self):
        distance = []
        for unknown_fea in self.unknown_feas:
            dis = cdist(np.expand_dims(unknown_fea, axis=0), self.known_feas, metric='correlation')[0]
            distance.append(dis)
        self.distances = np.stack(distance)


    def get_all_distanceChe(self):
        distance = []
        for unknown_fea in self.unknown_feas:
            dis = cdist(np.expand_dims(unknown_fea, axis=0), self.known_feas, metric='chebyshev')[0]
            distance.append(dis)
        self.distances = np.stack(distance)

    def classify_by_knn(self, K=5):
        error_cnt = 0.0
        for index, dis in zip(range(len(self.unknown_labels)), self.distances):
            sorted_dis_indices = dis.argsort()
            class_cnt = {}
            for i in range(K):
                classLabel = self.known_labels[sorted_dis_indices[i]]
                class_cnt[classLabel] = class_cnt.get(classLabel, 0) + 1
            sorted_class_cnt = sorted(class_cnt.items(), key=operator.itemgetter(1), reverse=True)
            predicted_label = sorted_class_cnt[0][0]
            if predicted_label != self.unknown_labels[index]:
                error_cnt += 1.0
        error_ratio = error_cnt / len(self.unknown_labels)
        return error_ratio

    def classify(self, K=[1,2,3,4,5]):
        self.error_ratio = {}
        for k in K:
            self.error_ratio[k] = self.classify_by_knn(K=k)

    def evaluting(self):
        for key in self.error_ratio.keys():
            print('K = %d 时的分类错误率为 %0.4f' % (key, self.error_ratio[key]), end=' ')
            print('K = %d 时的分类正确率为 %0.4f' % (key, 1 - self.error_ratio[key]))

