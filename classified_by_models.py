import os
import time

import numpy as np

from keras.models import Model, load_model
from keras.applications import vgg16, inception_v3
from keras.preprocessing import image

from base import base

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# # (nothing gets printed in Jupyter, only if you run it standalone)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Kera


#神经网络
class ClassifiedByModels(base):
    def __init__(self, type):
        super().__init__(type)
        if self.type is 'VGG':
            self.model_path = './weights/VGG.h5'
            self.layer_name = 'fc2'
        else:
            raise RuntimeError('输入的参数:type无效!')

    def get_model(self):
        base_model = load_model(self.model_path)
        self.model = Model(input=base_model.input, output=base_model.get_layer(self.layer_name).output)

    def vgg_feature(self, image_path):
        img = image.load_img(path=image_path, target_size=(224, 224))
        img = image.img_to_array(img=img)
        img = np.expand_dims(img, axis=0)
        img = vgg16.preprocess_input(img)
        fea = self.model.predict(img)[0]  # [[2,2,2,2]] --> [2, 2, 2, 2]
        return fea

    def get_image_fea(self, folder, feas_dir, labels_dir, feature_extractor):
        print('提取图像特征中.....')
        feas = []
        labels = []
        file_list = os.listdir(folder)
        for file in file_list:
            labels.append(int(file[0:3]))  # 保存每个特征对应的类别
            img_path = os.path.join(folder, file)  # 读取图像进行特征提取
            fea = feature_extractor(img_path)
            feas.append(fea)
        feas = np.array(feas)
        labels = np.array(labels)
        self.auto_norm(feas)
        np.save(feas_dir, feas)
        np.save(labels_dir, labels)
        print('图像特征提取完毕')
        return feas, labels

    def auto_norm(self, data_set):
        # ndarray.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
        data_min = data_set.min(0)
        data_max = data_set.max(0)
        data_range = data_max - data_min  # 此时的维度与原始数据不同,需要进行维度的扩展
        data_range = np.tile(data_range, (data_set.shape[0], 1))
        data_min = np.tile(data_min, (data_set.shape[0], 1))
        return (data_set - data_min) / data_range

    def image2features(self):
        if self.type == 'VGG':
            self.known_feas, self.known_labels = self.get_image_fea(folder=self.known_folder,
                                                                    feas_dir=self.known_feas_dir,
                                                                    labels_dir=self.known_labels_dir,
                                                                    feature_extractor=self.vgg_feature)

            self.unknown_feas, self.unknown_labels = self.get_image_fea(folder=self.unknown_folder,
                                                                        feas_dir=self.unknown_feas_dir,
                                                                        labels_dir=self.unknown_labels_dir,
                                                                        feature_extractor=self.vgg_feature)
        else:
            raise RuntimeError('输入的特征类别有误!')


def test_vgg(first=False):
    test = ClassifiedByModels('VGG')
    if first:
        test.get_model()
        start = time.time()
        test.image2features()
        end = time.time()
        used_time = int(end - start)
        print('VGG提取特征使用时间是 %d 分, %d 秒.' % (used_time / 60, used_time % 60))
    else:
        test.load_feas()
    activeList = ['cor', 'che', 'l2']
    for active in activeList:
        print('度量向量间的距离选择：', active)
        test.get_all_distance(active)
        test.classify()
        test.evaluting()



if __name__ == '__main__':
    start = time.time()
    test_vgg(True)
    # test_googlenet(True)
    end = time.time()
    used_time = int(end - start)
    print('VGG使用时间是 %d 分, %d 秒.' % (used_time / 60, used_time % 60))

"""
图像特征提取完毕
提取图像特征中.....
图像特征提取完毕
VGG提取特征使用时间是 2 分, 20 秒.
度量向量间的距离选择： cor
K = 1 时的分类错误率为 0.0800 K = 1 时的分类正确率为 0.9200
K = 2 时的分类错误率为 0.0800 K = 2 时的分类正确率为 0.9200
K = 3 时的分类错误率为 0.1100 K = 3 时的分类正确率为 0.8900
K = 4 时的分类错误率为 0.1200 K = 4 时的分类正确率为 0.8800
K = 5 时的分类错误率为 0.1400 K = 5 时的分类正确率为 0.8600
度量向量间的距离选择： che
K = 1 时的分类错误率为 0.1300 K = 1 时的分类正确率为 0.8700
K = 2 时的分类错误率为 0.1300 K = 2 时的分类正确率为 0.8700
K = 3 时的分类错误率为 0.1500 K = 3 时的分类正确率为 0.8500
K = 4 时的分类错误率为 0.1500 K = 4 时的分类正确率为 0.8500
K = 5 时的分类错误率为 0.1600 K = 5 时的分类正确率为 0.8400
度量向量间的距离选择： l2
K = 1 时的分类错误率为 0.0800 K = 1 时的分类正确率为 0.9200
K = 2 时的分类错误率为 0.0800 K = 2 时的分类正确率为 0.9200
K = 3 时的分类错误率为 0.1200 K = 3 时的分类正确率为 0.8800
K = 4 时的分类错误率为 0.1100 K = 4 时的分类正确率为 0.8900
K = 5 时的分类错误率为 0.1400 K = 5 时的分类正确率为 0.8600
VGG使用时间是 2 分, 23 秒.
"""

