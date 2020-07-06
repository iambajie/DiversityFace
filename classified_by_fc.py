import os
import time

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import optimizers
from keras.layers import Dropout
from matplotlib.ticker import FuncFormatter

from base import base
from classified_by_models import ClassifiedByModels

#迁移模型
class ClassifiedByFC(base):
    def __init__(self, type):
        super().__init__(type)
        time_string = time.strftime("%m-%d-%H-%M-%S", time.localtime())
        model_name = self.type + time_string
        self.model_path = os.path.join(self.root_dir, model_name)

    def continuous_labels(self, known_labels, unknown_labels):
        index = 0
        hash_map = {}
        known_list = []
        unknown_list = []

        for label in known_labels:
            if label not in hash_map.keys():
                hash_map[label] = index
                index += 1
            known_list.append(hash_map[label])

        for label in unknown_labels:
            if label not in hash_map.keys():
                raise RuntimeError('%d is not found', label)
            unknown_list.append(hash_map[label])

        known_labels = np.array(known_list)
        unknown_labels = np.array(unknown_list)
        return known_labels, unknown_labels

    def load_feas(self):
        if os.path.exists(self.known_feas_dir) and os.path.exists(self.known_labels_dir) and os.path.exists(
                self.unknown_feas_dir) and os.path.exists(self.unknown_labels_dir):
            super().load_feas()
            self.known_labels, self.unknown_labels = self.continuous_labels(self.known_labels, self.unknown_labels)
            self.known_labels = to_categorical(self.known_labels)
            self.unknown_labels = to_categorical(self.unknown_labels)
        else:
            raise RuntimeError('无法获得特征文件, 使用classified_by_models提取特征.')

    def get_model(self):
        self.model = Sequential()
        if self.type is 'VGGFC':
            self.model.add(Dense(1024, input_dim=4096, activation='relu'))
            self.model.add(Dropout(0.4))
            self.model.add(Dense(100, activation='softmax'))
            opt=optimizers.adam(lr=0.001,beta_1=0.9,beta_2=0.999,decay=1e-4)
            self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            self.model.summary()
        else:
            raise RuntimeError('参数错误.')

    def train_model(self):
        if self.type is 'VGGFC':
            self.history = self.model.fit(x=self.known_feas, y=self.known_labels,
                                          epochs=20, batch_size=8, validation_split=0.1, shuffle=True)
        else:
            raise RuntimeError('参数错误.')

    def classifiy(self):
        self.scores = self.model.evaluate(x=self.unknown_feas, y=self.unknown_labels, batch_size=32, verbose=1)

    def evaluting(self):
        print("图像分类的%s: %.2f%%" % (self.model.metrics_names[1], self.scores[1] * 100))
        if self.scores[1] >= 0.99:
            self.model.save(self.model_path)

    def load_model(self, model_weight):
        self.model = load_model(model_weight)


def train_and_test_VGGFC():
    fc = ClassifiedByFC('VGGFC')
    fc.get_model()
    fc.load_feas()
    fc.train_model()
    fc.classifiy()
    fc.evaluting()


def test_VGGFC():
    fc = ClassifiedByFC('VGGFC')
    fc.load_model()
    fc.load_feas()
    fc.classifiy()
    fc.evaluting()


if __name__ == '__main__':


    start = time.time()
    train_and_test_VGGFC()
    end = time.time()
    used_time = int(end - start)
    print('VGGFC使用时间是 %d 分, %d 秒.' % (used_time / 60, used_time % 60))

'''
VGGFC使用时间是 0 分, 57秒.
图像分类的accuracy: 86%
'''