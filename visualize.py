
import os
import cv2
import numpy as np
import sklearn.decomposition as dp
import imageio
from skimage import color
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


image_path='./used/000_0.bmp'


#颜色直方图可视化
img = cv2.imread(image_path)
color = ('b','g','r')
line=['-','--','-.']
label=['blue','green','red']
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None, [256], [0., 255.])
    plt.plot(histr,color = col,linestyle=line[i],label=label[i])
    plt.xlim([0,256])

plt.legend()
plt.grid(linestyle='-.') 
plt.show()


#pca特征脸可视化
def get_Image(folder):
    file_list = os.listdir(folder)
    for file in file_list:
        labels.append(int(file[0:3]))  # 保存每个特征对应的类别
        img_path = os.path.join(folder, file)  # 读取图像进行特征提取
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h,w = img_gray.shape
        img_col = img_gray.reshape(h*w)
        feas.append(img_col)
    return h,w
 
feas = []
labels = []
h,w = get_Image("./used")
 
X = np.array(feas)  
y = np.array(labels)

n_components = 20
 
pca = PCA(n_components=n_components, svd_solver='randomized', #选择一种svd方式
          whiten=True).fit(X)    #whiten是一种数据预处理方式，会损失一些数据信息，但可获得更好的预测结果
 
eigenfaces = pca.components_.reshape((n_components, h, w))    #特征脸
 
X_pca = pca.transform(X)      #得到训练集投影系数

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
 
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
 
plt.show()






