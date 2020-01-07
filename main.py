# -*- coding: utf-8 -*-
# ----------------------------------------------------
# Copyright (c) 2017, Wray Zheng. All Rights Reserved.
# Distributed under the BSD License.
# ----------------------------------------------------

from gmm import *
import pandas as pd

# 设置调试模式
DEBUG = True

# 载入数据
data = pd.read_csv('log/dataset.csv', header=0, index_col=0)
Y = np.array(data)
matY = np.matrix(Y, copy=True)


# 模型个数，即聚类的类别个数
K = 10

# 计算 GMM 模型参数
mu, cov, alpha = GMM_EM(matY, K, 100)

# 根据 GMM 模型，对样本数据进行聚类，一个模型对应一个类别
N = Y.shape[0]
# 求当前模型参数下，各模型对样本的响应度矩阵
gamma = getExpectation(matY, mu, cov, alpha)
# 对每个样本，求响应度最大的模型下标，作为其类别标识
category = gamma.argmax(axis=1).flatten().tolist()[0]
# 将每个样本放入对应类别的列表中
classes = []
for k in range(K):
    classes.append(np.array([Y[i] for i in range(N) if category[i] == k]))


# 绘制聚类结果
for i in range(K):
    plt.plot(classes[i][:, 0], classes[i][:, 1], 'o', label='class{}'.format(i))
# plt.plot(class1[:, 0], class1[:, 1], 'rs', label="class1")
# plt.plot(class2[:, 0], class2[:, 1], 'bo', label="class2")
# plt.plot(class3[:, 0], class3[:, 1], 'o', label="class3")
# plt.plot(class4[:, 0], class4[:, 1], 'o', label="class4")
# plt.plot(class5[:, 0], class5[:, 1], 'o', label="class5")

plt.legend(loc="best")
plt.title("GMM Clustering By EM Algorithm")
plt.show()

# label = pd.read_csv('label.csv', index_col=0)
# classes1 = []
# for k in range(K):
#     classes1.append(np.array([Y[i] for i in range(N) if label[i] == k]))
# for i in range(K):
#     plt.plot(classes1[i][:, 0], classes1[i][:, 1], 'o', label='class{}'.format(i))
# plt.title("ground truth")
# plt.show()


category = pd.DataFrame(category)
category.to_csv('log/prediction.csv')
print('finish')
