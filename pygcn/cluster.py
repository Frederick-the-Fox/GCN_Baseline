import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
import numpy
import sys
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score

embeds = numpy.load(sys.argv[1] + sys.argv[2] + '.npy')
X = embeds
y = numpy.load("/home/hangni/HeCo-main/data/" + sys.argv[1] + "/labels.npy")
n_clusters = int(sys.argv[3])

kmeans = KMeans(n_clusters, random_state=9999)
pre_y = kmeans.fit(embeds).predict(embeds)

y = y.astype(int)
print (y.dtype)

score = normalized_mutual_info_score(y, pre_y)
s2 = adjusted_rand_score(y, pre_y)
ho = homogeneity_score(y, pre_y)
print('NMI: {:.4f} , ARI : {:.4f}, HO :{:.4f}'.format(score, s2, ho))


# ## 模型可视化##
# tsne = TSNE(n_components=2, random_state=3747, init='pca', n_iter=500)
# X_2d = tsne.fit_transform(X)

# print("X_2d:{}".format(X_2d.shape))
# centers = kmeans.cluster_centers_
# 颜色设置
# colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2']

# fig = plt.figure()
# #创建3d绘图区域
# ax = plt.axes(projection='3d')

# # 循环读类别
# for i in range(n_clusters):
#     # 找到相同的索引
#     index_sets = numpy.where(pre_y == i)
#     # 将相同类的数据划分为一个聚类子集
#     cluster = X[index_sets]
#     # 展示样本点
#     ax.scatter3D(cluster[:, 0], cluster[:, 1], cluster[:, 2], c=colors[i], marker='o', alpha=0.3)
#     print('now the cluster is:{} and the number of the nodes of this cluster is:{}'.format(i, index_sets))

# ax.set_title('3d Scatter plot')
# plt.axis('off')
# ax.view_init(elev = 53, azim = 45)

# plt.savefig('test_dblp_53_45_new_0.3_cir.png', dpi=800)
# print('saved')