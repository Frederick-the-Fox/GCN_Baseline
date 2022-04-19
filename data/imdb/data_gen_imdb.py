import numpy as np
import scipy.sparse 
from sklearn.preprocessing import OneHotEncoder
import pickle as pkl
import sys

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

path = '/home/hangni/HeCo-main/data/imdb/'

#MAM
mam = np.load(path + 'mam.npz')
MAM = []
MAM.append(mam['row'])
MAM.append(mam['col'])
MAM = np.array(MAM)
MAM = MAM.transpose()
MAM = (MAM).astype(int)

#MKM
mkm = np.load(path + 'mkm.npz')
MKM = []
MKM.append(mkm['row'])
MKM.append(mkm['col'])
MKM = np.array(MKM)
MKM =MKM.transpose()
MKM = (MKM).astype(int)

#MDM
mdm = np.load(path + 'mdm.npz')
MDM = []
MDM.append(mdm['row'])
MDM.append(mdm['col'])
MDM = np.array(MDM)
MDM =MDM.transpose()
MDM = (MDM).astype(int)

#feature
feat = np.load(path + 'm_feat.npz')
feature = scipy.sparse.csr_matrix((feat['data'],feat['indices'], feat['indptr']), shape=(feat['shape'][0], feat['shape'][1]))
feature = (feature.A).astype(int)



#label
labels = np.load(path + 'labels.npy')


f1 = open('imdb.content', 'w', encoding='utf-8')
for i in range(feature.shape[0]):
    f1.writelines('\n' + str(i))
    for each in feature[i]:
        f1.writelines('\t' + str(each))

np.savetxt('MAM.cites', MAM, delimiter = '\t', fmt='%d')
np.savetxt('MKM.cites', MKM, delimiter = '\t', fmt='%d')
np.savetxt('MDM.cites', MDM, delimiter = '\t', fmt='%d')

f2 = open('imdb.label', 'w', encoding='utf-8')
for i in range(labels.shape[0]):
    f2.writelines(str(i) + '\t' + str(labels[i]) + '\n')

