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

path = '/home/hangni/HeCo-main/data/freebase/'

#MAM
mam = np.load(path + 'mam.npz')
MAM = []
MAM.append(mam['row'])
MAM.append(mam['col'])
MAM = np.array(MAM)
MAM = MAM.transpose()
MAM = (MAM).astype(int)

#MDM
mdm = np.load(path + 'mdm.npz')
MDM = []
MDM.append(mdm['row'])
MDM.append(mdm['col'])
MDM = np.array(MDM)
MDM =MDM.transpose()
MDM = (MDM).astype(int)

#MWM
mwm = np.load(path + 'mwm.npz')
MWM = []
MWM.append(mwm['row'])
MWM.append(mwm['col'])
MWM = np.array(MWM)
MWM =MWM.transpose()
MWM = (MWM).astype(int)

#feature
feature = scipy.sparse.eye(3492)
feature = feature.A
feature = (feature).astype(int)



#label
labels = np.load(path + 'labels.npy')


f1 = open('freebase.content', 'w', encoding='utf-8')
for i in range(feature.shape[0]):
    f1.writelines('\n' + str(i))
    for each in feature[i]:
        f1.writelines('\t' + str(each))

np.savetxt('MAM.cites', MAM, delimiter = '\t', fmt='%d')
np.savetxt('MDM.cites', MDM, delimiter = '\t', fmt='%d')
np.savetxt('MWM.cites', MWM, delimiter = '\t', fmt='%d')

f2 = open('freebase.label', 'w', encoding='utf-8')
for i in range(labels.shape[0]):
    f2.writelines(str(i) + '\t' + str(labels[i]) + '\n')

