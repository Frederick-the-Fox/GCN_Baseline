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

path = '/home/hangni/HeCo-main/data/dblp/'

#APA
apa = np.load(path + 'apa.npz')
APA = []
APA.append(apa['row'])
APA.append(apa['col'])
APA = np.array(APA)
APA = APA.transpose()
APA = (APA).astype(int)

#APCPA
apcpa = np.load(path + 'apcpa.npz')
APCPA = []
APCPA.append(apcpa['row'])
APCPA.append(apcpa['col'])
APCPA = np.array(APCPA)
APCPA =APCPA.transpose()
APCPA = (APCPA).astype(int)

#APTPA
aptpa = np.load(path + 'aptpa.npz')
APTPA = []
APTPA.append(aptpa['row'])
APTPA.append(aptpa['col'])
APTPA = np.array(APTPA)
APTPA =APTPA.transpose()
APTPA = (APTPA).astype(int)

#feature
feat = np.load(path + 'a_feat.npz')
feature = scipy.sparse.csr_matrix((feat['data'],feat['indices'], feat['indptr']), shape=(feat['shape'][0], feat['shape'][1]))
feature = (feature.A).astype(int)

#label
labels = np.load(path + 'labels.npy')


f1 = open('dblp.content', 'w', encoding='utf-8')
for i in range(feature.shape[0]):
    f1.writelines('\n' + str(i))
    for each in feature[i]:
        f1.writelines('\t' + str(each))

np.savetxt('APA.cites', APA, delimiter = '\t', fmt='%d')
np.savetxt('APCPA.cites', APCPA, delimiter = '\t', fmt='%d')
np.savetxt('APTPA.cites', APTPA, delimiter = '\t', fmt='%d')

f2 = open('dblp.label', 'w', encoding='utf-8')
for i in range(labels.shape[0]):
    f2.writelines(str(i) + '\t' + str(labels[i]) + '\n')

    

#saving
# file_data = {'APA':APA, 'APCPA':APCPA, 'APTPA':APTPA, 'feature':feature, 'label':label,
#             'test_idx_20':test_idx_20, 'train_idx_20':train_idx_20, 'val_idx_20':val_idx_20, 
#             'test_idx_40':test_idx_40, 'train_idx_40':train_idx_40, 'val_idx_40':val_idx_40, 
#             'test_idx_60':test_idx_60, 'train_idx_60':train_idx_60, 'val_idx_60':val_idx_60, 
#             'test_idx_eval':test_idx_eval, 'train_idx_eval':train_idx_eval, 'val_idx_eval':val_idx_eval
#             } 
# pkl.dump(file_data, open('dblp_ne w.pkl',"wb"), protocol=4)