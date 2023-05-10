import numpy as np
from scipy.stats import binom
from scipy.io import loadmat
from tqdm import trange

data = np.load('../data/8mer.npy')
datanor = np.load('../data/8mernor.npy')

# data = loadmat(r'D:\matlab\bin\5mer.mat')['lncRNA5mer655']
# datanor = loadmat(r'D:\matlab\bin\5mernor.mat')['lnc5mer655nor']

# m_c = np.sum(data[:426])
# m_e = np.sum(data[426:456])
# m_n = np.sum(data[456:612])
# m_r = np.sum(data[612:])


m_0 = np.sum(data[:25])
m_1 = np.sum(data[26:353])
m_2 = np.sum(data[354:853])
m_3 = np.sum(data[854:877])
m_4 = np.sum(data[878:974])
m_5 = np.sum(data[975:1008])
m_6 = np.sum(data[1009:1037])
m_7 = np.sum(data[1038:1229])

M = m_0 + m_1 + m_2 + m_3 + m_4 + m_5 + m_6 + m_7
#print('M: ',M)
q_0 = m_0 / M
q_1 = m_1 / M
q_2 = m_2 / M
q_3 = m_3 / M
q_4 = m_4 / M
q_5 = m_5 / M
q_6 = m_6 / M
q_7 = m_7 / M
Q = [q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7]
#print('Q: ',Q)
# ni_c = np.sum(data[:426],axis=0)
# ni_e = np.sum(data[426:456],axis=0)
# ni_n = np.sum(data[456:612],axis=0)
# ni_r = np.sum(data[612:],axis=0)

ni_0 = np.sum(data[:25], axis=0)
ni_1 = np.sum(data[26:353], axis=0)
ni_2 = np.sum(data[354:853], axis=0)
ni_3 = np.sum(data[854:877], axis=0)
ni_4 = np.sum(data[878:974], axis=0)
ni_5 = np.sum(data[975:1008], axis=0)
ni_6 = np.sum(data[1009:1037], axis=0)
ni_7 = np.sum(data[1038:1229], axis=0)

W = [ni_0, ni_1, ni_2, ni_3, ni_4, ni_5, ni_6, ni_7]
W = np.array(W)
#print('W.shape: ', W.shape)
W = W.T
G = np.sum(data, axis=0)
PP = []
fea_len = data.shape[1]
for i in trange(fea_len,colour = 'green'):
    '''if i % 10000 == 0:
        print('第', i, '正在进行')'''
    P = []
    for j in range(8):
        sum = 0
        for k in np.arange(W[i][j], G[i] + 1):
            sum += binom.pmf(k, G[i], Q[j])
        P.append(sum)
    PP.append(P)
PP = np.array(PP)
#print('PP: ',PP[0])
CL = 1 - PP
#print('CL: ',CL[0])
max_CL = np.max(CL, axis=1)
max_CL = max_CL.reshape(1, max_CL.shape[0]).T
index = np.argmax(CL, axis=1)
index = index.reshape(1, index.shape[0]).T
Cli = np.hstack((max_CL, index))
Climax = Cli[:, 0]
Climax = Climax.reshape(1, Climax.shape[0]).T
Feorder = np.arange(0, fea_len).reshape(1, fea_len).T
CLimax_oder = np.hstack((Climax, Feorder))
CLimax_oder = CLimax_oder[np.argsort(-CLimax_oder[:, 0])]
CLoder8 = CLimax_oder[:, 1]
#np.save('CLoder8',CLoder8)
np.savetxt('CLorder.csv', CLoder8, delimiter=',')
lnc8mernorCL = []
lnc8mernorCL = np.array(lnc8mernorCL)
#print(type(lnc8mernorCL))
for i in trange(fea_len, colour = 'green'):
    if i % 10000 == 0:
        print(f'feature dimension is {lnc8mernorCL.shape(1)}')
    E = datanor[:, int(CLoder8[i])]
    E = E.reshape(len(E), 1)
    if i == 0:
        lnc8mernorCL = E
    else:
        lnc8mernorCL = np.c_[lnc8mernorCL, E]
np.save(r'../data/8merbino', arr=lnc8mernorCL)
print(np.shape(lnc8mernorCL),lnc8mernorCL[0])