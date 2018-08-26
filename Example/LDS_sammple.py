import numpy as np

A = np.loadtxt('/mnt/nfs/takagi/Mastercorse_program/Picture_future_world/inverter_LDS/LDS_param/A_pram.npy')
C = np.loadtxt('/mnt/nfs/takagi/Mastercorse_program/Picture_future_world/inverter_LDS/LDS_param/C_pram.npy')
Q = np.loadtxt('/mnt/nfs/takagi/Mastercorse_program/Picture_future_world/inverter_LDS/LDS_param/Q_pram.npy')
R = np.loadtxt('/mnt/nfs/takagi/Mastercorse_program/Picture_future_world/inverter_LDS/LDS_param/R_pram.npy')
pi_1 = np.loadtxt('/mnt/nfs/takagi/Mastercorse_program/Picture_future_world/inverter_LDS/LDS_param/pi_1_pram.npy')
V_1 = np.loadtxt('/mnt/nfs/takagi/Mastercorse_program/Picture_future_world/inverter_LDS/LDS_param/V_1_pram.npy')

sample_Z = []

#サンプルしたい時系列の数(各自で調整)
T =10

#LDSの学習済みのパラメータから，各時刻のGANの入力ノイズがサンプルされる．
for t in range(T):
    if(t == 0):
        x = np.random.multivariate_normal((pi_1.T)[0], V_1)
        z = np.random.multivariate_normal(np.dot(C, x), R)
        sample_Z.append(y)
    
    elif(t > 0):
        x = np.random.multivariate_normal(np.dot(A, x), Q)
        z = np.random.multivariate_normal(np.dot(C, x), R)
        sample_Z.append(y)
