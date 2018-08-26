import numpy as np

"""A = np.load('/mnt/nfs/takagi/Mastercorse_program/Picture_future_world/inverter_LDS/USING/LDS_param/params_55/A_param.npy')
C = np.load('/mnt/nfs/takagi/Mastercorse_program/Picture_future_world/inverter_LDS/USING/LDS_param/params_55/C_param.npy')
Q = np.load('/mnt/nfs/takagi/Mastercorse_program/Picture_future_world/inverter_LDS/USING/LDS_param/params_55/Q_param.npy')
R = np.load('/mnt/nfs/takagi/Mastercorse_program/Picture_future_world/inverter_LDS/USING/LDS_param/params_55/R_param.npy')
pi_1 = np.load('/mnt/nfs/takagi/Mastercorse_program/Picture_future_world/inverter_LDS/USING/LDS_param/params_55/pi_1_param.npy')
V_1 = np.load('/mnt/nfs/takagi/Mastercorse_program/Picture_future_world/inverter_LDS/USING/LDS_param/params_55/V_1_param.npy')"""

A = np.load('/opt/pfw/takagi/LDS_param/params_55/A_param.npy')
C = np.load('/opt/pfw/takagi/LDS_param/params_55/C_param.npy')
Q = np.load('/opt/pfw/takagi/LDS_param/params_55/Q_param.npy')
R = np.load('/opt/pfw/takagi/LDS_param/params_55/R_param.npy')
pi_1 = np.load('/opt/pfw/takagi/LDS_param/params_55/pi_1_param.npy')
V_1 = np.load('/opt/pfw/takagi/LDS_param/params_55/V_1_param.npy')

sample_Z = []

#サンプルしたい時系列の数(各自で調整)
T =10

#LDSの学習済みのパラメータから，各時刻のGANの入力ノイズがサンプルされる．
for t in range(T):
    if(t == 0):
        x = np.random.multivariate_normal((pi_1.T)[0], V_1)
        z = np.random.multivariate_normal(np.dot(C, x), R)
        sample_Z.append(z)
    
    elif(t > 0):
        x = np.random.multivariate_normal(np.dot(A, x), Q)
        z = np.random.multivariate_normal(np.dot(C, x), R)
        sample_Z.append(z)

print(sample_Z)