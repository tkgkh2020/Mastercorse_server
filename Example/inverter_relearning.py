import tensorflow as tf
import numpy as np
import random as ra
import os
import glob
from numpy.random import *
import time

# 計算グラフ
g2 = tf.Graph()
with g2.as_default():
    # プレースホルダー
    x_ = tf.placeholder(tf.float32, shape=(None, 128, 128, 3))
    y_ = tf.placeholder(tf.float32, shape=(None, 100))
    keep_prob = tf.placeholder(tf.float32)

    # 畳み込み層1
    with tf.name_scope('conv1'):
        conv1_features = 40 # 畳み込み層1の出力次元数
        max_pool_size1 = 2 # 畳み込み層1のマックスプーリングサイズ
        conv1_w = tf.Variable(tf.truncated_normal([5, 5, 3, conv1_features], stddev=0.1), dtype=tf.float32, name='conv1_w') # 畳み込み層1の重み(初期値)
        conv1_b = tf.Variable(tf.constant(0.1, shape=[conv1_features]), dtype=tf.float32, name='conv1_b') # 畳み込み層1のバイアス(初期値)
        conv1_c2 = tf.nn.conv2d(x_, conv1_w, strides=[1, 1, 1, 1], padding="SAME", name='conv1_conv2d') # 畳み込み層1-畳み込み
        conv1_relu = tf.nn.relu(conv1_c2+conv1_b, name='conv1_ReLU') #畳み込み層1-ReLU
        conv1_drop = tf.nn.dropout(conv1_relu, keep_prob, name='conv1_dropout')#畳み込み層1-ドロップアウト
        conv1_mp = tf.nn.max_pool(conv1_drop, ksize=[1, max_pool_size1, max_pool_size1, 1], strides=[1, max_pool_size1, max_pool_size1, 1], padding="SAME", name='conv1_max_polling') #畳み込み層1-マックスプーリング

    # 畳み込み層2
    with tf.name_scope('conv2'):
        conv2_features = 80 # 畳み込み層2の出力次元数
        max_pool_size2 = 2 # 畳み込み層2のマックスプーリングのサイズ
        conv2_w = tf.Variable(tf.truncated_normal([5, 5, conv1_features, conv2_features], stddev=0.1), dtype=tf.float32, name='conv2_w') # 畳み込み層2の重み
        conv2_b = tf.Variable(tf.constant(0.1, shape=[conv2_features]), dtype=tf.float32, name='conv2_b') # 畳み込み層2のバイアス
        conv2_c2 = tf.nn.conv2d(conv1_mp, conv2_w, strides=[1, 1, 1, 1], padding="SAME", name='conv2_conv2d') # 畳み込み層2-畳み込み
        conv2_relu = tf.nn.relu(conv2_c2+conv2_b, name='conv2_ReLU') # 畳み込み層2-ReLU
        conv2_drop = tf.nn.dropout(conv2_relu, keep_prob, name='conv2_dropout')#畳み込み層2-ドロップアウト
        conv2_mp = tf.nn.max_pool(conv2_drop, ksize=[1, max_pool_size2, max_pool_size2, 1], strides=[1, max_pool_size2, max_pool_size2, 1], padding="SAME", name='conv2_max_polling') # 畳み込み層2-マックスプーリング

    # 畳み込み層3
    with tf.name_scope('conv3'):
        conv3_features = 160 # 畳み込み層3の出力次元数
        max_pool_size3 = 2 # 畳み込み層3のマックスプーリングのサイズ
        conv3_w = tf.Variable(tf.truncated_normal([5, 5, conv2_features, conv3_features], stddev=0.1), dtype=tf.float32, name='conv3_w') # 畳み込み層3の重み
        conv3_b = tf.Variable(tf.constant(0.1, shape=[conv3_features]), dtype=tf.float32, name='conv3_b') # 畳み込み層3のバイアス
        conv3_c2 = tf.nn.conv2d(conv2_mp, conv3_w, strides=[1, 1, 1, 1], padding="SAME", name='conv3_conv2d') # 畳み込み層3-畳み込み
        conv3_relu = tf.nn.relu(conv3_c2+conv3_b, name='conv3_ReLU') # 畳み込み層3-ReLU
        conv3_drop = tf.nn.dropout(conv3_relu, keep_prob, name='conv3_dropout')#畳み込み層3-ドロップアウト
        conv3_mp = tf.nn.max_pool(conv3_drop, ksize=[1, max_pool_size3, max_pool_size3, 1], strides=[1, max_pool_size3, max_pool_size3, 1], padding="SAME", name='conv3_max_polling') # 畳み込み層3-マックスプーリング
    
    # 畳み込み層4
    with tf.name_scope('conv4'):
        conv4_features = 320 # 畳み込み層4の出力次元数
        max_pool_size4 = 2 # 畳み込み層4のマックスプーリングのサイズ
        conv4_w = tf.Variable(tf.truncated_normal([5, 5, conv3_features, conv4_features], stddev=0.1), dtype=tf.float32, name='conv4_w') # 畳み込み層4の重み
        conv4_b = tf.Variable(tf.constant(0.1, shape=[conv4_features]), dtype=tf.float32, name='conv4_b') # 畳み込み層4のバイアス
        conv4_c2 = tf.nn.conv2d(conv3_mp, conv4_w, strides=[1, 1, 1, 1], padding="SAME", name='conv4_conv2d') # 畳み込み層4-畳み込み
        conv4_relu = tf.nn.relu(conv4_c2+conv4_b, name='conv4_ReLU') # 畳み込み層4-ReLU
        conv4_drop = tf.nn.dropout(conv4_relu, keep_prob, name='conv4_dropout')#畳み込み層4-ドロップアウト
        conv4_mp = tf.nn.max_pool(conv4_drop, ksize=[1, max_pool_size4, max_pool_size4, 1], strides=[1, max_pool_size4, max_pool_size4, 1], padding="SAME", name='conv4_max_polling') # 畳み込み層4-マックスプーリング

    # 畳み込み層5
    with tf.name_scope('conv5'):
        conv5_features = 640 # 畳み込み層5の出力次元数
        max_pool_size5 = 2 # 畳み込み層5のマックスプーリングのサイズ
        conv5_w = tf.Variable(tf.truncated_normal([5, 5, conv4_features, conv5_features], stddev=0.1), dtype=tf.float32, name='conv5_w') # 畳み込み層5の重み
        conv5_b = tf.Variable(tf.constant(0.1, shape=[conv5_features]), dtype=tf.float32, name='conv5_b') # 畳み込み層5のバイアス
        conv5_c2 = tf.nn.conv2d(conv4_mp, conv5_w, strides=[1, 1, 1, 1], padding="SAME", name='conv5_conv2d') # 畳み込み層5-畳み込み
        conv5_relu = tf.nn.relu(conv5_c2+conv5_b, name='conv5_ReLU') # 畳み込み層5-ReLU
        conv5_drop = tf.nn.dropout(conv5_relu, keep_prob, name='conv5_dropout')#畳み込み層5-ドロップアウト
        conv5_mp = tf.nn.max_pool(conv5_drop, ksize=[1, max_pool_size5, max_pool_size5, 1], strides=[1, max_pool_size5, max_pool_size5, 1], padding="SAME", name='conv5_max_polling') # 畳み込み層5-マックスプーリング

    # 畳み込み層6
    with tf.name_scope('conv6'):
        conv6_features = 1280 # 畳み込み層6の出力次元数
        max_pool_size6 = 2 # 畳み込み層6のマックスプーリングのサイズ
        conv6_w = tf.Variable(tf.truncated_normal([5, 5, conv5_features, conv6_features], stddev=0.1), dtype=tf.float32, name='conv6_w') # 畳み込み層6の重み
        conv6_b = tf.Variable(tf.constant(0.1, shape=[conv6_features]), dtype=tf.float32, name='conv6_b') # 畳み込み層6のバイアス
        conv6_c2 = tf.nn.conv2d(conv5_mp, conv6_w, strides=[1, 1, 1, 1], padding="SAME", name='conv6_conv2d') # 畳み込み層6-畳み込み
        conv6_relu = tf.nn.relu(conv6_c2+conv6_b, name='conv6_ReLU') # 畳み込み層6-ReLU
        conv6_drop = tf.nn.dropout(conv6_relu, keep_prob, name='conv6_dropout')#畳み込み層6-ドロップアウト
        conv6_mp = tf.nn.max_pool(conv6_drop, ksize=[1, max_pool_size6, max_pool_size6, 1], strides=[1, max_pool_size6, max_pool_size6, 1], padding="SAME", name='conv6_max_polling') # 畳み込み層6-マックスプーリング
        
    
    # 全結合層1
    with tf.name_scope('fully1'):
        result_w = x_.shape[1] // (max_pool_size1*max_pool_size2*max_pool_size3*max_pool_size4*max_pool_size5*max_pool_size6)
        result_h = x_.shape[2] // (max_pool_size1*max_pool_size2*max_pool_size3*max_pool_size4*max_pool_size5*max_pool_size6)
        fc_input_size = result_w * result_h * conv6_features # 畳み込んだ結果、全結合層に入力する次元数
        fc_features = 1000 # 全結合層1の出力次元数（隠れ層の次元数）
        s = conv6_mp.get_shape().as_list() # [None, result_w, result_h, conv4_features]
        conv_result = tf.reshape(conv6_mp, [-1, s[1]*s[2]*s[3]]) # 畳み込みの結果を1*N層に変換
        fc1_w = tf.Variable(tf.truncated_normal([fc_input_size.value, fc_features], stddev=0.1), dtype=tf.float32, name='fully1_w') # 重み
        fc1_b = tf.Variable(tf.constant(0.1, shape=[fc_features]), dtype=tf.float32, name='fully1_b') # バイアス
        fc1 = tf.nn.relu(tf.matmul(conv_result, fc1_w)+fc1_b, name='fully1_ReLU') # 全結合層1
        fc1_drop = tf.nn.dropout(fc1, keep_prob, name='fully1_dropout') #ドロップアウト
    
    # 全結合層2
    with tf.name_scope('fully2'):
        fc2_features = 1000 # 全結合層2の出力次元数（隠れ層の次元数）
        fc2_w = tf.Variable(tf.truncated_normal([fc_features, fc2_features], stddev=0.1), dtype=tf.float32, name='fully2_w') # 重み
        fc2_b = tf.Variable(tf.constant(0.1, shape=[fc2_features]), dtype=tf.float32, name='fully2_b') # バイアス
        fc2 = tf.nn.relu(tf.matmul(fc1_drop, fc2_w)+fc2_b, name='fully2_ReLU') # 全結合層2
        fc2_drop = tf.nn.dropout(fc2, keep_prob, name='fully2_dropout')#ドロップアウト

    # 全結合層3
    with tf.name_scope('fully3'):
        fc3_features = 500 # 全結合層3の出力次元数（隠れ層の次元数）
        fc3_w = tf.Variable(tf.truncated_normal([fc2_features, fc3_features], stddev=0.1), dtype=tf.float32, name='fully3_w') # 重み
        fc3_b = tf.Variable(tf.constant(0.1, shape=[fc3_features]), dtype=tf.float32, name='fully3_b') # バイアス
        fc3 = tf.nn.relu(tf.matmul(fc2_drop, fc3_w)+fc3_b, name='fully3_ReLU') # 全結合層3
        fc3_drop = tf.nn.dropout(fc3, keep_prob, name='fully3_dropout')#ドロップアウト
    
    # 全結合層4
    with tf.name_scope('fully4'):
        fc4_features = 100 # 全結合層3の出力次元数（隠れ層の次元数）
        fc4_w = tf.Variable(tf.truncated_normal([fc3_features, fc4_features], stddev=0.1), dtype=tf.float32, name='fully4_w') # 重み
        fc4_b = tf.Variable(tf.constant(0.1, shape=[fc4_features]), dtype=tf.float32, name='fully4_b') # バイアス
        y = tf.matmul(fc3_drop, fc4_w)+fc4_b

    # 平均2乗和誤差(square_error)
    with tf.name_scope('loss'):
        loss = tf.reduce_sum(tf.square(y_ - y), name='loss')
    
    # 勾配法
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # グラフを保存する．
    saver = tf.train.Saver()


#データ読み込み&データ生成(バッチによる読み込み)の関数
def batch(batch_size):
    #batch_size = 60

    #再構成した画像のPATHのリスト
    path_rec = "/opt/pfw/dragon_ball_reconsted/reconsted/disc11/story1/*.npy"
    filelists_rec = glob.glob(path_rec, recursive=True)

    #zのPATHのリスト
    path_z = "/opt/pfw/dragon_ball_reconsted/z/disc11/story1/*.npy"
    filelists_z = glob.glob(path_z, recursive=True)

    #サンプルしてくるインデックスのリスト
    ind = ra.sample(range(len(filelists_rec)), batch_size)

    #サンプルしてくるインデックスの各PATHを格納したリスト
    batch_images_rec = [filelists_rec[i] for i in ind ]
    batch_images_z = [filelists_z[i] for i in ind ]

    #再構成した画像とzの各バッチ
    batch_x = np.array([np.load(f).tolist() for f in batch_images_rec]) #再構成画像のファイル読み込み
    batch_z = np.array([np.load(f).tolist() for f in batch_images_z]) #zのファイル読み込み
    
    return batch_x, batch_z
    
    
#実行フェーズ
start = time.time()

n_epochs = 50000000
training_loss = []
eps = 0.01
batch_size = 60
batch_x, batch_z = batch(batch_size)
kp_p = 1.0

epo = 23000

with tf.Session(graph=g2) as sess:
    #実行フェーズでの値の復元
    saver.restore(sess, './savepoint/inverter_graph/train_epoch_'+str(epo)+'/trained-model')
    
    # TensorBoardで追跡する変数を定義
    with tf.name_scope('summary'):
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logsdata/inverter_rel_graph', sess.graph)
    
    #500エポックでモデルをトレーニング
    for e in range(n_epochs):
        c, _, summary = sess.run([loss, train_step, merged], feed_dict={x_: batch_x, y_: batch_z, keep_prob: 0.5})
        writer.add_summary(summary, e)
        training_loss.append(c)
        
        e = epo+e+1
        print('Epoch %4d: %.4f' % (e, c), '\t', 'time %4d' %(time.time()-start))
        
        if not e % 50:
            batch_x, batch_z = batch(batch_size)
        
        if(e > 0 and c < eps):
            saver.save(sess, './savepoint/inverter_rel_graph/train_epoch_'+str(e)+'/trained-model')
            break
        
        if(e % 1000 == 0):
            saver.save(sess, './savepoint/inverter_rel_graph/train_epoch_'+str(e)+'/trained-model')
        e = e-epo-1