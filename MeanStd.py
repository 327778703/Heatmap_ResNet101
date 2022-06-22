# -*- coding: utf-8 -*-
# 平均值和标准差已经通过计算得到结果

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

class MeanStd():
    def Getmean_std(self):
        train_mean = tf.constant([141.97156, 128.29279, 116.32985], shape=(3,), dtype=tf.float32)
        train_std = tf.constant([69.89145, 71.5751, 74.1216], shape=(3,), dtype=tf.float32)
        return train_mean, train_std